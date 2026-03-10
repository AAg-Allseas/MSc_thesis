import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax  # https://github.com/deepmind/optax
import mlflow

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from thesis.prototyping.dataloader import ParquetDataset
from thesis.prototyping.data_handling import find_parquet_files

mlflow.enable_system_metrics_logging()


def lipswish(x):
    return 0.909 * jnn.silu(x)


class VectorField(eqx.Module):
    scale: int | jnp.ndarray
    mlp: eqx.nn.MLP

    def __init__(self, hidden_size, width_size, depth, scale, *, key, **kwargs):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(scale_key, (hidden_size,), minval=0.9, maxval=1.1)
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )

    def __call__(self, t, y, args):
        t = jnp.asarray(t)
        return self.scale * self.mlp(jnp.concatenate([t[None], y]))


class PosteriorField(eqx.Module):
    """Posterior drift that conditions on encoder context passed via args=(ts, ctx)."""

    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __init__(
        self, latent_size, context_size, width_size, depth, scale, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(scale_key, (latent_size,), minval=0.9, maxval=1.1)
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=latent_size + context_size + 1,  # [t, y, ctx]
            out_size=latent_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )

    def __call__(self, t, y, args):
        t = jnp.asarray(t)
        ts, ctx = args
        i = jnp.minimum(jnp.searchsorted(ts, t, side="right"), ts.shape[0] - 1)
        return self.scale * self.mlp(jnp.concatenate([t[None], y, ctx[i]]))


class Encoder(eqx.Module):
    gru: eqx.nn.GRUCell
    lin: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, data_size: int, hidden_size: int, ctx_size: int, *, key) -> None:
        gru_key, lin_key = jr.split(key)
        self.hidden_size = hidden_size
        self.gru = eqx.nn.GRUCell(
            input_size=data_size, hidden_size=hidden_size, key=gru_key
        )
        self.lin = eqx.nn.Linear(hidden_size, ctx_size, key=lin_key)

    def __call__(self, xs: jnp.ndarray) -> jnp.ndarray:
        """Encode sequence xs: (T, data_size) -> (T, ctx_size)."""

        def step(h, x):
            h = self.gru(x, h)
            return h, self.lin(h)

        _, ctx = jax.lax.scan(step, jnp.zeros(self.hidden_size), xs)
        return ctx


class LatentSDE(eqx.Module):
    encoder: Encoder
    f: PosteriorField  # Posterior drift (uses context via args)
    h: VectorField  # Prior drift
    g: VectorField  # Shared diagonal diffusion
    qz0_posterior: eqx.nn.Linear
    pz0_mean: jnp.ndarray
    pz0_logvar: jnp.ndarray
    log_sigma: jnp.ndarray
    decoder: eqx.nn.MLP
    latent_size: int = eqx.field(static=True)
    data_size: int = eqx.field(static=True)

    def __init__(
        self,
        data_size: int,
        latent_size: int,
        context_size: int,
        hidden_size: int,
        *,
        key,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        (enc_key, qz0_key, f_key, h_key, g_key, dec_key, mean_key, logvar_key) = (
            jr.split(key, 8)
        )

        self.data_size = data_size
        self.latent_size = latent_size

        self.encoder = Encoder(data_size, hidden_size, context_size, key=enc_key)
        self.qz0_posterior = eqx.nn.Linear(context_size, 2 * latent_size, key=qz0_key)

        self.f = PosteriorField(
            latent_size, context_size, hidden_size, 4, scale=True, key=f_key
        )
        self.h = VectorField(latent_size, hidden_size, 4, scale=True, key=h_key)
        self.g = VectorField(latent_size, hidden_size, 4, scale=True, key=g_key)

        self.decoder = eqx.nn.MLP(latent_size, data_size, hidden_size, 1, key=dec_key)

        self.pz0_mean = jr.normal(mean_key, (latent_size,)) * 0.1
        self.pz0_logvar = jr.uniform(
            logvar_key, (latent_size,), minval=-0.5, maxval=0.5
        )

        self.log_sigma = jnp.full((data_size,), -1.0)

    def __call__(
        self, xs: jnp.ndarray, ts: jnp.ndarray, *, key
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        bm_key, z0_key = jr.split(key)

        # Encode context (reverse time GRU, flip back)
        ctx = jnp.flip(self.encoder(jnp.flip(xs, axis=0)), axis=0)  # (T, ctx_size)

        # SDE integration setup
        t0, t1, dt0 = ts[0], ts[-1], 1.0
        bm = diffrax.VirtualBrownianTree(
            t0, t1, tol=dt0 / 2, shape=(self.latent_size + 1,), key=bm_key
        )

        drift_term = diffrax.ODETerm(self.drift)
        diffusion_term = diffrax.ControlTerm(self.diffusion, bm)
        sde = diffrax.MultiTerm(drift_term, diffusion_term)
        solver = diffrax.Euler()
        saveat = diffrax.SaveAt(ts=ts)

        # Sample initial condition from posterior
        qz0_mean, qz0_logstd = jnp.split(self.qz0_posterior(ctx[0]), 2, axis=-1)
        qz0_logstd = jnp.clip(qz0_logstd, -5.0, 2.0)
        z0 = qz0_mean + jnp.exp(qz0_logstd) * jr.normal(z0_key, shape=qz0_mean.shape)

        # Augmented initial state: [z0, 0] (KL accumulator starts at 0)
        z0_aug = jnp.concatenate([z0, jnp.zeros(1)])

        # Context passed via args so PosteriorField can access it (no mutation)
        sol = diffrax.diffeqsolve(
            sde,
            solver,
            t0,
            t1,
            dt0,
            z0_aug,
            saveat=saveat,
            args=(ts, ctx),
            max_steps=ts.shape[0] * 10,
        )
        zs = sol.ys[..., :-1]  # (T, latent_size)
        logqp_path = sol.ys[-1, -1]  # Accumulated path KL

        # Decode latent path to observation space
        xs_hat = jax.vmap(self.decoder)(zs)  # (T, data_size)

        # --- Log-likelihood of observations ---
        sigma = jnp.exp(jnp.clip(self.log_sigma, -4.0, 2.0))
        log_pxs = (
            jax.scipy.stats.norm.logpdf(xs, xs_hat, sigma).sum(axis=-1).mean()
        )  # sum features, mean time

        # --- KL divergence of initial condition ---
        pz0_logstd = jnp.clip(self.pz0_logvar * 0.5, -5.0, 2.0)
        kl_z0 = (
            pz0_logstd
            - qz0_logstd
            + (jnp.exp(2 * qz0_logstd) + (qz0_mean - self.pz0_mean) ** 2)
            / (2 * jnp.exp(2 * pz0_logstd))
            - 0.5
        )
        logqp0 = kl_z0.sum(axis=-1)  # sum over latent dims

        return log_pxs, logqp0 + logqp_path

    def sample(self, ts: jnp.ndarray, *, key) -> jnp.ndarray:
        """Sample from the prior SDE (no observations needed).
        Returns decoded trajectory, shape (T, data_size).
        """
        bm_key, z0_key = jr.split(key)

        t0, t1, dt0 = ts[0], ts[-1], 1.0
        bm = diffrax.VirtualBrownianTree(
            t0, t1, tol=dt0 / 2, shape=(self.latent_size,), key=bm_key
        )

        def prior_drift(t, y, args):
            return self.h(t, y, args)

        def prior_diffusion(t, y, args):
            return jnp.diag(self.g(t, y, args))

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(prior_drift),
            diffrax.ControlTerm(prior_diffusion, bm),
        )
        solver = diffrax.Euler()
        saveat = diffrax.SaveAt(ts=ts)

        # Sample z0 from prior
        pz0_std = jnp.exp(jnp.clip(self.pz0_logvar * 0.5, -5.0, 2.0))
        z0 = self.pz0_mean + pz0_std * jr.normal(z0_key, shape=self.pz0_mean.shape)

        sol = diffrax.diffeqsolve(
            terms, solver, t0, t1, dt0, z0, saveat=saveat, max_steps=ts.shape[0] * 10
        )
        return jax.vmap(self.decoder)(sol.ys)  # (T, data_size)

    def drift(self, t: jnp.ndarray, y: jnp.ndarray, args) -> jnp.ndarray:
        y_state = y[..., :-1]  # Strip KL accumulator
        z_f = self.f(t, y_state, args)
        z_h = self.h(t, y_state, args)
        z_g = self.g(t, y_state, args)
        u = (z_f - z_h) / jnp.clip(jnp.abs(z_g), min=1e-7)
        return jnp.concatenate([z_f, jnp.array([0.5 * jnp.sum(u**2)])])

    def diffusion(self, t: jnp.ndarray, y: jnp.ndarray, args) -> jnp.ndarray:
        y_state = y[..., :-1]
        g_val = self.g(t, y_state, args)
        # Diagonal matrix for augmented state: [g(z), 0]
        return jnp.diag(jnp.concatenate([g_val, jnp.zeros(1)]))
