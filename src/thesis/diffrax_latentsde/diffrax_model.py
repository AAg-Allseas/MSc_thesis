import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import jax.random as jr
import mlflow

from thesis.diffrax_latentsde.utils import Encoder
from thesis.diffrax_latentsde.vector_fields import (
    ContextStateField,
    ContextTimeStateField,
    FieldConfig,
    StateField,
    TimeStateField,
    init_vector_field,
)

mlflow.enable_system_metrics_logging()


class LatentSDE(eqx.Module):
    encoder: Encoder
    f: ContextTimeStateField | ContextStateField | TimeStateField | StateField | jnp.ndarray  # Prior drift
    h: ContextTimeStateField | ContextStateField | TimeStateField | StateField | jnp.ndarray  # Control drift
    g: ContextTimeStateField | ContextStateField | TimeStateField | StateField | jnp.ndarray  # Diffusion
    qz0_posterior: eqx.nn.Linear
    pz0_mean: jnp.ndarray
    """
    Latent SDE model implementation using Diffrax and Equinox.

    This module defines the LatentSDE class for modeling latent stochastic differential equations (SDEs)
    with context-dependent drift and diffusion, suitable for time series modeling and generative tasks.

    References:
        - Equinox: https://github.com/patrick-kidger/equinox
        - Diffrax: https://github.com/patrick-kidger/diffrax
    """
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
        f_config: FieldConfig,
        h_config: FieldConfig,
        g_config: FieldConfig,
        *,
        key,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        (enc_key, qz0_key, dec_key, mean_key, logvar_key) = (
            jr.split(key, 5)
        )

        self.data_size = data_size
        self.latent_size = latent_size

        self.encoder = Encoder(data_size, hidden_size, context_size, key=enc_key)
        self.qz0_posterior = eqx.nn.Linear(context_size, 2 * latent_size, key=qz0_key)

        if f_config.context_size != context_size:
            raise ValueError(
                f"f_config context_size {f_config.context_size} does not match encoder context_size {context_size}"
            )
        if f_config.latent_size != latent_size:
            raise ValueError(
                f"f_config latent_size {f_config.latent_size} does not match latent_size {latent_size}"
            )
        if h_config.latent_size != latent_size:
            raise ValueError(
                f"h_config latent_size {h_config.latent_size} does not match latent_size {latent_size}"
            )
        if g_config.latent_size != latent_size:
            raise ValueError(
                f"g_config latent_size {g_config.latent_size} does not match latent_size {latent_size}"
            )

        self.f = init_vector_field(f_config)
        self.h = init_vector_field(h_config)
        self.g = init_vector_field(g_config)

        self.decoder = eqx.nn.MLP(latent_size, data_size, hidden_size, 1, key=dec_key)

        self.pz0_mean = jr.normal(mean_key, (latent_size,)) * 0.1
        self.pz0_logvar = jr.uniform(
            logvar_key, (latent_size,), minval=-0.5, maxval=0.5
        )

        self.log_sigma = jnp.full((data_size,), -1.0)

    def __call__(
        self, xs: jnp.ndarray, ts: jnp.ndarray, *, key
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ Compute negative ELBO for given data xs and time points ts.
         Args:
            xs: Observed data, shape (T, data_size).
            ts: Time points corresponding to xs, shape (T,).
            key: JAX random key for sampling.

         Returns:
            Tuple of (log_likelihood, kl_divergence_initial, kl_divergence_path).
            log_likelihood: Scalar log p(x|z) averaged over time.
            kl_divergence_initial: Scalar KL divergence between q(z0|x) and p(z0).
            kl_divergence_path: Scalar KL divergence accumulated along the SDE path.
        """
        bm_key, z0_key = jr.split(key)

        # Encode context (reverse time GRU, flip back)
        ctx = jnp.flip(self.encoder(jnp.flip(xs, axis=0)), axis=0)  # (T, ctx_size)

        # SDE integration setup
        t0, t1, dt0 = ts[0], ts[-1], 1.0
        bm = diffrax.VirtualBrownianTree(
            t0, t1, tol=dt0 / 2, shape=(self.g.control_size + 1,), key=bm_key
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

        return log_pxs, logqp0, logqp_path

    def sample(self, ts: jnp.ndarray, *, key) -> jnp.ndarray:
        """Sample from the prior SDE (no observations needed).
        Returns decoded trajectory, shape (T, data_size).
        """
        bm_key, z0_key = jr.split(key)

        t0, t1, dt0 = ts[0], ts[-1], 1.0
        bm = diffrax.VirtualBrownianTree(
            t0, t1, tol=dt0 / 2, shape=(self.g.control_size,), key=bm_key
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
        """Compute drift and KL integrand for augmented state [z, kl_accumulator].
        KL drift is based on Girsanov's theorem for non-diagonal diffusion.

        Let f = prior drift, h = posterior drift, g = diffusion matrix (latent_size, control_size).
        The instantaneous KL divergence rate is:
            u^2 = 0.5 * (f - h)^T @ (g g^T)^{-1} @ (f - h)
        where g g^T is the diffusion covariance matrix (latent_size, latent_size).
        """
        y_state = y[..., :-1]  # Strip KL accumulator
        z_f = self.f(t, y_state, args)
        z_h = self.h(t, y_state, args)
        G = self.g(t, y_state, args)

        delta = z_f - z_h  # (latent_size,)
        eps = 1e-6  # Regularization for numerical stability
        GtG = G.T @ G                          # (m, m)
        rhs = G.T @ delta                      # (m,)
        u = jnp.linalg.solve(GtG + eps*jnp.eye(GtG.shape[0]), rhs)  # (m,)
        kl_rate = 0.5 * (u @ u)                # scalar
        
        return jnp.append(z_f, kl_rate)

    def diffusion(self, t: jnp.ndarray, y: jnp.ndarray, args) -> jnp.ndarray:
        y_state = y[..., :-1]
        g_val = self.g(t, y_state, args)
        # Diagonal matrix for augmented state: [g(z), 0]
        return jax.scipy.linalg.block_diag(g_val, jnp.zeros(1))
