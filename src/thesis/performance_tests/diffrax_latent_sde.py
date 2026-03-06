import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import mlflow

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

class PosteriorField(VectorField):
    def __init__(self, hidden_size, width_size, depth, scale, *, key, **kwargs):
        super().__init__(hidden_size, width_size, depth, scale, key=key, **kwargs)
        self._context: Optional[tuple[jnp.ndarray, jnp.ndarray]] = None

    @property
    def context(self) -> None:
        return self._context

    @context.setter
    def context(self, ts: jnp.ndarray, ctx: jnp.ndarray) -> None:
        self._context = (ts, ctx)

    def __call__(self, t, y, args) -> jnp.ndarray:
        t = jnp.asarray(t)
        ts, ctx = self.context
        i = min(jnp.searchsorted(ts, t, side="right"), len(ts) - 1)
        return super.__call__(t, jnp.concatenate([y, ctx[i]]))

class ControlledVectorField(eqx.Module):
    scale: int | jnp.ndarray
    mlp: eqx.nn.MLP
    control_size: int
    hidden_size: int

    def __init__(
        self, control_size, hidden_size, width_size, depth, scale, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(
                scale_key, (hidden_size, control_size), minval=0.9, maxval=1.1
            )
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size * control_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )
        self.control_size = control_size
        self.hidden_size = hidden_size

    def __call__(self, t, y, args):
        t = jnp.asarray(t)
        return self.scale * self.mlp(jnp.concatenate([t[None], y])).reshape(
            self.hidden_size, self.control_size
        )

class Encoder(eqx.Module):
    def __init__(self, data_size: int, hidden_size: int, ctx_size: int, *, key) -> None:
        gru_key, lin_key = jr.split(key)
        self.gru = eqx.nn.GRUCell(input_size=data_size, hidden_size=hidden_size, key=gru_key)
        self.lin = eqx.nn.Linear(hidden_size, ctx_size, key=lin_key)

    def __call__(self, data: jnp.ndarray) -> jnp.ndarray:
        return self.lin(self.gru(data))


class LatentSDE(eqx.Module):
    encoder: eqx.Module
    f: VectorField  # Posterior drift
    h: VectorField  # Prior drift
    g: ControlledVectorField  # Shared diffusion
    qz0_posterior: eqx.nn.MLP
    pz0_mean: jnp.ndarray  # Prior mean
    pz0_logvar: jnp.ndarray  # Prior log variance
    decoder: eqx.nn.MLP

    def __init__(self, data_size: int, latent_size: int, context_size: int, hidden_size: int, *, key, **kwargs) -> None:
        super().__init__(**kwargs)
        (enc_key, qz0_key, f_key, h_key, g_key, dec_key,
         mean_key, logvar_key) = jr.split(key, 8)

        self.encoder = Encoder(data_size, hidden_size, context_size, key=enc_key)
        self.qz0_posterior = eqx.nn.Linear(context_size, 2 * latent_size, key=qz0_key)

        self.f = PosteriorField(latent_size, hidden_size, 4, scale=True, key=f_key)
        self.h = VectorField(latent_size, hidden_size, 4, scale=True, key=h_key)
        self.g = ControlledVectorField(latent_size, latent_size, hidden_size, 2, scale=True, key=g_key)

        self.decoder = eqx.nn.MLP(latent_size, data_size, hidden_size, 1, key=dec_key)

        self.pz0_mean = jr.normal(mean_key, (latent_size,)) * 0.1
        self.pz0_logvar = jr.uniform(logvar_key, (latent_size,), minval=-0.5, maxval=0.5)

        self.sigma = 0.1

    def __call__(self, xs: jnp.ndarray, ts: jnp.ndarray, *, key) -> None:
        bm_key, z0_key = jr.split(key)

        # Encode context
        ctx = jnp.flip(self.encoder(jnp.flip(xs, axis=0)), axis=0)
        self.f.context = (ts, ctx)

        # Generate Brownian motion
        t0 = ts[0]
        t1 = ts[-1]
        dt0 = 1
        bm = diffrax.VirtualBrownianTree(t0, t1, tol=dt0 / 2, shape=(self.latent_size + 1,), key=bm_key)

        # Define terms
        drift_term = diffrax.ODETerm(self.drift)
        diffusion_term = diffrax.ControlTerm(self.diffusion)

        sde = diffrax.MultiTerm([drift_term, diffusion_term])
        solver = diffrax.SEA  # Assuming additive noise, cheapest solver. 
        saveat = diffrax.SaveAt(ts)

        qz0_mean, qz0_logstd = jnp.split(self.qz0_posterior(self.f.context[1]), 2, axis=1)
        z0 = qz0_mean + jnp.exp(qz0_logstd) * jr.normal(z0_key, shape=qz0_mean.shape)
        sol = diffrax.diffeqsolve(sde, solver, t0, t1, dt0, z0, saveat=saveat)
        logqp = sol[..., -1]
        ys = jax.vmap(self.decoder)(sol[..., :-1])

        return ys, logqp


    def drift(self, t: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        y = y[..., :-1] # Remove last KL Loss
        z_f = self.f(t, y)
        z_h = self.h(t, y)
        z_g = self.g(t, y)
        u = (z_f - z_h) / jnp.clip(z_g, min=1e-7)
        return jnp.concatenate([z_f, 0.5 * u ** 2])

    def diffusion(self, t: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        y = y[..., :-1]
        return jnp.concatenate([self.g(t, y), 0], )
    
def generate_ou_data(
    key,
    dim: int = 12,
    t0: float = 0.0,
    t1: float = 10.0,
    dt0: float = 0.01,
    theta: jnp.ndarray | None = None,
    mu: jnp.ndarray | None = None,
    sigma: jnp.ndarray | None = None,
):
    """Generate a single d-dimensional Ornstein-Uhlenbeck sample with per-dimension parameters.

    If theta/mu/sigma are None, random per-dimension values are drawn:
      - theta ~ LogUniform[0.001, 0.1]  (mean-reversion speed)
      - mu    ~ Uniform[-5, 5]          (long-run mean)
      - sigma ~ LogUniform[0.05, 1.0]   (volatility)
    """
    bm_key, y0_key, param_key = jr.split(key, 3)
    tk, mk, sk = jr.split(param_key, 3)

    if theta is None:
        theta = jnp.exp(jr.uniform(tk, (dim,), minval=jnp.log(0.001), maxval=jnp.log(0.1)))
    if mu is None:
        mu = jr.uniform(mk, (dim,), minval=-5.0, maxval=5.0)
    if sigma is None:
        sigma = jnp.exp(jr.uniform(sk, (dim,), minval=jnp.log(0.05), maxval=jnp.log(1.0)))

    def ou_drift(t, y, args):
        return theta * (mu - y)

    def ou_diffusion(t, y, args):
        return jnp.diag(sigma)

    bm = diffrax.VirtualBrownianTree(t0, t1, tol=dt0 / 2, shape=(dim,), key=bm_key)
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(ou_drift),
        diffrax.ControlTerm(ou_diffusion, bm),
    )
    solver = diffrax.Euler()
    saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, dt0))

    y0 = jr.normal(y0_key, (dim,))
    sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, saveat=saveat, max_steps=None)
    return sol.ts, sol.ys


def generate_ou_batch(
    key,
    batch_size: int = 16,
    **kwargs,
):
    """Generate a batch of OU processes. Returns (ts, ys) with ys shape (batch, steps, dim)."""
    keys = jr.split(key, batch_size)
    ts, ys = jax.vmap(lambda k: generate_ou_data(k, **kwargs))(keys)
    return ts[0], ys  # ts identical across batch


if __name__ == "__main__":
    key = jax.random.key(7777)
    ou_key, model_key, call_key = jr.split(key, 3)

    ts, xs = generate_ou_batch(ou_key, batch_size=16, t1=10800.0, dim=12)
    # xs shape: (16, num_steps, 12)

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    for i, ax in enumerate(axes.flat):
        for b in range(xs.shape[0]):
            ax.plot(ts, xs[b, :, i], alpha=0.3, linewidth=0.5)
        ax.set_title(f"Dim {i}")
        ax.set_ylabel("x")
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.suptitle("12-D Ornstein-Uhlenbeck (16 samples)")
    plt.tight_layout()
    plt.show()
    # latent_sde = LatentSDE(12, 4, 64, 128, key=model_key)