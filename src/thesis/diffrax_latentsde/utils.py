import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import mlflow


mlflow.enable_system_metrics_logging()


def lipswish(x):
    return 0.909 * jnn.silu(x)


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
