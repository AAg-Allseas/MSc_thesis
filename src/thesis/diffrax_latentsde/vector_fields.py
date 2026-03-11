"""
Vector field modules for latent SDEs using Diffrax and Equinox.

Implements neural vector fields that can condition on time, state, and context for use in latent SDE models.

References:
    - https://github.com/patrick-kidger/equinox
    - https://github.com/patrick-kidger/diffrax

"""

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any
import os

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


from optax import scale
from thesis.diffrax_latentsde.utils import lipswish


class FieldType(Enum):
    TIME_STATE = "time_state"
    STATE = "state"
    CONTEXT_TIME_STATE = "context_time_state"
    CONTEXT_STATE = "context_state"
    STATIC = "static"


@dataclass
class FieldConfig:
    field_type: FieldType = FieldType.CONTEXT_TIME_STATE
    latent_size: int = 32
    hidden_layer_width: int = 64
    depth: int = 2
    scale: bool = True
    context_size: int = 32
    control_size: int = 1
    key: jax.Array = field(default_factory=lambda: jr.key(time.time_ns()))




class TimeStateField(eqx.Module):
    """
    Vector field that conditions on time and state.

    Attributes:
        scale: Scaling factor for the output, can be a scalar or array.
        mlp: Equinox MLP module for the vector field.
    """

    scale: int | jnp.ndarray
    mlp: eqx.nn.MLP
    control_size: int = eqx.field(static=True)

    def __init__(self, config: FieldConfig, *, key, **kwargs):
        """
        Initializes a TimeStateField.

        Args:
            config: FieldConfig instance containing initialization parameters.
            key: JAX PRNG key for initialization.
            **kwargs: Additional arguments for eqx.Module.
        """
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if config.scale:
            if config.control_size > 1:
                scale_shape = (config.latent_size, config.control_size)
            else:
                scale_shape = (config.latent_size,)
            self.scale = jr.uniform(
                scale_key, scale_shape, minval=0.9, maxval=1.1
            )
        else:
            self.scale = 1
        self.control_size = config.control_size

        self.mlp = eqx.nn.MLP(
            in_size=config.latent_size + 1,
            out_size=config.latent_size * config.control_size,
            width_size=config.hidden_layer_width,
            depth=config.depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )

    def __call__(self, t: jnp.ndarray, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        """
        Evaluates the vector field at a given time and state.

        Args:
            t: Time (scalar or array).
            y: State vector.
            args: Unused, for API compatibility.

        Returns:
            Output of the vector field (same shape as y).
        """
        if os.environ.get("DEBUG_PRINT", "0") == "1":
            print("[DEBUG] JIT compiling: TimeStateField.__call__")

        t = jnp.asarray(t)
        out = self.mlp(jnp.concatenate([t[None], y])).reshape(-1, self.control_size)
        if self.control_size == 1:
            out = out.squeeze(axis=-1)
        return self.scale *  out


class StateField(eqx.Module):
    """
    Vector field that conditions on state only.

    Attributes:
        scale: Scaling factor for the output, can be a scalar or array.
        mlp: Equinox MLP module for the vector field.
    """

    scale: int | jnp.ndarray
    mlp: eqx.nn.MLP
    control_size: int = eqx.field(static=True)

    def __init__(self, config: FieldConfig, *, key, **kwargs):
        """
        Initializes a StateField.

        Args:
            config: FieldConfig instance containing initialization parameters.
            key: JAX PRNG key for initialization.
            **kwargs: Additional arguments for eqx.Module.
        """
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if config.scale:
            if config.control_size > 1:
                scale_shape = (config.latent_size, config.control_size)
            else:
                scale_shape = (config.latent_size,)
            self.scale = jr.uniform(
                scale_key, scale_shape, minval=0.9, maxval=1.1
            )
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=config.latent_size,
            out_size=config.latent_size * config.control_size,
            width_size=config.hidden_layer_width,
            depth=config.depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )
        self.control_size = config.control_size

    def __call__(self, t: jnp.ndarray, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        import os

        """
        Evaluates the vector field at a given state.

        Args:
            t: Time (unused, for API compatibility).
            y: State vector.
            args: Unused, for API compatibility.

        Returns:
            Output of the vector field (same shape as y).
        """
        if os.environ.get("DEBUG_PRINT", "0") == "1":
            print("[DEBUG] JIT compiling: StateField.__call__")

        out = self.mlp(y).reshape(-1, self.control_size)
        if self.control_size == 1:
            out = out.squeeze(axis=-1)
        return self.scale * out


class ContextTimeStateField(eqx.Module):
    """
    Vector field that conditions on time, state, and context (e.g. encoder output).

    Attributes:
        scale: Scaling factor for the output, can be a scalar or array.
        mlp: Equinox MLP module for the vector field.
    """

    scale: jnp.ndarray | int
    mlp: eqx.nn.MLP
    control_size: int = eqx.field(static=True)

    def __init__(self, config: FieldConfig, *, key, **kwargs) -> None:
        """
        Initializes a ContextTimeStateField.

        Args:
            config: FieldConfig instance containing initialization parameters.
            key: JAX PRNG key for initialization.
            **kwargs: Additional arguments for eqx.Module.
        """
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if config.scale:
            if config.control_size > 1:
                scale_shape = (config.latent_size, config.control_size)
            else:
                scale_shape = (config.latent_size,)
            self.scale = jr.uniform(
                scale_key, scale_shape, minval=0.9, maxval=1.1
            )
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=config.latent_size + config.context_size + 1,  # [t, y, ctx]
            out_size=config.latent_size * config.control_size,
            width_size=config.hidden_layer_width,
            depth=config.depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )
        self.control_size = config.control_size

    def __call__(self, t: jnp.ndarray, y: jnp.ndarray, args: Any) -> jnp.ndarray:

        if os.environ.get("DEBUG_PRINT", "0") == "1":
            print("[DEBUG] JIT compiling: ContextTimeStateField.__call__")
        """
        Evaluates the vector field at a given time, state, and context.

        Args:
            t: Time (scalar or array).
            y: State vector.
            args: Tuple of (ts, ctx), where ts is a time array and ctx is a context array.

        Returns:
            Output of the vector field (same shape as y or output_size).
        """
        t = jnp.asarray(t)
        ts, ctx = args
        i = jnp.minimum(jnp.searchsorted(ts, t, side="right"), ts.shape[0] - 1)
        out = self.mlp(jnp.concatenate([t[None], y, ctx[i]])).reshape(-1, self.control_size)
        if self.control_size == 1:
            out = out.squeeze(axis=-1)
        return self.scale * out


class ContextStateField(eqx.Module):
    """
    Vector field that conditions on time, state, and context (e.g. encoder output).

    Attributes:
        scale: Scaling factor for the output, can be a scalar or array.
        mlp: Equinox MLP module for the vector field.
    """

    scale: jnp.ndarray | int
    mlp: eqx.nn.MLP
    control_size: int = eqx.field(static=True)

    def __init__(self, config: FieldConfig, *, key, **kwargs) -> None:
        """
        Initializes a ContextTimeStateField.

        Args:
            config: FieldConfig instance containing initialization parameters.
            key: JAX PRNG key for initialization.
            **kwargs: Additional arguments for eqx.Module.
        """
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if config.scale:
            if config.control_size > 1:
                scale_shape = (config.latent_size, config.control_size)
            else:
                scale_shape = (config.latent_size,)
            self.scale = jr.uniform(
                scale_key, scale_shape, minval=0.9, maxval=1.1
            )
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=config.latent_size + config.context_size,  # [y, ctx]
            out_size=config.latent_size * config.control_size,
            width_size=config.hidden_layer_width,
            depth=config.depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )
        self.control_size = config.control_size

    def __call__(self, t: jnp.ndarray, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        """
        Evaluates the vector field at a given time, state, and context.

        Args:
            t: Time (scalar or array).
            y: State vector.
            args: Tuple of (ts, ctx), where ts is a time array and ctx is a context array.

        Returns:
            Output of the vector field (same shape as y or output_size).
        """ 
        if os.environ.get("DEBUG_PRINT", "0") == "1":
            print("[DEBUG] JIT compiling: ContextStateField.__call__")

        ts, ctx = args
        i = jnp.minimum(jnp.searchsorted(ts, t, side="right"), ts.shape[0] - 1)
        out = self.mlp(jnp.concatenate([y, ctx[i]])).reshape(-1, self.control_size)
        if self.control_size == 1:
            out = out.squeeze(axis=-1)
        return self.scale * out




def init_vector_field(
    config: FieldConfig,
    ) -> (
        ContextTimeStateField
        | ContextStateField
        | TimeStateField
        | StateField
        | jnp.ndarray
    ):
    match config.field_type:
        case FieldType.CONTEXT_TIME_STATE:
            return ContextTimeStateField(config, key=config.key)
        
        case FieldType.CONTEXT_STATE:
            return ContextStateField(config, key=config.key)
        
        case FieldType.TIME_STATE:
            return TimeStateField(config, key=config.key)
        
        case FieldType.STATE:
            return StateField(config, key=config.key)
        
        case FieldType.STATIC:
            return jr.uniform(
                config.key,
                (config.latent_size, config.control_size),
                minval=-1,
                maxval=1,
            )