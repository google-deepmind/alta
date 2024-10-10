# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Library for initializing network parameters."""

from collections.abc import Callable
import jax
import jax.numpy as jnp
import numpy as np


# Defines type for model parameters.
LayerParameters = tuple[jnp.ndarray, jnp.ndarray]
Parameters = list[LayerParameters]
InitializationFn = Callable[[int, int, jax.Array], LayerParameters]


def random_layer_params(
    m: int, n: int, key: jax.Array, scale: float = 1e-2
) -> LayerParameters:
  """Initializes a layer randomly scaled by `scale`."""
  w_key, b_key = jax.random.split(key)
  return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(
      b_key, (n,)
  )


def xavier_normal_layer_params(m, n, key) -> LayerParameters:
  """Initializes a layer using Xavier initialization."""
  w_key, _ = jax.random.split(key)
  stddev = np.sqrt(2 / (m + n))
  return stddev * jax.random.normal(w_key, (n, m)), jnp.zeros(n)


def he_layer_params(m, n, key) -> LayerParameters:
  """Initializes a layer using He initialization."""
  w_key, _ = jax.random.split(key)
  stddev = jnp.sqrt(2 / m)
  return jax.random.normal(w_key, (n, m)) * stddev, jnp.zeros(n)


def init_network_params(
    sizes: list[int],
    key: jax.Array,
    layer_initialization_fn: InitializationFn,
) -> Parameters:
  """Initializes network parameters."""
  keys = jax.random.split(key, len(sizes))
  return [
      layer_initialization_fn(m, n, k)
      for m, n, k in zip(sizes[:-1], sizes[1:], keys)
  ]


def get_initialization_fn(initialization_fn_name: str) -> InitializationFn:
  """Returns initialization function with the given name."""
  if initialization_fn_name == "he_layer_params":
    return he_layer_params
  elif initialization_fn_name == "random_layer_params":
    return random_layer_params
  elif initialization_fn_name == "xavier_normal_layer_params":
    return xavier_normal_layer_params
  else:
    raise ValueError(
        f"Unknown initialization function: {initialization_fn_name}"
    )
