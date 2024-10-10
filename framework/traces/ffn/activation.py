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

"""Library of activation functions."""

from collections.abc import Callable
import jax
import jax.numpy as jnp


def relu(x: jnp.ndarray) -> jnp.ndarray:
  """Relu activation functino."""
  return jnp.maximum(0, x)


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
  """Sigmoid activation function."""
  return jax.scipy.special.expit(x)


def get_activation_fn(
    activation_fn_name: str,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Returns activation function with the given name."""
  if activation_fn_name == "sigmoid":
    return sigmoid
  elif activation_fn_name == "relu":
    return relu
  elif activation_fn_name == "tanh":
    return jnp.tanh
  else:
    raise ValueError(f"Unknown activation function: {activation_fn_name}")
