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

"""Library for computing metrics."""

import jax.numpy as jnp


def vectors_almost_equal(
    vector_1: jnp.ndarray, vector_2: jnp.ndarray, tolerance: float = 1e-1
):
  """Returns whether all elements of vectors are almost equal."""
  abs_diff = jnp.abs(vector_1 - vector_2)
  max_diff = jnp.max(abs_diff, axis=1)
  return max_diff <= tolerance


def vector_elements_almost_equal(
    vector_1: jnp.ndarray, vector_2: jnp.ndarray, tolerance: float = 1e-1
):
  """Returns boolean vec indicating whether input vec elements are almost equal."""
  abs_diff = jnp.abs(vector_1 - vector_2)
  return abs_diff <= tolerance


def get_vector_element_accuracy(
    predictions: jnp.ndarray, targets: jnp.ndarray, tolerance: float = 1e-1
):
  """Returns fraction of predicted vector elements that are correct."""
  return jnp.mean(vector_elements_almost_equal(predictions, targets, tolerance))


def get_vector_accuracy(
    predictions: jnp.ndarray, targets: jnp.ndarray, tolerance: float = 1e-1
):
  """Returns fraction of predicted vectors that are correct."""
  return jnp.mean(vectors_almost_equal(predictions, targets, tolerance))
