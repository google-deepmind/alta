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

"""Library for training FFN."""

import dataclasses
import functools
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from alta.framework.traces.ffn import inference


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
  """Configuration for training FFN."""

  train_examples_path: str
  test_examples_path: str
  vector_length: int

  layer_sizes: list[int]
  learning_rate: float
  # Linear warmup for learning rate.
  warmup_steps: int

  num_steps: int
  batch_size: int
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray]
  initialization_fn: Callable[
      [int, int, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
  ]
  optimization_fn: Callable[
      [optax.ScalarOrSchedule | None], optax.GradientTransformation
  ]
  noise_stddev: float | None

  eval_period: int
  eval_size: int
  # How many steps between writing basic stats, e.g. loss.
  stats_period: int

  # How many steps between writing checkpoints.
  checkpoint_period: int | None
  # Directory to write checkpoints and metrics to.
  output_dir: str


def l2_loss(params, activation_fn, inputs, targets):
  """Returns L2 loss."""
  predictions = inference.batched_predict(params, inputs, activation_fn)
  squared_error = jnp.mean(jnp.square(predictions - targets))

  return squared_error


@functools.partial(
    jax.jit,
    # Pass immutable objects as static arguments.
    static_argnames=("activation_fn", "optimizer"),
)
def update(params, activation_fn, optimizer, opt_state, x, y):
  """Performs single update step and returns updated params.

  Args:
    params: The model's parameters.
    activation_fn: The activation function to use inside the neural network.
    optimizer: The Optax optimizer class.
    opt_state: The state of the optimizer. State is maintained separately
      because Optax optimizers are implemented using pure functions.
    x: The neural network input.
    y: The neural network target.

  Returns:
    A tuple of the loss, gradients, updated model parameters, and optimizer
    state.
  """
  loss, grads = jax.value_and_grad(l2_loss)(params, activation_fn, x, y)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)

  return loss, grads, params, opt_state
