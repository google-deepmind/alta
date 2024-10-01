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

"""Library for serializing and deserializing model parameters."""

import json
import jax.numpy as jnp
from tensorflow.io import gfile


def save_params(path: str, params: list[tuple[jnp.ndarray, jnp.ndarray]]):
  """Saves model parameters to given path."""
  serializable_data = [
      ([weights.tolist(), biases.tolist()]) for weights, biases in params
  ]
  json_string = json.dumps(serializable_data)
  with gfile.GFile(path, "w") as fp:
    fp.write(json_string)


def load_params(path: str) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
  """Loads model parameters from given path."""
  with gfile.GFile(path, "rb") as fp:
    json_data = json.loads(fp.read())
    params = [
        (jnp.array(json_weights), jnp.array(json_biases))
        for json_weights, json_biases in json_data
    ]
    return params
