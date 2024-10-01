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

"""Helpful utilities for debugging."""

from typing import Any, Optional

import numpy as np

from alta.framework.compiler import dim_utils

# Allow some flexibility in the value.
THRESHOLD = 0.9


def _get_categorical_var_value(
    vector: np.ndarray,
    var_mapping: dim_utils.CategoricalVarDimMapping,
) -> Optional[int]:
  value = 0
  for idx in range(var_mapping.start_idx, var_mapping.end_idx):
    if vector[idx] > THRESHOLD:
      return value
    value += 1
  return None


def _get_set_var_value(
    vector: np.ndarray,
    var_mapping: dim_utils.CategoricalVarDimMapping,
) -> set[int]:
  """Returns the set of categorical values for a given vector and variable."""
  values = set()
  value = 0
  for idx in range(
      var_mapping.start_idx, var_mapping.end_idx - var_mapping.start_idx
  ):
    if vector[idx] > THRESHOLD:
      values.add(value)
    value += 1
  return values


def vector_to_variables(
    vector: np.ndarray,
    var_mappings: dim_utils.VarDimMappings,
) -> dict[str, Any]:
  """Converts a vector to a dictionary of variables."""
  variables = {}
  for var_name, var_mapping in var_mappings.var_mappings.items():
    if isinstance(var_mapping, dim_utils.NumericalVarDimMapping):
      variables[var_name] = vector[var_mapping.idx]
    elif isinstance(var_mapping, dim_utils.SetVarDimMapping):
      variables[var_name] = _get_set_var_value(vector, var_mapping)
    elif isinstance(var_mapping, dim_utils.CategoricalVarDimMapping):
      variables[var_name] = _get_categorical_var_value(vector, var_mapping)
    else:
      raise ValueError("Unsupported variable type.")
  return variables
