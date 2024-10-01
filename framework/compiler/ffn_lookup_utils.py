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

"""Utilities for FFN layers that implement a lookup table.

We generate 4-layer FNNs. The first 2 layers "expand" numeric variables
to a discretized one-hot representation. The final 2 layers implement a lookup
table that maps from different combinations of input variables to different
outputs. The utilities in this module implement this lookup table.
"""

import dataclasses

import numpy as np

from framework import program
from framework.compiler import dim_utils
from framework.mlp import mlp_rules


@dataclasses.dataclass(frozen=True)
class LookupParams:
  weights_1: np.ndarray
  bias_1: np.ndarray
  weights_2: np.ndarray
  bias_2: np.ndarray


def _get_start_idx(var_mapping: dim_utils.VarDimMapping):
  """Returns start index of variable in dimension mapping."""
  if isinstance(var_mapping, dim_utils.CategoricalVarDimMapping):
    return var_mapping.start_idx
  elif isinstance(var_mapping, dim_utils.ExpandedNumericalVarDimMapping):
    return var_mapping.start_idx
  elif isinstance(var_mapping, dim_utils.NumericalVarDimMapping):
    return var_mapping.idx
  elif isinstance(var_mapping, dim_utils.ExpandedSetVarDimMapping):
    return var_mapping.start_idx
  elif isinstance(var_mapping, dim_utils.SetVarDimMapping):
    return var_mapping.start_idx
  else:
    raise ValueError(f"Unsupported var mapping: {var_mapping}")


def _get_antecedent_vector(
    rule_lhs: mlp_rules.LHS,
    expanded_dim_mappings: dim_utils.VarDimMappings,
):
  """Returns antecedent vector for a given rule."""
  vector = np.zeros(
      expanded_dim_mappings.end_idx - expanded_dim_mappings.start_idx
  )
  for atom in rule_lhs:
    var_mapping = expanded_dim_mappings.var_mappings[atom.variable]
    assignment_idx = _get_start_idx(var_mapping) + atom.value_idx
    vector[assignment_idx] = 1.0
  return vector


def _assignment_to_vector(
    dim_mappings: dim_utils.VarDimMappings,
    var_spec: program.VarSpec,
    var_name: str,
    var_value: program.VarValue,
):
  """Gets vector encoding of a variable assignment."""
  output_vector = np.zeros(dim_mappings.end_idx - dim_mappings.start_idx)
  var_mapping = dim_mappings.var_mappings[var_name]
  if isinstance(var_spec, program.CategoricalVarSpec):
    assert isinstance(var_mapping, dim_utils.CategoricalVarDimMapping)
    assert isinstance(var_value, int)
    output_vector[var_mapping.start_idx + var_value] = 1.0
  elif isinstance(var_spec, program.NumericalVarSpec):
    assert isinstance(var_mapping, dim_utils.NumericalVarDimMapping)
    assert isinstance(var_value, float)
    output_vector[var_mapping.idx] = var_value
  elif isinstance(var_spec, program.SetVarSpec):
    assert isinstance(var_mapping, dim_utils.CategoricalVarDimMapping)
    assert isinstance(var_value, frozenset)
    for value_idx in var_value:
      output_vector[var_mapping.start_idx + value_idx] = 1.0
  return output_vector


def _get_consequent_vector(
    program_spec: program.Program,
    rule_rhs: mlp_rules.RHS,
    dim_mappings: dim_utils.VarDimMappings,
):
  """Returns consequent vector for a given rule."""
  vector = np.zeros(dim_mappings.end_idx - dim_mappings.start_idx)
  output_var_name = rule_rhs.variable
  output_var_spec = program_spec.variables[output_var_name]
  new_value = rule_rhs.new_value
  old_value = rule_rhs.old_value
  # Subtract the old value.
  # Note that for numerical variables there may be some residual non-zero
  # value equal to the difference between the input value and the closest
  # bucket center.
  vector -= _assignment_to_vector(
      dim_mappings, output_var_spec, output_var_name, old_value
  )
  # Add the new value unless we are resetting the variable to be null.
  if new_value is not None:
    vector += _assignment_to_vector(
        dim_mappings, output_var_spec, output_var_name, new_value
    )
  return vector


def build_lookup_params(
    program_spec: program.Program,
    dim_mappings: dim_utils.VarDimMappings,
    expanded_dim_mappings: dim_utils.VarDimMappings,
) -> LookupParams:
  """Builds the parameters for last two FFN layers."""
  weights_1_stack = []
  weights_2_stack = []
  bias_1_stack = []

  for rule in program_spec.mlp.get_rules():
    antecedent_vector = _get_antecedent_vector(rule.lhs, expanded_dim_mappings)
    consequent_vector = _get_consequent_vector(
        program_spec, rule.rhs, dim_mappings
    )
    num_constraints = len(rule.lhs)
    weights_1_stack.append(antecedent_vector)
    weights_2_stack.append(consequent_vector)
    bias_1_stack.append(1 - float(num_constraints))

  weights_1 = np.stack(weights_1_stack, axis=0)
  weights_1 = np.transpose(weights_1)
  weights_2 = np.stack(weights_2_stack, axis=0)

  bias_1 = np.stack(bias_1_stack, axis=0)
  bias_2 = np.zeros(weights_2.shape[1])

  return LookupParams(weights_1, bias_1, weights_2, bias_2)
