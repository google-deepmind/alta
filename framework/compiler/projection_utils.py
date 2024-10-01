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

"""Get parameters for attention heads and output transforms."""

from typing import List

import numpy as np

from alta.framework import program
from alta.framework.compiler import compiler_config
from alta.framework.compiler import dim_utils
from alta.framework.transformer import parameters


def _get_start_idx(var_mapping):
  if isinstance(var_mapping, dim_utils.NumericalVarDimMapping):
    return var_mapping.idx
  elif isinstance(var_mapping, dim_utils.CategoricalVarDimMapping):
    return var_mapping.start_idx
  elif isinstance(var_mapping, dim_utils.SetVarDimMapping):
    return var_mapping.start_idx
  else:
    raise ValueError(f"Unsupported var mapping: {var_mapping}")


def _get_end_idx(var_mapping):
  if isinstance(var_mapping, dim_utils.NumericalVarDimMapping):
    return var_mapping.idx + 1
  elif isinstance(var_mapping, dim_utils.CategoricalVarDimMapping):
    return var_mapping.end_idx
  elif isinstance(var_mapping, dim_utils.SetVarDimMapping):
    return var_mapping.end_idx
  else:
    raise ValueError(f"Unsupported var mapping: {var_mapping}")


def select_variable(
    var_mappings: dim_utils.VarDimMappings,
    var_name: str,
    scalar: float = 1.0,
):
  """Returns matrix that selects a variable."""
  var_mapping = var_mappings.var_mappings[var_name]
  start_idx = _get_start_idx(var_mapping)
  end_idx = _get_end_idx(var_mapping)
  var_dims = end_idx - start_idx

  total_dims = var_mappings.end_idx - var_mappings.start_idx
  mat = np.zeros([total_dims, var_dims])
  identity_mat = np.identity(var_dims) * scalar
  mat[
      start_idx:end_idx,
      :,
  ] = identity_mat
  return mat


def project_variable(
    var_mappings: dim_utils.VarDimMappings,
    var_name: str,
):
  """Returns matrix that projects a variable."""
  var_mapping = var_mappings.var_mappings[var_name]
  start_idx = _get_start_idx(var_mapping)
  end_idx = _get_end_idx(var_mapping)
  var_dims = end_idx - start_idx

  total_dims = var_mappings.end_idx - var_mappings.start_idx
  mat = np.zeros([var_dims, total_dims])
  identity_mat = np.identity(var_dims)
  mat[
      :,
      start_idx:end_idx,
  ] = identity_mat
  return mat


def _get_attention_head_params(
    head_spec: program.AttentionHeadSpec,
    var_mappings: dim_utils.VarDimMappings,
    config: compiler_config.Config,
) -> parameters.AttentionHeadParameters:
  """Return attention head parameters for a given head."""
  query_transform = select_variable(
      var_mappings,
      head_spec.query,
      scalar=config.attention_scalar,
  )
  key_transform = select_variable(
      var_mappings,
      head_spec.key,
      scalar=config.attention_scalar,
  )
  value_transform = select_variable(var_mappings, head_spec.value)
  output_transform = project_variable(var_mappings, head_spec.output)
  return parameters.AttentionHeadParameters(
      query_transform=query_transform,
      key_transform=key_transform,
      value_transform=value_transform,
      output_transform=output_transform,
      relative_position_mask=head_spec.relative_position_mask,
  )


def get_attention_params(
    program_spec: program.Program,
    var_mappings: dim_utils.VarDimMappings,
    config: compiler_config.Config,
) -> List[parameters.AttentionHeadParameters]:
  return [
      _get_attention_head_params(head_spec, var_mappings, config)
      for head_spec in program_spec.head_specs
  ]


def get_output_transform(
    program_spec: program.Program, var_mappings: dim_utils.VarDimMappings
) -> List[parameters.AttentionHeadParameters]:
  return select_variable(var_mappings, program_spec.outputs)
