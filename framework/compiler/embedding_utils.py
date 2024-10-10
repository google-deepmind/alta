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

"""Get parameters for embeddings."""

import numpy as np

from framework import program
from framework.compiler import dim_utils
from framework.transformer import parameters


def variables_to_vector(
    attention_outputs: frozenset[str],
    variables: program.Activations,
    var_mappings: dim_utils.VarDimMappings,
) -> np.ndarray:
  """Return a vector encoding the given variables.

  Args:
    attention_outputs: Set of variable names that are attention outputs.
      Variables in `attention_outputs` are initialized to all zeroes. This is so
      that they can be "written" to easily by the first attention operation
      without being "overridden" by the residual connection.
    variables: Dict mapping variable names to their (non-vector) value.
    var_mappings: Mapping of all variables to dimensions of an embedding vector.

  Returns:
    A vector encoding the given variables.
  """
  vector = np.zeros(var_mappings.end_idx)
  for var_name, var_value in variables.items():
    # Initialize variables in `attention_outputs` to all zeroes so that they can
    # be "written" to easily by the first attention operation without being
    # "overridden" by the residual connection.
    if var_name in attention_outputs:
      continue
    var_mapping = var_mappings.var_mappings[var_name]
    if isinstance(var_mapping, dim_utils.NumericalVarDimMapping):
      if not isinstance(var_value, float):
        raise ValueError(
            f"Numerical variable `{var_name}` is not a float: {var_value}"
        )
      vector[var_mapping.idx] = var_value
    elif isinstance(var_mapping, dim_utils.SetVarDimMapping):
      assert isinstance(var_value, frozenset)
      for value in var_value:
        assert isinstance(value, int)
        var_idx = var_mapping.start_idx + value
        if var_idx >= var_mapping.end_idx:
          raise ValueError(
              "Value exceeds variable range.\n%s\n%s\n%s"
              % (var_name, var_mapping, var_value)
          )
        vector[var_idx] = 1.0
    elif isinstance(var_mapping, dim_utils.CategoricalVarDimMapping):
      assert isinstance(var_value, int)
      var_idx = var_mapping.start_idx + var_value
      if var_idx >= var_mapping.end_idx:
        raise ValueError(
            "Value exceeds variable range.\n%s\n%s\n%s"
            % (var_name, var_mapping, var_value)
        )
      vector[var_idx] = 1.0
    else:
      raise ValueError
  return vector


def _get_default_non_position_variables(
    program_spec: program.Program, input_id: int
) -> program.Activations:
  """Returns default value of non-position variables."""
  var_map = {}
  for var_name, var_spec in program_spec.variables.items():
    if var_spec.position_init_fn is not None:
      continue
    elif var_spec.input_init_fn is not None:
      var_map[var_name] = var_spec.input_init_fn(input_id)
    else:
      var_map[var_name] = var_spec.default
  return var_map


def _get_default_position_variables(
    program_spec: program.Program, position: int
) -> program.Activations:
  """Returns default value of position variables."""
  var_map = {}
  for var_name, var_spec in program_spec.variables.items():
    if var_spec.position_init_fn is not None:
      var_map[var_name] = var_spec.position_init_fn(position)
  return var_map


def get_attention_outputs(program_spec: program.Program) -> frozenset[str]:
  """Returns attention outputs for the given program spec."""
  attention_outputs = set()
  for head_spec in program_spec.head_specs:
    attention_outputs.add(head_spec.output)
  return frozenset(attention_outputs)


def get_embedding_parameters(
    program_spec: program.Program, var_mapping: dim_utils.VarDimMappings
) -> parameters.EmbeddingParameters:
  """Returns embedding parameters for the given program spec."""
  attention_outputs = get_attention_outputs(program_spec)

  input_embedding_stack = []
  for input_id in range(program_spec.input_range):
    variables = _get_default_non_position_variables(program_spec, input_id)
    input_embedding_stack.append(
        variables_to_vector(attention_outputs, variables, var_mapping)
    )
  input_embeddings = np.stack(input_embedding_stack, axis=0)

  index_embeddings_stack = []
  index_embeddings = None
  if program_spec.position_range:
    for position in range(program_spec.position_range):
      variables = _get_default_position_variables(program_spec, position)
      index_embeddings_stack.append(
          variables_to_vector(attention_outputs, variables, var_mapping)
      )
    index_embeddings = np.stack(index_embeddings_stack, axis=0)

  return parameters.EmbeddingParameters(
      input_embeddings=input_embeddings, index_embeddings=index_embeddings
  )
