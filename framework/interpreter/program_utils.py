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

"""Utility functions for configuring and initializing programs."""

from framework import program


def _get_default_var_map(
    var_specs: program.VariablesMap,
    attention_output_variables: frozenset[str],
    position: int,
    input_id: int,
):
  """Get map of variables to their initial values."""
  var_map = {}
  for var_name, var_spec in var_specs.items():
    if var_name in attention_output_variables:
      # Initializes attention outputs to None for consistency with compiler.
      # Compiler initializes attention outputs to zero vector to allow attention
      # output variable to be "written" to easily without being "overridden" by
      # the residual connection.
      var_map[var_name] = None
    elif var_spec.input_init_fn is not None:
      var_map[var_name] = var_spec.input_init_fn(input_id)
    elif var_spec.position_init_fn is not None:
      var_map[var_name] = var_spec.position_init_fn(position)
    else:
      var_map[var_name] = var_spec.default
  return var_map


def initialize_activations(
    program_spec: program.Program,
    input_ids: list[int],
    position_shift: int = 0,
) -> list[program.Activations]:
  """Initializes activations to default values."""
  attention_output_variables = frozenset(
      head_spec.output for head_spec in program_spec.head_specs
  )
  activations_seq = []
  for position, input_id in enumerate(input_ids):
    activations_seq.append(
        _get_default_var_map(
            program_spec.variables,
            attention_output_variables,
            # Optionally shift positional indexes by specified amount.
            position + position_shift,
            input_id,
        )
    )
  return activations_seq
