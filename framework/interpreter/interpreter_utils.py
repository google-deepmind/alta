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

"""Defines an interpreter for executing programs."""

import copy
import itertools
from typing import Optional
from framework import program
from framework.interpreter import logger_utils
from framework.mlp import mlp_logger


def _get_activations(
    query: frozenset[int],
    keys: list[int],
    query_idx: int,
    relative_position_mask: frozenset[int],
) -> list[bool]:
  """Returns activations for the given `query`.

  Applies `query` to each `key` to generate list of activations. Uses
  `relative_position_mask` and `query_idx` to filter out activations that
  are masked if `relative_position_mask` is non-empty.

  Args:
    query: Set of values to select.
    keys: List of keys to apply `query` to.
    query_idx: Index of the query.
    relative_position_mask: Set of relative positions that are unmasked. If
      empty, all positions are unmasked.

  Returns:
    List of activations.
  """
  activations = []
  for key_idx, key in enumerate(keys):
    if (
        relative_position_mask
        and key_idx - query_idx not in relative_position_mask
    ):
      activations.append(False)
    else:
      activations.append(key in query)
  return activations


def _get_categorical_attention_output(
    query: frozenset[int],
    keys: list[int],
    values: list[int],
    query_idx: int,
    relative_position_mask: frozenset[int],
    verbose=False,
) -> int | None:
  """Returns the output of attention op or None for undefined output."""
  activations = _get_activations(query, keys, query_idx, relative_position_mask)
  if sum(activations) == 0:
    if verbose:
      print(f"Warning: No selected values for keys: `{keys}`.")
    return None
  elif sum(activations) > 1:
    if verbose:
      print("Warning: more than one selected value for categorical head.")
    return None
  for idx, activation in enumerate(activations):
    if activation:
      return values[idx]
  if verbose:
    print("No selected value for categorical head.")
  return None


def _get_numerical_attention_output(
    query: frozenset[int],
    keys: list[int],
    values: list[float],
    query_idx: int,
    relative_position_mask: frozenset[int],
    verbose=False,
) -> float | None:
  """Returns the output of attention op or None for undefined output."""
  activations = _get_activations(query, keys, query_idx, relative_position_mask)
  if sum(activations) == 0:
    if verbose:
      print(f"Warning: No selected values for keys: `{keys}`.")
    return None
  selected_values = []
  for idx, activation in enumerate(activations):
    if activation:
      selected_values.append(values[idx])
  mean_value = float(sum(selected_values) / len(selected_values))
  return mean_value


def _get_attn_fn(head_spec: program.AttentionHeadSpec):
  if isinstance(head_spec, program.CategoricalAttentionHeadSpec):
    return _get_categorical_attention_output
  elif isinstance(head_spec, program.NumericalAttentionHeadSpec):
    return _get_numerical_attention_output
  else:
    raise ValueError(f"Unsupported head spec: {head_spec}")


def _get_query(
    program_spec: program.Program,
    head_spec: program.AttentionHeadSpec,
    activations: program.Activations,
) -> frozenset[int]:
  """Return set representing attention query."""
  query_var_name = head_spec.query
  query_var_spec = program_spec.variables[query_var_name]
  var_value = activations[query_var_name]
  if isinstance(query_var_spec, program.SetVarSpec):
    # Run query set.
    assert isinstance(var_value, frozenset)
    return var_value
  elif isinstance(query_var_spec, program.CategoricalVarSpec):
    # Treat categorical value as a singleton set.
    if not isinstance(var_value, int):
      raise ValueError(
          f"Query var spec is categorical but value is not an int: {var_value}"
      )
    return frozenset([var_value])
  else:
    raise ValueError(f"Unsupported query var spec: {query_var_spec}")


def run_attention_head(
    program_spec: program.Program,
    head_spec: program.AttentionHeadSpec,
    activations_seq: list[program.Activations],
):
  """Runs an attention head."""
  keys = [activations[head_spec.key] for activations in activations_seq]
  values = [activations[head_spec.value] for activations in activations_seq]
  attn_fn = _get_attn_fn(head_spec)
  for i, activations in enumerate(activations_seq):
    query = _get_query(program_spec, head_spec, activations)
    output = attn_fn(query, keys, values, i, head_spec.relative_position_mask)
    activations[head_spec.output] = output


def run_layer(
    program_spec: program.Program,
    attention_output_variables: set[str],
    activations_seq: list[program.Activations],
    logger: Optional[logger_utils.ActivationsLogger],
    logger_mlp: Optional[mlp_logger.MLPLogger],
):
  """Simulates a single Transformer layer."""
  # Run attention heads.
  for head_spec in program_spec.head_specs:
    try:
      run_attention_head(program_spec, head_spec, activations_seq)
    except ValueError as e:
      raise ValueError(
          "Error executing attention head `%s` with error `%s`."
          % (head_spec, e)
      ) from e
  # Copy activations if needed for logging.
  mlp_input = copy.deepcopy(activations_seq) if logger else None

  # Run element-wise feedforward function.
  for activations in activations_seq:
    program_spec.mlp.run_layer(activations, logger_mlp)
    # Set attention outputs to None after ffn for consistency with compiler.
    for attention_output_var in attention_output_variables:
      activations[attention_output_var] = None
  if logger:
    logger.add_layer_activations(mlp_input, copy.deepcopy(activations_seq))


def _should_break(
    activations_seq: list[program.Activations],
    layer_idx: int,
    max_layers: int | None,
    halt_spec: program.HaltSpec | None,
):
  """Whether to break from the Transformer loop."""
  if not max_layers and not halt_spec:
    raise ValueError("Must specify either `max_layers` or `halt_spec`.")
  if max_layers and layer_idx >= max_layers:
    return True
  if halt_spec:
    return all(
        [z[halt_spec.halt_var] == halt_spec.halt_value for z in activations_seq]
    )
  return False


def run_transformer(
    program_spec: program.Program,
    activations_seq: list[program.Activations],
    logger: Optional[logger_utils.ActivationsLogger] = None,
    logger_mlp: Optional[mlp_logger.MLPLogger] = None,
    max_layers: int | None = 100,
):
  """Simulate a Transformer."""
  attention_output_variables = set(
      head_spec.output for head_spec in program_spec.head_specs
  )

  if logger:
    logger.set_initial_activations(copy.deepcopy(activations_seq))

  for layer_idx in itertools.count():
    run_layer(
        program_spec,
        attention_output_variables,
        activations_seq,
        logger,
        logger_mlp,
    )
    if _should_break(
        activations_seq, layer_idx, max_layers, program_spec.halt_spec
    ):
      break
  # Return outputs.
  return [activations[program_spec.outputs] for activations in activations_seq]
