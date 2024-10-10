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

"""Define dataclass for traces."""

import dataclasses
import random
from typing import Any

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from framework import program
from framework.common import io_utils
from framework.common import tf_utils
from framework.compiler import dim_utils
from framework.interpreter import interpreter_utils
from framework.interpreter import logger_utils
from framework.interpreter import program_utils


@dataclasses.dataclass
class FFNTrace:
  """Represents the inputs and outputs of FFN during execution."""

  # Input and output variable maps.
  variables_in: program.Activations
  variables_out: program.Activations
  # Metadata for trace.
  layer_idx: int = 0
  element_idx: int = 0
  # Optional metadata for trace.
  model_input: list[int] | None = None
  # Input and output vectors.
  vector_in: npt.ArrayLike | None = None
  vector_out: npt.ArrayLike | None = None

  def serialize_to_dict(self) -> dict[str, Any]:
    """Return JSON serializable dict."""
    json_dict = {
        "variables_in": self.variables_in,
        "variables_out": self.variables_out,
        "model_input": self.model_input,
        "layer_idx": self.layer_idx,
        "element_idx": self.element_idx,
    }
    if self.vector_in is not None:
      json_dict["vector_in"] = self.vector_in.tolist()  # pytype: disable=attribute-error
    if self.vector_out is not None:
      json_dict["vector_out"] = self.vector_out.tolist()  # pytype: disable=attribute-error
    return json_dict


def trace_from_dict(json_dict: dict[str, Any]) -> FFNTrace:
  """Return Trace object from json dict."""
  kwargs = json_dict.copy()
  if "vector_in" in kwargs:
    kwargs["vector_in"] = np.array(json_dict["vector_in"])
  if "vector_out" in kwargs:
    kwargs["vector_out"] = np.array(json_dict["vector_out"])
  return FFNTrace(**kwargs)


def write_traces_jsonl(output_path: str, traces: list[FFNTrace]):
  rows = [trace.serialize_to_dict() for trace in traces]
  io_utils.write_jsonl(output_path, rows)


def read_traces_jsonl(input_path: str) -> list[FFNTrace]:
  rows = io_utils.read_jsonl(input_path)
  return [trace_from_dict(row) for row in rows]


def _set_random_variable_vector(
    var_mapping: dim_utils.CategoricalVarDimMapping,
    vector: np.ndarray,
) -> None:
  """Randomizes portion of `vector` corresponding to `var_mapping`."""
  start_idx = var_mapping.start_idx
  end_idx = var_mapping.end_idx
  # TODO(jamesfcohan): Make range configurable?
  vector[start_idx:end_idx] = np.random.uniform(
      low=-10, high=10, size=end_idx - start_idx
  )


def variables_to_vector(
    variables: dict[str, Any],
    var_mappings: dim_utils.VarDimMappings,
    randomize_undefined_variables=False,
) -> np.ndarray:
  """Return a vector encoding the given variables.

  Same as `embedding_utils.variables_to_vector` but with a couple of exceptions.

  1. `attention_outputs` are NOT converted to all 0s. The compiler initializes
  attention output variables to all 0s so that after the first layer's attention
  op (before the FFN has ever run) we can add the attention output back to the
  initial embeddings (the residual connection). In trace supervision though,
  there's no residual connection. We just take the input to the MLP in the
  interpreter.

  2. Includes an option to randomize undefined variables. Used for trace
  supervision inputs (not outputs) to ensure that the learned FFN is robust to
  any value for an undefined variable.

  Args:
    variables: Dict mapping variable names to their (non-vector) value.
    var_mappings: Mapping of all variables to dimensions of an embedding vector.
    randomize_undefined_variables: If true, randomizes the value of undefined
      variables. Used for trace supervision inputs (not outputs) to ensure that
      the learned FFN is robust to any value for an undefined variable.

  Returns:
    A vector encoding the given variables.
  """
  vector = np.zeros(var_mappings.end_idx)
  for var_name, var_value in variables.items():
    # None corresponds to zero vector.
    if var_value is None and not randomize_undefined_variables:
      continue
    var_mapping = var_mappings.var_mappings[var_name]
    if isinstance(var_mapping, dim_utils.NumericalVarDimMapping):
      if var_value is None:
        # TODO(jamesfcohan): Make range configurable?
        var_value = random.uniform(-10, 10)
      if not isinstance(var_value, float):
        raise ValueError
      vector[var_mapping.idx] = var_value
    elif isinstance(var_mapping, dim_utils.SetVarDimMapping):
      if var_value is None:
        _set_random_variable_vector(var_mapping, vector)
        continue
      assert isinstance(var_value, frozenset[int])
      for value in var_value:
        var_idx = var_mapping.start_idx + value
        if var_idx >= var_mapping.end_idx:
          raise ValueError(
              "Value exceeds variable range.\n%s\n%s\n%s"
              % (var_name, var_mapping, var_value)
          )
        vector[var_idx] = 1.0
    elif isinstance(var_mapping, dim_utils.CategoricalVarDimMapping):
      if var_value is None:
        _set_random_variable_vector(var_mapping, vector)
        continue
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


def add_vectors(program_spec: program.Program, traces: list[FFNTrace]):
  """Add vector representations of input and output variables to traces."""
  var_mapping = dim_utils.get_var_mapping(program_spec)
  for trace in traces:
    trace.vector_in = variables_to_vector(
        variables=trace.variables_in,
        var_mappings=var_mapping,
        # Randomize undefined variables to ensure that the learned FFN is
        # robust to any value for an undefined variable.
        randomize_undefined_variables=True,
    )
    trace.vector_out = variables_to_vector(
        variables=trace.variables_out,
        var_mappings=var_mapping,
    )


def extract_traces(
    model_inputs: list[list[int]],
    program_spec: program.Program,
    max_layers: int,
) -> list[FFNTrace]:
  """Extract traces by running interpreter for every model input."""
  traces = []
  for input_ids in model_inputs:
    activations_seq = program_utils.initialize_activations(
        program_spec, input_ids
    )
    logger = logger_utils.ActivationsLogger()
    _ = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        logger=logger,
        max_layers=max_layers,
    )
    for layer_idx, (mlp_input, mlp_output) in enumerate(
        logger.get_layer_activations()
    ):
      for element_idx, (variables_in, variables_out) in enumerate(
          zip(mlp_input, mlp_output)
      ):
        trace = FFNTrace(
            layer_idx=layer_idx,
            element_idx=element_idx,
            variables_in=variables_in,
            variables_out=variables_out,
            model_input=input_ids,
        )
        traces.append(trace)
  return traces


def create_example(trace):
  """Creates a tf.Example for a trace."""
  example = tf.train.Example()
  tf_utils.add_float_list_feature(example, "input", trace.vector_in)
  tf_utils.add_float_list_feature(example, "output", trace.vector_out)
  # For debugging.
  tf_utils.add_int_list_feature(example, "model_input", trace.model_input)
  tf_utils.add_int_feature(example, "layer_idx", trace.layer_idx)
  tf_utils.add_int_feature(example, "element_idx", trace.element_idx)
  return example
