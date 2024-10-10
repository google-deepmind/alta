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

"""Runs inference on TF examples of traces."""

import collections

from absl import app
from absl import flags
from absl import logging
import numpy as np

from framework import program_registry
from framework.compiler import debug_utils
from framework.compiler import dim_utils
from framework.traces.ffn import activation
from framework.traces.ffn import data
from framework.traces.ffn import inference
from framework.traces.ffn import metrics
from framework.traces.ffn import serialize


_EXAMPLES_PATH = flags.DEFINE_string(
    "examples_path", None, "Path to traces as TF examples.", required=True
)

_PARAMS_PATH = flags.DEFINE_string("params_path", None, "", required=True)

_VECTOR_LENGTH = flags.DEFINE_integer(
    "vector_length", None, "Length of vector inputs and outputs.", required=True
)

_PROGRAM_NAME = flags.DEFINE_string(
    "program_name",
    None,
    "Name of program to use for debugging.",
)

_NUM_EXAMPLES = flags.DEFINE_integer(
    "num_examples", 2, "Number of examples to run."
)

_ACTIVATION_FN = flags.DEFINE_enum(
    "activation_fn", "relu", ["sigmoid", "relu"], "Activation function."
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  np.set_printoptions(precision=2, threshold=10_000, suppress=True)

  trace_inputs, trace_outputs = data.get_all_data(
      _EXAMPLES_PATH.value,
      _VECTOR_LENGTH.value,
      sample_size=_NUM_EXAMPLES.value,
  )

  program = program_registry.get_program(_PROGRAM_NAME.value)
  var_mappings = dim_utils.get_var_mapping(program)

  params = serialize.load_params(_PARAMS_PATH.value)

  activation_fn = activation.get_activation_fn(_ACTIVATION_FN.value)
  predictions = inference.batched_predict(params, trace_inputs, activation_fn)
  vector_accuracy = metrics.get_vector_accuracy(
      predictions, trace_outputs
  ).item()
  vector_element_accuracy = metrics.get_vector_element_accuracy(
      predictions, trace_outputs
  ).item()

  logging.info("Vector accuracy %s", vector_accuracy)
  logging.info("Vector element accuracy %s", vector_element_accuracy)

  error_counts = collections.defaultdict(int)
  for x, y, y_pred in zip(trace_inputs, trace_outputs, predictions):
    x_vars = debug_utils.vector_to_variables(x, var_mappings)
    y_vars = debug_utils.vector_to_variables(y, var_mappings)
    y_pred_vars = debug_utils.vector_to_variables(y_pred, var_mappings)
    if y_vars != y_pred_vars:
      logging.info("x_vars: %s", x_vars)
      logging.info("y_vars: %s", y_vars)

      for idx, (e, e_pred) in enumerate(zip(y.tolist(), y_pred.tolist())):
        if abs(e - e_pred) > 0.1:
          logging.info("diff at %d: %.2f vs. %.2f (pred)", idx, e, e_pred)

      for key, value in y_pred_vars.items():
        y_ref = y_vars[key]
        if y_ref != value:
          logging.info(
              "diff for `%s`: %s vs. %s (pred) - %s",
              key,
              y_ref,
              value,
              var_mappings.var_mappings[key],
          )
          error_counts[(key, value, y_ref)] += 1

  for (variable, value, ref), count in error_counts.items():
    logging.info("%s, %s, %s, %d", variable, value, ref, count)


if __name__ == "__main__":
  app.run(main)
