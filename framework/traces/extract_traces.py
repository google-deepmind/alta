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

"""Extract traces from model."""

from absl import app
from absl import flags

from alta.framework import program_registry
from alta.framework.common import io_utils
from alta.framework.traces import trace_utils


_INPUT = flags.DEFINE_string("input", "", "Jsonl file with model inputs.")

_OUTPUT = flags.DEFINE_string("output", "", "Path to write serialized traces.")

_OUTPUT_FORMAT = flags.DEFINE_enum(
    "output_format",
    "json",
    ["json", "tfrecord"],
    "File format to use for output.",
)

_PROGRAM = flags.DEFINE_string("program", "", "Program name.")

_MAX_LAYERS = flags.DEFINE_integer("max_layers", 8, "Number of layers to run.")

_SAMPLE = flags.DEFINE_integer(
    "sample",
    0,
    "Sample only this many examples if > 0.",
)


def create_examples(traces):
  """Creates a tf.Example for each trace."""
  return [trace_utils.create_example(trace) for trace in traces]


def main(unused_argv):
  model_inputs = io_utils.read_jsonl(_INPUT.value)
  if _SAMPLE.value > 0:
    model_inputs = model_inputs[: _SAMPLE.value]

  program_spec = program_registry.get_program(_PROGRAM.value)
  traces = trace_utils.extract_traces(
      model_inputs, program_spec, max_layers=_MAX_LAYERS.value
  )
  trace_utils.add_vectors(program_spec, traces)

  if _OUTPUT_FORMAT.value == "json":
    trace_utils.write_traces_jsonl(_OUTPUT.value, traces)
  elif _OUTPUT_FORMAT.value == "tfrecord":
    examples = create_examples(traces)
    io_utils.write_tfrecords(examples, _OUTPUT.value)


if __name__ == "__main__":
  app.run(main)
