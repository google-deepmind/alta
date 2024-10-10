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

"""Generates traces using Beam.

Expects model inputs to be in jsonl format with a list of integers per line.
"""

import json

from absl import app
from absl import flags
import apache_beam as beam
import tensorflow as tf

from framework import program_registry
from framework.traces import trace_utils


_INPUT = flags.DEFINE_string("input", "", "Model inputs jsonl file.")

_OUTPUT = flags.DEFINE_string("output", "", "Path to write traces jsonl file.")

_PROGRAM = flags.DEFINE_string("program", "", "Program name from registry.")

_MAX_LAYERS = flags.DEFINE_integer(
    "max_layers", 128, "Maximum number of layers."
)


class ConvertToTraces(beam.DoFn):
  """Converts model inputs to traces."""

  def setup(self):
    self.program_spec = program_registry.get_program(_PROGRAM.value)

  def process(self, model_inputs_row: str):
    model_inputs = [json.loads(model_inputs_row)]
    beam.metrics.Metrics.counter("counters", "num_inputs").inc()
    traces = trace_utils.extract_traces(
        model_inputs, self.program_spec, max_layers=_MAX_LAYERS.value
    )
    trace_utils.add_vectors(self.program_spec, traces)
    for trace in traces:
      beam.metrics.Metrics.counter("counters", "num_traces").inc()
      yield trace_utils.create_example(trace)


def pipeline(root: beam.Pipeline) -> None:
  """Configure beam pipeline."""

  _ = (
      root
      | "Read" >> beam.io.ReadFromText(_INPUT.value)
      | "Reshuffle1" >> beam.Reshuffle()
      | "Convert" >> beam.ParDo(ConvertToTraces())
      | "Reshuffle2" >> beam.Reshuffle()
      | "Write"
      >> beam.io.WriteToTFRecord(
          _OUTPUT.value,
          coder=beam.coders.ProtoCoder(tf.train.Example),
      )
  )


def main(argv):
  with beam.Pipeline(
      options=beam.options.pipeline_options.PipelineOptions(argv[1:]),
  ) as root:
    pipeline(root)


if __name__ == "__main__":
  app.run(main)
