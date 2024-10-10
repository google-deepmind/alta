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

"""Writes set of satisfied rules."""

import json

from absl import app
from absl import flags
import apache_beam as beam

from framework import program_registry
from framework.interpreter import interpreter_utils
from framework.interpreter import program_utils
from framework.mlp import mlp_logger
from framework.mlp import rule_io


_INPUT = flags.DEFINE_string(
    "input",
    "",
    "Jsonl file where each line is tuple of input ids.",
)

_PROGRAM = flags.DEFINE_string(
    "program",
    "",
    "Name of program to run.",
)

_OUTPUT = flags.DEFINE_string(
    "output", "", "Where to write output jsonl file of rules."
)

_MAX_LAYERS = flags.DEFINE_integer(
    "max_layers", 512, "Maximum number of layers."
)


class GetRules(beam.DoFn):
  """Determines set of satisfied rules for given model inputs."""

  def setup(self):
    self.logger = mlp_logger.MLPLogger()
    self.program_spec = program_registry.get_program(_PROGRAM.value)

  def process(self, model_inputs_row: str):
    input_ids = json.loads(model_inputs_row)
    beam.metrics.Metrics.counter("counters", "num_inputs").inc()

    activations_seq = program_utils.initialize_activations(
        self.program_spec,
        input_ids,
    )
    _ = interpreter_utils.run_transformer(
        self.program_spec,
        activations_seq,
        max_layers=_MAX_LAYERS.value,
        logger=None,
        logger_mlp=self.logger,
    )

    for rule in self.logger.seen:
      beam.metrics.Metrics.counter("counters", "num_rules").inc()
      yield rule_io.rule_to_json(rule)
    self.logger.reset()


def pipeline(root: beam.Pipeline) -> None:
  """Configure beam pipeline."""

  _ = (
      root
      | "Read" >> beam.io.ReadFromText(_INPUT.value)
      | "Reshuffle1" >> beam.Reshuffle()
      | "Convert" >> beam.ParDo(GetRules())
      | "Deduplicate" >> beam.Distinct()
      | "Reshuffle2" >> beam.Reshuffle()
      | "Write" >> beam.io.WriteToText(_OUTPUT.value)
  )


def main(argv):
  with beam.Pipeline(
      options=beam.options.pipeline_options.PipelineOptions(argv[1:]),
  ) as root:
    pipeline(root)


if __name__ == "__main__":
  app.run(main)
