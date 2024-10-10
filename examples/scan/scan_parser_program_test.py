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

"""Tests interpreter for SCAN parser program."""

from absl.testing import absltest

from examples.scan import grammar_utils
from examples.scan import scan_parser_program
from examples.scan import scan_utils
from framework.interpreter import interpreter_utils
from framework.interpreter import logger_utils
from framework.interpreter import program_utils


class ScanParserProgramTest(absltest.TestCase):

  def test_scan(self):
    program_spec = scan_parser_program.build_program_spec()

    input_tokens = ["jump", "twice", "after", "walk", "eos"]
    input_ids = [scan_utils.get_input_id(token) for token in input_tokens]
    activations_seq = program_utils.initialize_activations(
        program_spec, input_ids
    )
    logger = logger_utils.ActivationsLogger()
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=32,
        logger=logger,
    )
    # To add more debug logging, e.g.:
    # logger.print_activations_table()
    rule_ids = [x - 1 for x in outputs if x > 0]
    rule_sources = [" ".join(grammar_utils.RULES[x].source) for x in rule_ids]
    self.assertEqual(rule_sources, ["jump", "S twice", "walk", "S after S"])


if __name__ == "__main__":
  absltest.main()
