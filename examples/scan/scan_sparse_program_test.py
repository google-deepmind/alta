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

"""Tests interpreter for SCAN program with sparse MLP."""

import unittest

from absl.testing import absltest

from alta.examples.scan import scan_sparse_program
from alta.examples.scan import scan_utils
from alta.framework.compiler import compiler_config
from alta.framework.compiler import compiler_utils
from alta.framework.interpreter import interpreter_utils
from alta.framework.interpreter import logger_utils
from alta.framework.interpreter import program_utils
from alta.framework.transformer import transformer_utils


class ScanGraphTest(absltest.TestCase):

  def test_scan_1(self):
    program_spec = scan_sparse_program.build_program_spec(max_num_padding=8)

    input_ids = scan_utils.input_string_to_input_ids(
        "jump twice after walk", padding=2
    )
    activations_seq = program_utils.initialize_activations(
        program_spec, input_ids
    )
    logger = logger_utils.ActivationsLogger()
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=128,
        logger=logger,
    )
    # To add more debug logging, e.g.:
    # logger.print_activations_table(elements_to_include=[0])
    # logger.print_activations_table(variables_to_include=["symbol_id"])
    output_tokens = scan_utils.decode_output(outputs)
    self.assertEqual(output_tokens, ["WALK", "JUMP", "JUMP"])

  def test_scan_2(self):
    program_spec = scan_sparse_program.build_program_spec()

    input_ids = scan_utils.input_string_to_input_ids(
        "look right and turn opposite right twice", padding=0
    )
    activations_seq = program_utils.initialize_activations(
        program_spec, input_ids
    )
    logger = logger_utils.ActivationsLogger()
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=128,
        logger=logger,
    )
    # To add more debug logging, e.g.:
    # logger.print_activations_table(elements_to_include=[0])
    # logger.print_activations_table(variables_to_include=["symbol_id"])
    output_tokens = scan_utils.decode_output(outputs)
    self.assertEqual(
        output_tokens, ["RTURN", "LOOK", "RTURN", "RTURN", "RTURN", "RTURN"]
    )

  @unittest.skip("This test runs slowly.")
  def test_scan_3(self):
    program_spec = scan_sparse_program.build_program_spec()

    input_ids = scan_utils.input_string_to_input_ids(
        "run around right thrice after jump around left thrice", padding=0
    )
    activations_seq = program_utils.initialize_activations(
        program_spec, input_ids
    )
    logger = logger_utils.ActivationsLogger()
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=512,
        logger=logger,
    )
    output_tokens = scan_utils.decode_output(outputs)
    print("output_tokens: %s" % output_tokens)
    self.assertEqual(
        output_tokens,
        [
            "LTURN",
            "JUMP",
            "LTURN",
            "JUMP",
            "LTURN",
            "JUMP",
            "LTURN",
            "JUMP",
            "LTURN",
            "JUMP",
            "LTURN",
            "JUMP",
            "LTURN",
            "JUMP",
            "LTURN",
            "JUMP",
            "LTURN",
            "JUMP",
            "LTURN",
            "JUMP",
            "LTURN",
            "JUMP",
            "LTURN",
            "JUMP",
            "RTURN",
            "RUN",
            "RTURN",
            "RUN",
            "RTURN",
            "RUN",
            "RTURN",
            "RUN",
            "RTURN",
            "RUN",
            "RTURN",
            "RUN",
            "RTURN",
            "RUN",
            "RTURN",
            "RUN",
            "RTURN",
            "RUN",
            "RTURN",
            "RUN",
            "RTURN",
            "RUN",
            "RTURN",
            "RUN",
        ],
    )

  def test_scan_4_interpreter(self):
    program_spec = scan_sparse_program.build_program_spec(max_num_padding=0)

    input_ids = scan_utils.input_string_to_input_ids(
        "jump twice", padding=0
    )
    activations_seq = program_utils.initialize_activations(
        program_spec, input_ids
    )
    logger = logger_utils.ActivationsLogger()
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=64,
        logger=logger,
    )
    # To add more debug logging, e.g.:
    # logger.print_activations_table(elements_to_include=[0])
    # logger.print_activations_table(variables_to_include=["symbol_id"])
    output_tokens = scan_utils.decode_output(outputs)
    self.assertEqual(output_tokens, ["JUMP", "JUMP"])

  def test_scan_4_compiler(self):
    program_spec = scan_sparse_program.build_program_spec(max_num_padding=0)

    input_ids = scan_utils.input_string_to_input_ids(
        "jump twice", padding=0
    )
    config = compiler_config.Config()
    parameters = compiler_utils.compile_transformer(program_spec, config)
    outputs = transformer_utils.run_transformer(
        parameters,
        learned_ffn_params=None,
        input_ids=input_ids,
        max_layers=64,
        verbose=False,
    )

    output_tokens = scan_utils.decode_output(outputs)
    self.assertEqual(output_tokens, ["JUMP", "JUMP"])


if __name__ == "__main__":
  absltest.main()
