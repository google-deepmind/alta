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

"""Tests interpreter and compiler for addition task."""

from absl.testing import absltest
from examples import addition
from framework.compiler import compiler_config
from framework.compiler import compiler_utils
from framework.interpreter import interpreter_utils
from framework.interpreter import program_utils
from framework.transformer import transformer_utils


class AdditionTest(absltest.TestCase):

  def test_addition(self):
    program_spec = addition.build_program_spec()

    input_a = 789
    input_b = 456
    expected = 1245  # 789 + 456
    input_ids = addition.preprocess_input(input_a, input_b)
    activations_seq = program_utils.initialize_activations(
        program_spec, input_ids
    )
    output_ids = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=None,
    )
    print("output_ids: %s" % output_ids)
    output = addition.process_output(output_ids)

    self.assertEqual(output, expected)

  def test_addition_sparse(self):
    program_spec = addition.build_sparse_program_spec()

    input_a = 789
    input_b = 456
    expected = 1245  # 789 + 456
    input_ids = addition.preprocess_input(input_a, input_b)
    activations_seq = program_utils.initialize_activations(
        program_spec, input_ids
    )
    output_ids = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=None,
    )
    print("output_ids: %s" % output_ids)
    output = addition.process_output(output_ids)

    self.assertEqual(output, expected)

  def test_addition_compiled(self):
    program_spec = addition.build_sparse_program_spec()

    input_a = 789
    input_b = 456
    expected = 1245  # 789 + 456
    input_ids = addition.preprocess_input(input_a, input_b)
    config = compiler_config.Config()
    parameters = compiler_utils.compile_transformer(program_spec, config)
    output_ids = transformer_utils.run_transformer(
        parameters,
        learned_ffn_params=None,
        input_ids=input_ids,
        max_layers=10,
        verbose=False,
    )
    print("output_ids: %s" % output_ids)
    output = addition.process_output(output_ids)

    self.assertEqual(output, expected)


if __name__ == "__main__":
  absltest.main()
