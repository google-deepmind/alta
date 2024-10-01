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

"""Tests interpreter and compiler for SUBLEQ."""

from absl.testing import absltest

from alta.examples import subleq
from alta.framework.compiler import compiler_config
from alta.framework.compiler import compiler_utils
from alta.framework.interpreter import interpreter_utils
from alta.framework.interpreter import logger_utils
from alta.framework.interpreter import program_utils
from alta.framework.transformer import transformer_utils


class SubleqTest(absltest.TestCase):

  def test_add_interpreter(self):
    """Tests SUBLEQ program for addition.

    From Wikipedia:
      ADD a, b:
        subleq a, z
        subleq z, b
        subleq z, z

      The first instruction subtracts the content at location a
      from the content at location Z (which is 0) and stores the result (which
      is the negative of the content at a) in location Z. The second instruction
      subtracts this result from b, storing in b this difference (which is now
      the sum of the contents originally at a and b); the third instruction
      restores the value 0 to Z.

    In the following program, a == 9, b == 10, and Z == 11. mem[a] is 5 and
    mem[b] is 10. After addition, mem[b] should be 15.
    """
    mem_a = 5
    mem_b = 10
    a = 9
    b = 10
    z = 11
    mem_z = 0
    inputs = [
        a,
        z,
        3,  # Position of next instruction.
        z,
        b,
        6,  # Position of next instruction.
        z,
        z,
        -1,  # Negative position indicates end of program.
        mem_a,
        mem_b,
        mem_z,
    ]
    print("inputs: %s" % inputs)

    program_spec = subleq.build_program_spec()

    activations_seq = program_utils.initialize_activations(
        program_spec, subleq.encode_inputs(inputs)
    )
    logger = logger_utils.ActivationsLogger()
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        logger=logger,
    )
    logger.print_activations_table(elements_to_include=[0])
    logger.print_activations_table(variables_to_include=["mem"])
    outputs = subleq.decode_outputs(outputs)
    print("outputs: ", outputs)
    self.assertEqual(outputs[b], 15)

  def test_add_interpreter_sparse(self):
    """Same as above but using sparse implementaiton."""
    mem_a = 5
    mem_b = 10
    a = 9
    b = 10
    z = 11
    mem_z = 0
    inputs = [
        a,
        z,
        3,  # Position of next instruction.
        z,
        b,
        6,  # Position of next instruction.
        z,
        z,
        -1,  # Negative position indicates end of program.
        mem_a,
        mem_b,
        mem_z,
    ]

    program_spec = subleq.build_program_spec_sparse()
    rules = program_spec.mlp.get_rules()
    print("len(rules): %s" % len(rules))

    activations_seq = program_utils.initialize_activations(
        program_spec, subleq.encode_inputs(inputs)
    )
    logger = logger_utils.ActivationsLogger()
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        logger=logger,
    )
    logger.print_activations_table(elements_to_include=[0])
    logger.print_activations_table(variables_to_include=["mem"])
    outputs = subleq.decode_outputs(outputs)
    self.assertEqual(outputs[b], 15)

  def test_add_compiled(self):
    mem_a = 5
    mem_b = 10
    a = 9
    b = 10
    z = 11
    mem_z = 0
    inputs = [
        a,
        z,
        3,  # Position of next instruction.
        z,
        b,
        6,  # Position of next instruction.
        z,
        z,
        -1,  # Negative position indicates end of program.
        mem_a,
        mem_b,
        mem_z,
    ]
    input_ids = subleq.encode_inputs(inputs)
    program_spec = subleq.build_program_spec_sparse()

    config = compiler_config.Config()
    parameters = compiler_utils.compile_transformer(
        program_spec, config, verbose=True
    )
    outputs = transformer_utils.run_transformer(
        parameters,
        learned_ffn_params=None,
        input_ids=input_ids,
        max_layers=16,
        verbose=False,
    )
    outputs = subleq.decode_outputs(outputs)
    self.assertEqual(outputs[b], 15)


if __name__ == "__main__":
  absltest.main()
