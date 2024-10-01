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

"""Tests interpreter and compiler for sparse parity programs."""

import enum
import random

from absl.testing import absltest
from absl.testing import parameterized

from alta.examples import parity_sparse
from alta.framework.compiler import compiler_config
from alta.framework.compiler import compiler_utils
from alta.framework.interpreter import interpreter_utils
from alta.framework.interpreter import logger_utils
from alta.framework.interpreter import program_utils
from alta.framework.mlp import mlp_logger
from alta.framework.transformer import transformer_utils


class ParityAlgorithm(enum.Enum):
  SEQUENTIAL_ABSOLUTE = 1
  SEQUENTIAL_RELATIVE = 2
  SUM_MOD_2 = 3


def is_even(ids: list[int]) -> bool:
  """Returns whether `ids` contains an even number of 1's."""
  return ids.count(1) % 2 == 0


def get_program_spec(
    parity_algorithm: ParityAlgorithm,
    max_input_length: int = 10,
):
  """Returns Program for given `parity_algorithm`."""
  if parity_algorithm == ParityAlgorithm.SEQUENTIAL_ABSOLUTE:
    return parity_sparse.build_sequential_program_absolute(
        max_input_length=max_input_length,
    )
  elif parity_algorithm == ParityAlgorithm.SEQUENTIAL_RELATIVE:
    return parity_sparse.build_sequential_program_relative()
  elif parity_algorithm == ParityAlgorithm.SUM_MOD_2:
    return parity_sparse.build_sum_mod_2_program_spec(
        max_input_length=max_input_length,
    )
  else:
    raise ValueError(f"Unsupported parity algorithm: {parity_algorithm}")


def get_inputs(
    parity_algorithm: ParityAlgorithm, input_ids: list[int]
) -> list[int]:
  """Adds special token either to start or end of `input_ids`."""
  if parity_algorithm == ParityAlgorithm.SEQUENTIAL_RELATIVE:
    return [parity_sparse.START] + input_ids
  elif parity_algorithm == ParityAlgorithm.SUM_MOD_2:
    return input_ids + [parity_sparse.EOS_VALUE]
  else:
    return input_ids


class ParityTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Set random seed for determinism.
    random.seed(0)

  @parameterized.named_parameters(
      ("sequential_absolute", ParityAlgorithm.SEQUENTIAL_ABSOLUTE),
      ("sequential_relative", ParityAlgorithm.SEQUENTIAL_RELATIVE),
      ("sum_mod_2", ParityAlgorithm.SUM_MOD_2),
  )
  def test_parity(self, parity_algorithm: ParityAlgorithm):
    """Tests on random strings of 0's and 1's."""
    num_tests = 10
    program_spec = get_program_spec(parity_algorithm)
    for _ in range(num_tests):
      input_length = random.randint(1, 10)
      # L must equal at least n + 2. MLP 1 sets `bos`. MLP 2 sets `curr_idx`.
      # MLP 3 is the first to adjust parity.
      max_layers = random.randint(input_length + 2, 20)
      input_ids = [random.choice([0, 1]) for _ in range(input_length)]
      activations_seq = program_utils.initialize_activations(
          program_spec, get_inputs(parity_algorithm, input_ids)
      )
      outputs = interpreter_utils.run_transformer(
          program_spec,
          activations_seq,
          max_layers=max_layers,
      )

      error_msg = (
          f"input_ids: {input_ids}, outputs: {outputs}, max_layers:"
          f" {max_layers}"
      )
      expected_parity = 0 if is_even(input_ids) else 1
      self.assertEqual(outputs[-1], expected_parity, msg=error_msg)

  @parameterized.named_parameters(
      ("sequential_absolute", ParityAlgorithm.SEQUENTIAL_ABSOLUTE),
      ("sequential_relative", ParityAlgorithm.SEQUENTIAL_RELATIVE),
      ("sum_mod_2", ParityAlgorithm.SUM_MOD_2),
  )
  def test_ends_in_one(self, parity_algorithm: ParityAlgorithm):
    """Test that the final "1" is accounted for."""
    input_ids = [1, 0, 1]
    program_spec = get_program_spec(parity_algorithm)
    activations_seq = program_utils.initialize_activations(
        program_spec, get_inputs(parity_algorithm, input_ids)
    )
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=10,
    )
    self.assertEqual(outputs[-1], 0)

  @parameterized.named_parameters(
      ("sequential_absolute", ParityAlgorithm.SEQUENTIAL_ABSOLUTE),
      ("sequential_relative", ParityAlgorithm.SEQUENTIAL_RELATIVE),
      ("sum_mod_2", ParityAlgorithm.SUM_MOD_2),
  )
  def test_empty_sequence(self, parity_algorithm: ParityAlgorithm):
    """Tests single element input."""
    input_ids = []
    program_spec = get_program_spec(parity_algorithm)
    activations_seq = program_utils.initialize_activations(
        program_spec, get_inputs(parity_algorithm, input_ids)
    )
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=10,
    )
    # The output will be empty for absolute sequential program which does not
    # include a start or eos token.
    if outputs:
      self.assertEqual(outputs[-1], 0)

  @parameterized.named_parameters(
      ("sequential_absolute", ParityAlgorithm.SEQUENTIAL_ABSOLUTE),
      ("sequential_relative", ParityAlgorithm.SEQUENTIAL_RELATIVE),
      ("sum_mod_2", ParityAlgorithm.SUM_MOD_2),
  )
  def test_single_element_sequence(self, parity_algorithm: ParityAlgorithm):
    """Tests single element input."""
    input_ids = [1]
    program_spec = get_program_spec(parity_algorithm)
    activations_seq = program_utils.initialize_activations(
        program_spec, get_inputs(parity_algorithm, input_ids)
    )
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=10,
    )
    self.assertEqual(outputs[-1], 1)

  @parameterized.named_parameters(
      ("sequential_absolute", ParityAlgorithm.SEQUENTIAL_ABSOLUTE),
      ("sequential_relative", ParityAlgorithm.SEQUENTIAL_RELATIVE),
      ("sum_mod_2", ParityAlgorithm.SUM_MOD_2),
  )
  def test_interpreter_even(self, parity_algorithm: ParityAlgorithm):
    """Tests interpreted program can count even number of 1's."""
    input_ids = [1, 0, 1]
    program_spec = get_program_spec(parity_algorithm)
    activations_seq = program_utils.initialize_activations(
        program_spec, get_inputs(parity_algorithm, input_ids)
    )
    logger = logger_utils.ActivationsLogger()
    outputs = interpreter_utils.run_transformer(
        program_spec, activations_seq, max_layers=10, logger=logger
    )
    # For debugging:
    # logger.print_activations_table()
    self.assertEqual(outputs[-1], 0)

  @parameterized.named_parameters(
      ("sequential_absolute", ParityAlgorithm.SEQUENTIAL_ABSOLUTE),
      ("sequential_relative", ParityAlgorithm.SEQUENTIAL_RELATIVE),
      ("sum_mod_2", ParityAlgorithm.SUM_MOD_2),
  )
  def test_interpreter_odd(self, parity_algorithm: ParityAlgorithm):
    """Tests interpreted program can count odd number of 1's."""
    input_ids = [1, 0, 1, 1]
    program_spec = get_program_spec(parity_algorithm)
    activations_seq = program_utils.initialize_activations(
        program_spec, get_inputs(parity_algorithm, input_ids)
    )
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=10,
    )
    self.assertEqual(outputs[-1], 1)

  @parameterized.named_parameters(
      ("sequential_absolute", ParityAlgorithm.SEQUENTIAL_ABSOLUTE),
      ("sequential_relative", ParityAlgorithm.SEQUENTIAL_RELATIVE),
      ("sum_mod_2", ParityAlgorithm.SUM_MOD_2),
  )
  def test_compiler_even(self, parity_algorithm: ParityAlgorithm):
    """Tests compiled program can count even number of 1's."""
    program_spec = get_program_spec(parity_algorithm=parity_algorithm)
    config = compiler_config.Config(expansion_scalar_1=1000)
    parameters = compiler_utils.compile_transformer(program_spec, config)
    input_ids = [1, 0, 1]
    outputs = transformer_utils.run_transformer(
        parameters,
        learned_ffn_params=None,
        input_ids=get_inputs(parity_algorithm, input_ids),
        max_layers=4,
        verbose=False,
    )
    self.assertEqual(outputs[-1], 0)

  @parameterized.named_parameters(
      ("sequential_absolute", ParityAlgorithm.SEQUENTIAL_ABSOLUTE),
      ("sequential_relative", ParityAlgorithm.SEQUENTIAL_RELATIVE),
      ("sum_mod_2", ParityAlgorithm.SUM_MOD_2),
  )
  def test_compiler_odd(self, parity_algorithm: ParityAlgorithm):
    """Tests compiled program can count odd number of 1's."""
    program_spec = get_program_spec(parity_algorithm=parity_algorithm)
    config = compiler_config.Config(expansion_scalar_1=1000)
    parameters = compiler_utils.compile_transformer(program_spec, config)
    input_ids = [1, 0, 1, 1]
    outputs = transformer_utils.run_transformer(
        parameters,
        learned_ffn_params=None,
        input_ids=get_inputs(parity_algorithm, input_ids),
        max_layers=5,
        verbose=False,
    )
    self.assertEqual(outputs[-1], 1)

  @parameterized.named_parameters(
      ("sequential_absolute", ParityAlgorithm.SEQUENTIAL_ABSOLUTE),
      ("sequential_relative", ParityAlgorithm.SEQUENTIAL_RELATIVE),
      ("sum_mod_2", ParityAlgorithm.SUM_MOD_2),
  )
  def test_compiler_even_exta_layers(self, parity_algorithm: ParityAlgorithm):
    """Tests compiled program can count even number of 1's with extra layers."""
    program_spec = get_program_spec(parity_algorithm=parity_algorithm)
    config = compiler_config.Config(expansion_scalar_1=1000)
    parameters = compiler_utils.compile_transformer(program_spec, config)
    input_ids = [1, 0, 1]
    outputs = transformer_utils.run_transformer(
        parameters,
        learned_ffn_params=None,
        input_ids=get_inputs(parity_algorithm, input_ids),
        max_layers=10,
        verbose=False,
    )
    self.assertEqual(outputs[-1], 0)

  @parameterized.named_parameters(
      ("sequential_absolute", ParityAlgorithm.SEQUENTIAL_ABSOLUTE),
      ("sequential_relative", ParityAlgorithm.SEQUENTIAL_RELATIVE),
      ("sum_mod_2", ParityAlgorithm.SUM_MOD_2),
  )
  def test_compiler_odd_extra_layers(self, parity_algorithm: ParityAlgorithm):
    """Tests compiled program can count odd number of 1's with extra layers."""
    program_spec = get_program_spec(parity_algorithm=parity_algorithm)
    config = compiler_config.Config(expansion_scalar_1=1000)
    parameters = compiler_utils.compile_transformer(program_spec, config)
    input_ids = [1, 0, 1, 1]
    outputs = transformer_utils.run_transformer(
        parameters,
        learned_ffn_params=None,
        input_ids=get_inputs(parity_algorithm, input_ids),
        max_layers=10,
        verbose=False,
    )
    self.assertEqual(outputs[-1], 1)

  def test_dynamic_halting_relative(self):
    """Tests relative program with dynamic halting."""
    input_ids = [2, 1, 0, 1, 0, 1]
    program_spec = parity_sparse.build_sequential_program_relative(
        dynamic_halting=True
    )
    activations_seq = program_utils.initialize_activations(
        program_spec, input_ids
    )
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=None,
    )
    self.assertEqual(outputs, [0, 1, 1, 0, 0, 1])

  def test_dynamic_halting_absolute(self):
    """Tests absolute program with dynamic halting."""
    input_ids = [1, 0, 1, 0, 1]
    program_spec = parity_sparse.build_sequential_program_absolute(
        dynamic_halting=True
    )
    activations_seq = program_utils.initialize_activations(
        program_spec, input_ids
    )
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=None,
    )
    self.assertEqual(outputs, [1, 1, 0, 0, 1])

  def test_sum_mod_2(self):
    """Tests sum mod 2 for previously problematic input."""
    input_ids = [0, 0, 1, 1, 0, 1, 1, 0, 1]
    program_spec = parity_sparse.build_sum_mod_2_program_spec()
    rules = program_spec.mlp.get_rules()
    print("len(rules): %s" % len(rules))
    for rule in rules:
      print(rule)
    activations_seq = program_utils.initialize_activations(
        program_spec, get_inputs(ParityAlgorithm.SUM_MOD_2, input_ids)
    )
    logger_mlp = mlp_logger.MLPLogger()
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        max_layers=3,
        logger_mlp=logger_mlp,
    )
    self.assertEqual(outputs[-1], 1)

  def test_sum_mod_2_different_num_ones(self):
    """Tests sum mod 2 for different numbers of ones."""
    program_spec = parity_sparse.build_sum_mod_2_program_spec()
    for num_ones in range(41):
      input_ids = [1] * num_ones
      activations_seq = program_utils.initialize_activations(
          program_spec, get_inputs(ParityAlgorithm.SUM_MOD_2, input_ids)
      )
      outputs = interpreter_utils.run_transformer(
          program_spec,
          activations_seq,
          max_layers=3,
      )
      expected_parity = int(num_ones % 2 != 0)
      self.assertEqual(outputs[-1], expected_parity)


if __name__ == "__main__":
  absltest.main()
