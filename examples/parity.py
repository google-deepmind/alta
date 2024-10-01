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

"""Implements parity task.

This task was discussed in What Algorithms can Transformers Learn?
A Study in Length Generalization.

https://arxiv.org/pdf/2310.16028.pdf

The algorithm used here is to take n steps for an input of length n and keep
track of the running parity. Outputs a sequence of all 0's if there are an even
number of 1's and a sequence of all 1's if there are an odd number.
"""

from framework import program_builder as pb

# Some programs use EOS or START token.
EOS_VALUE = 2
START = 2


def build_sequential_program_absolute(
    dynamic_halting: bool = False,
    max_input_length: int = 50,
    generate_rules: bool = True,
):
  """Sequential algorithm using absolute positions."""
  done_value = 1
  variables = {
      "parity": pb.input_var(2),
      "done": pb.position_var(2, init_fn=lambda x: x == 0),
      "idx": pb.position_var(max_input_length),
      "idx_left": pb.position_var(
          max_input_length, init_fn=lambda x: max(0, x - 1)
      ),
  }
  attention_heads = {
      "parity_left": pb.qkv("idx_left", "idx", "parity"),
      "done_left": pb.qkv("idx_left", "idx", "done"),
  }

  def ffn_fn(z):
    """MLP for computing parity."""
    if z["done"] != done_value and z["done_left"] == done_value:
      z["parity"] = z["parity_left"] ^ z["parity"]
      z["done"] = done_value

  if dynamic_halting:
    halt_spec = pb.halt_spec("done", halt_value=done_value)
  else:
    halt_spec = None

  return pb.program_spec(
      variables=variables,
      heads=attention_heads,
      ffn_fn=ffn_fn,
      output_name="parity",
      input_range=2,
      position_range=max_input_length,
      halt=halt_spec,
      generate_rules=generate_rules,
  )


def build_sequential_program_relative(
    dynamic_halting: bool = False, generate_rules: bool = True
):
  """Sequential algorithm using relative positions."""
  done_value = 1
  variables = {
      "parity": pb.input_var(2, init_fn=lambda x: 0 if x == START else x),
      "done": pb.input_var(2, init_fn=lambda x: x == START),
  }
  attention_heads = {
      "parity_left": pb.v_relative("parity", -1),
      "done_left": pb.v_relative("done", -1),
  }

  def ffn_fn(z):
    """MLP for computing parity."""
    if z["done"] != done_value and z["done_left"] == done_value:
      z["parity"] = z["parity_left"] ^ z["parity"]
      z["done"] = done_value

  if dynamic_halting:
    halt_spec = pb.halt_spec("done", halt_value=done_value)
  else:
    halt_spec = None

  return pb.program_spec(
      variables=variables,
      heads=attention_heads,
      ffn_fn=ffn_fn,
      output_name="parity",
      input_range=3,
      position_range=None,
      halt=halt_spec,
      generate_rules=generate_rules,
  )


def build_sum_mod_2_program_spec(
    max_input_length: int = 50, generate_rules: bool = True
):
  """Returns a program spec for parity task using sum mod 2 algorithm."""
  input_range = 3

  variables = {
      "parity": pb.var(2),
      "eos": pb.numeric_var(
          input_init_fn=lambda x: float(x == EOS_VALUE), values=(0, 1)
      ),
      "eos_or_one": pb.input_var(2, init_fn=lambda x: x in {1, EOS_VALUE}),
      "query": pb.var(2, default=1),
  }

  buckets = tuple(sorted([1 / x for x in range(1, max_input_length + 1)]))
  output_spec = pb.numeric_var(values=buckets)
  attention_heads = {
      "x": pb.qkv("query", "eos_or_one", "eos", output_spec=output_spec),
  }

  def ffn_fn(activations):
    """MLP for computing parity."""
    # Uses `selector_width` algorithm to calculate number of 1's in input.
    num_ones = round(1 / activations["x"]) - 1
    activations["parity"] = int(num_ones % 2 != 0)

  return pb.program_spec(
      variables=variables,
      heads=attention_heads,
      ffn_fn=ffn_fn,
      output_name="parity",
      input_range=input_range,
      position_range=None,
      generate_rules=generate_rules,
  )


def build_intermediate_variable_sum_mod_2_program_spec(
    max_input_length: int = 50,
    generate_rules: bool = False,
):
  """Uses sum mod 2 algorithm but stores `num_ones` as an intermediate variable.

  Modified version of build_sum_mod_2_program_spec used for intermediate
  supervision. It makes two changes:
  1. Stores `num_ones` as an intermediate, categorical variable. In practice
     it is easier for the FFN to learn to map a categorical `num_ones` variable
     to `parity` than a numerical `x` variable.
  2. Scales `x` up to allow for adding noise during training.

  Args:
    max_input_length: Maximum input length.
    generate_rules: It is slow to run this program with `generate_rules=True`
      due to the number of possible combinations, so it is disabled by default.

  Returns:
    Program spec for sum mod 2 parity algorithm.
  """
  input_range = 3
  position_range = max_input_length + 1
  # Scale `eos` so that `x` values have a larger range. Makes it possible to
  # add noise when training with intermediate supervision.
  scalar = 1000
  variables = {
      "state": pb.var(3),
      "parity": pb.var(2),
      "eos": pb.numeric_var(
          values=(0, scalar),
          input_init_fn=lambda x: float(x == EOS_VALUE) * scalar,
      ),
      "num_ones": pb.var(position_range),
      "eos_or_one": pb.input_var(2, init_fn=lambda x: x in {1, EOS_VALUE}),
      "query": pb.var(2, default=1),
  }

  buckets = tuple(sorted([scalar / x for x in range(1, max_input_length + 1)]))
  output_spec = pb.numeric_var(values=buckets)
  attention_heads = {
      "x": pb.qkv(
          "query",
          "eos_or_one",
          "eos",
          output_spec=output_spec,
      ),
  }

  def ffn_fn(activations):
    """MLP for computing parity."""
    # Uses `selector_width` algorithm to calculate number of 1's in input.
    if activations["state"] == 0:
      activations["num_ones"] = round(scalar / activations["x"]) - 1
      activations["state"] = 1
    else:
      activations["parity"] = int(activations["num_ones"] % 2 != 0)

  return pb.program_spec(
      variables=variables,
      heads=attention_heads,
      ffn_fn=ffn_fn,
      output_name="parity",
      input_range=input_range,
      position_range=None,
      generate_rules=generate_rules,
  )
