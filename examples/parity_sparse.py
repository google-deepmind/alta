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

"""Implements sparse version of programs for parity task."""

from alta.framework import program_builder as pb

# Some programs use EOS or START token.
EOS_VALUE = 2
START = 2


def build_sequential_program_absolute(
    dynamic_halting: bool = False,
    max_input_length: int = 50,
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

  if dynamic_halting:
    halt_spec = pb.halt_spec("done", halt_value=done_value)
  else:
    halt_spec = None

  x = pb.MLPBuilder(variables, attention_heads)
  for done in x.get("done"):
    if done != done_value:
      for done_left in x.get("done_left"):
        if done_left == done_value:
          x.set("done", done_value)
          for parity_left, parity in x.get("parity_left", "parity"):
            x.set("parity", parity_left ^ parity)

  return pb.program_spec_from_rules(
      variables=variables,
      heads=attention_heads,
      rules=x.rules,
      output_name="parity",
      input_range=2,
      position_range=max_input_length,
      halt=halt_spec,
  )


def build_sequential_program_relative(
    dynamic_halting: bool = False,
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

  if dynamic_halting:
    halt_spec = pb.halt_spec("done", halt_value=done_value)
  else:
    halt_spec = None

  x = pb.MLPBuilder(variables, attention_heads)
  for done in x.get("done"):
    if done != done_value:
      for done_left in x.get("done_left"):
        if done_left == done_value:
          x.set("done", done_value)
          for parity_left, parity in x.get("parity_left", "parity"):
            x.set("parity", parity_left ^ parity)

  return pb.program_spec_from_rules(
      variables=variables,
      heads=attention_heads,
      rules=x.rules,
      output_name="parity",
      input_range=3,
      position_range=None,
      halt=halt_spec,
  )


def build_sum_mod_2_program_spec(max_input_length: int = 100):
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
      "inv_num_ones": pb.qkv(
          "query", "eos_or_one", "eos", output_spec=output_spec
      ),
  }

  x = pb.MLPBuilder(variables, attention_heads)
  for inv_num_ones in x.get("inv_num_ones"):
    num_ones = round(1 / inv_num_ones) - 1
    parity = int(num_ones % 2 != 0)
    x.set("parity", parity)

  return pb.program_spec_from_rules(
      variables=variables,
      heads=attention_heads,
      rules=x.rules,
      output_name="parity",
      input_range=input_range,
      position_range=None,
  )
