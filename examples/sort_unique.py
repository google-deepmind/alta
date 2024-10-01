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

"""Implements sorting a unique set of numbers.

This is a toy task from Tracr paper that sorts a unique set of numbers.

The RASP program for this is:
  smaller = select (tokens, tokens, <=)
  target_pos = selector_width (smaller)
  sel_sort = select (target_pos, indices, ==)
  sort = aggregate (sel_sort, tokens)

This implementation relies on the presence of a special BOS token.
"""

from alta.framework import program_builder as pb

BOS_VALUE = 0
INPUT_RANGE = 16
POSITION_RANGE = 16


def _get_target_pos_query(input_id):
  if input_id == BOS_VALUE:
    return frozenset([BOS_VALUE])
  else:
    return frozenset(range(input_id + 1))


def _possible_target_pos_query_values():
  possible_values = set()
  for input_id in range(INPUT_RANGE):
    possible_values.add(_get_target_pos_query(input_id))
  return frozenset(possible_values)


def build_program_spec():
  """Returns a program spec for sort unique task."""

  variables = {
      "inputs": pb.input_var(INPUT_RANGE),
      "indices": pb.position_var(INPUT_RANGE),
      "state": pb.var(2),
      "target_pos_query": pb.set_var(
          range=INPUT_RANGE,
          values=_possible_target_pos_query_values(),
          input_init_fn=_get_target_pos_query,
      ),
      # Values should be of the form [1, 0, 0, ..., 0]. That way be we can
      # calculate the inverse of the number of Trues in a select matrix row (aka
      # the selector_width) by selecting the 1 from index zero and 0's
      # elsewhere.
      "bos": pb.numeric_var(position_init_fn=lambda x: float(x == 0)),
      "target_pos": pb.var(POSITION_RANGE),
      "program_ouput": pb.var(INPUT_RANGE),
  }
  attenion_heads = {
      "x": pb.qkv("target_pos_query", "inputs", "bos"),
      "output": pb.qkv("indices", "target_pos", "inputs"),
  }

  def ffn_fn(activations):
    if activations["state"] == 0:
      # We computed x = 1/(1 + w) where w was the true selector width (excluding
      # BOS value). Here we compute w.
      activations["target_pos"] = round(1 / activations["x"]) - 1
      activations["state"] = 1
    elif activations["state"] == 1:
      # Attention outputs are set to undefined, so must copy to program output.
      activations["program_ouput"] = activations["output"]

  return pb.program_spec(
      variables=variables,
      heads=attenion_heads,
      ffn_fn=ffn_fn,
      output_name="program_ouput",
      input_range=INPUT_RANGE,
      position_range=POSITION_RANGE,
      generate_rules=False,
  )
