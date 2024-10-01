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

"""Implements a CFG parser for SCAN.

This implements only the parsing stage of the overall SCAN program to offer
a simpler illustration of how the overall algorithm works.
"""

from alta.examples.scan import grammar_utils
from alta.examples.scan import scan_utils
from alta.framework import program_builder as pb


EOS_ID = scan_utils.get_input_id("eos")
NUM_POSITIONS = 16
NUM_RULES = scan_utils.NUM_RULES
NUM_SYMBOLS = scan_utils.NUM_SYMBOLS
NUM_INPUT_TOKENS = scan_utils.NUM_INPUT_TOKENS


def get_stack_symbol(symbol_id):
  if symbol_id == 0:
    return None
  else:
    return grammar_utils.get_symbol_token(symbol_id - 1)


def maybe_match_rule(
    input_pointer_token_id, stack_symbol_1, stack_symbol_2, stack_symbol_3
):
  """Check if top of stack matches any rule."""
  # Get string represenations of parsing state.
  input_token_str = scan_utils.get_input_token(input_pointer_token_id)
  stack_symbol_ids = (
      stack_symbol_1,
      stack_symbol_2,
      stack_symbol_3,
  )
  stack_symbols = [
      get_stack_symbol(symbol_id) for symbol_id in stack_symbol_ids
  ]
  for rule in grammar_utils.RULES:
    source_len = len(rule.source)
    stack_source = stack_symbols[:source_len]
    stack_source = tuple(reversed(stack_source))
    if stack_source == rule.source:
      # Rules for `S and S` and `S after S` must be at root.
      if rule.source == ("S", "and", "S") or rule.source == ("S", "after", "S"):
        if input_token_str != "eos":
          continue
      # Matched rule.
      return rule
  return None


def rule_id(rule):
  for idx, rule_b in enumerate(grammar_utils.RULES):
    if rule_b == rule:
      return idx + 1


def rule_len(rule):
  return len(rule.source)


def rule_lhs_id(rule):
  return grammar_utils.get_symbol_id(rule.lhs) + 1


def get_symbol_id(input_id):
  input_string = scan_utils.get_input_token(input_id)
  symbol_id = grammar_utils.get_symbol_id(input_string)
  if symbol_id is None:
    return 0
  else:
    return symbol_id + 1


def shift_stack_pointers(z, stack_pointer_offset: int):
  new_stack_pointer_0 = z["stack_pointer_0"] + stack_pointer_offset
  z["stack_pointer_0"] = max(0, new_stack_pointer_0)
  z["stack_pointer_1"] = max(0, new_stack_pointer_0 - 1)
  z["stack_pointer_2"] = max(0, new_stack_pointer_0 - 2)
  z["stack_pointer_3"] = max(0, new_stack_pointer_0 - 3)


def reduce(z, matched_rule):
  """Implements reduce action."""
  # Pop RHS elements and add LHS nonterminal to stack.
  if z["position"] == (z["stack_pointer_0"] - rule_len(matched_rule)):
    z["symbol_id"] = rule_lhs_id(matched_rule)
  shift_stack_pointers(z, 1 - rule_len(matched_rule))
  # Add rule to parse tree.
  if z["position"] == z["tree_pointer"]:
    # Use 1-indexing to reserve 0 for no rule.
    z["rule_id"] = rule_id(matched_rule)
  z["tree_pointer"] += 1


def shift(z):
  """Implements shift action."""
  # Shift the next token to the stack.
  if z["position"] == z["stack_pointer_0"]:
    z["symbol_id"] = get_symbol_id(z["input_pointer_token_id"])
  shift_stack_pointers(z, 1)
  z["input_pointer"] += 1


def ffn_fn(z):
  """Feed-forward function for the program."""
  if not z["done"]:
    # Check if top-3 stack symbols (and 1 lookahead token) match any rule.
    matched_rule = maybe_match_rule(
        z["input_pointer_token_id"],
        z["stack_symbol_1"],
        z["stack_symbol_2"],
        z["stack_symbol_3"],
    )
    if matched_rule is not None:
      reduce(z, matched_rule)
    else:
      # Check if parsing is complete.
      if z["input_pointer_token_id"] == EOS_ID:
        z["done"] = 1
      else:
        shift(z)


def build_program_spec():
  """Returns a program spec for SCAN task."""
  variables = {
      "token": pb.input_var(NUM_INPUT_TOKENS),
      "position": pb.position_var(NUM_POSITIONS),
      # Whether parsing is complete.
      "done": pb.var(2),
      # Pointer to the next stack position, and then the top 3 elements on
      # the stack.
      "stack_pointer_0": pb.var(NUM_POSITIONS, default=0),
      "stack_pointer_1": pb.var(NUM_POSITIONS, default=0),
      "stack_pointer_2": pb.var(NUM_POSITIONS, default=0),
      "stack_pointer_3": pb.var(NUM_POSITIONS, default=0),
      # Pointer to write the next rule to.
      "tree_pointer": pb.var(NUM_POSITIONS, default=0),
      # Pointer to the next input token to process.
      "input_pointer": pb.var(NUM_POSITIONS, default=0),
      # Stores index of associated parsing rule.
      "rule_id": pb.var(NUM_RULES),
      # Stores symbol ID associated with stack element.
      "symbol_id": pb.var(NUM_SYMBOLS + 1),
  }

  heads = {
      # Get token at input pointer.
      "input_pointer_token_id": pb.qkv("input_pointer", "position", "token"),
      # Get top 3 symbols on stack.
      "stack_symbol_1": pb.qkv("stack_pointer_1", "position", "symbol_id"),
      "stack_symbol_2": pb.qkv("stack_pointer_2", "position", "symbol_id"),
      "stack_symbol_3": pb.qkv("stack_pointer_3", "position", "symbol_id"),
  }

  return pb.program_spec(
      variables=variables,
      heads=heads,
      ffn_fn=ffn_fn,
      output_name="rule_id",
      input_range=scan_utils.NUM_INPUT_TOKENS,
      position_range=NUM_POSITIONS,
      generate_rules=False,
  )
