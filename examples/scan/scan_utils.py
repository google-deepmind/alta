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

"""Utilities for processing SCAN inputs and outputs."""

import functools

from examples.scan import grammar_utils


# Size of stack and memory data structures.
STACK_LENGTH = 8
TREE_LENGTH = 8
# Longest output sequence is 48 tokens.
OUTPUT_LENGTH = 48

# Special input tokens.
SPECIAL_INPUTS = (
    "start",
    "memory",
    "eos",
    "pad",
)

# Sizes of relevant categorical domains.
NUM_RULES = len(grammar_utils.RULES)
NUM_INPUT_TOKENS = len(grammar_utils.SOURCE_TERMINALS) + len(SPECIAL_INPUTS)
NUM_SYMBOLS = (
    1
    + len(grammar_utils.SOURCE_TERMINALS)
    + len(grammar_utils.TARGET_TERMINALS)
    + len(grammar_utils.NONTERMINALS)
)
# Longest input sequence is 9 tokens.
MAX_INPUT_LENGTH = 9


# The difference between the longest output sequence in the test and train
# sets of the length split is 32.
def get_num_positions(max_num_padding: int):
  # Include start and eos symbols.
  return (
      2
      + (STACK_OFFSET - 1)
      + MAX_INPUT_LENGTH
      + STACK_LENGTH
      + TREE_LENGTH
      + OUTPUT_LENGTH
      + max_num_padding
  )

# Offsets relative to start symbol.
STACK_OFFSET = 4
TREE_OFFSET = STACK_OFFSET + STACK_LENGTH
OUTPUT_OFFSET = TREE_OFFSET + TREE_LENGTH
INPUT_OFFSET = OUTPUT_OFFSET + OUTPUT_LENGTH


@functools.cache
def get_input_vocab():
  input_tokens = grammar_utils.SOURCE_TERMINALS + SPECIAL_INPUTS
  return grammar_utils.Vocab(input_tokens)


def get_input_id(input_token):
  if input_token is None:
    return None
  input_vocab = get_input_vocab()
  return input_vocab.token_to_idx[input_token]


def get_input_token(input_id):
  if input_id is None:
    return None
  input_vocab = get_input_vocab()
  return input_vocab.idx_to_token[input_id]


def input_string_to_input_ids(source_string, padding=0):
  """Encode input string as sequence of input IDs."""
  source_tokens = source_string.split()
  input_tokens = (
      ["pad"] * padding
      + ["start"]
      + ["memory"] * (STACK_OFFSET - 1)
      + ["memory"] * STACK_LENGTH
      + ["memory"] * TREE_LENGTH
      + ["memory"] * OUTPUT_LENGTH
      + source_tokens
      + ["eos"]
  )
  input_ids = [get_input_id(token) for token in input_tokens]
  return input_ids


def decode_output(output_ids):
  output_tokens = []
  for output_id in output_ids:
    if output_id != 0:
      token = grammar_utils.get_symbol_token(output_id)
      output_tokens.append(token)
  return output_tokens
