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

"""Implements multi-digit arithmetic.

Assumes inputs have been padded to equal length.

The algorithm initializes a pointer to the first digit of each input, and a
pointer to an output buffer. The algorithm then iterates over the digits,
adding the current digits, adding the value to the output buffer, and updating
the carry value. When the last digit has been processed, the final carry value
is added to the output buffer.
"""

from alta.framework import program_builder as pb

# Define processing steps.
STEP_INIT = 0
STEP_ITERATE = 1
STEP_FINALIZE = 2
STEP_DONE = 3
NUM_STEPS = 4

# Define token vocabulary special tokens.
# We reserve 0-9 to represent digits, and start special tokens at 10.
START_TOKEN = 10
ADD_TOKEN = 11
END_TOKEN = 12

# Number of unique inputs includes digits 0-9 and 3 special tokens.
INPUT_RANGE = 13


def format_input(input_a: int, input_b: int):
  """Used to generate model input string."""
  # Pad to equal length.
  input_length = max(len(str(input_a)), len(str(input_b)))
  input_a = str(input_a).zfill(input_length)
  input_b = str(input_b).zfill(input_length)
  return "s" + input_a + "+" + input_b + "e"


def get_vocab():
  """Returns a vocabulary mapping from tokens to token IDs."""
  vocab = {}
  for i in range(10):
    vocab[str(i)] = i
  vocab["s"] = START_TOKEN
  vocab["+"] = ADD_TOKEN
  vocab["e"] = END_TOKEN
  return vocab


def preprocess_input(input_a: int, input_b: int):
  """Converts input to a list of token IDs."""
  input_string = format_input(input_a, input_b)
  vocab = get_vocab()
  input_ids = [vocab[c] for c in input_string]
  return input_ids


def process_output(output_ids):
  """Converts output to a final result."""
  # Trim end token.
  output_ids = output_ids[:-1]
  return int("".join([str(x) for x in output_ids]))


def init(x):
  """Initializes pointers."""
  # If `x["token"] == END_TOKEN` then `x["token_right"]` will be undefined.
  if x["token"] != END_TOKEN:
    if x["token_right"] == END_TOKEN:
      x["ptr_b"] = 1
      x["ptr_out"] = 1
    if x["token_right"] == ADD_TOKEN:
      x["ptr_a"] = 1


def iterate(x):
  """Execute one step of addition."""
  # Compute sum.
  raw_sum = x["value_carry"] + x["ptr_a_token"] + x["ptr_b_token"]
  if x["ptr_out"]:
    x["value_out"] = raw_sum % 10
  x["value_carry"] = raw_sum // 10

  # Move all pointers to the left.
  # Attention heads attending to the right will be undefined.
  if x["token"] != END_TOKEN:
    x["ptr_out"] = x["ptr_out_right"]
    x["ptr_a"] = x["ptr_a_right"]
    x["ptr_b"] = x["ptr_b_right"]


def finalize(x):
  """Finalize output by adding the final carry to the output."""
  if x["ptr_out"]:
    x["value_out"] = x["value_carry"]
  x["step"] = STEP_DONE


def ffn_fn(x):
  if x["step"] == STEP_INIT:
    init(x)
    x["step"] = STEP_ITERATE
  elif x["step"] == STEP_ITERATE:
    if x["ptr_a_token"] == START_TOKEN:
      x["step"] = STEP_FINALIZE
    else:
      iterate(x)
  elif x["step"] == STEP_FINALIZE:
    finalize(x)


def get_variables():
  """Returns a dictionary of variables."""
  variables = {
      "token": pb.input_var(INPUT_RANGE),
      # This variable tracks the current processing step.
      "step": pb.var(NUM_STEPS),
      # These are pointers to which digit is currently being processed.
      # They are `1` at the position of the current digit to process, and `0`
      # otherwise.
      "ptr_a": pb.var(2),
      "ptr_b": pb.var(2),
      # This pointer is `1` at the position to write the next output to,
      # and `0` otherwise.
      "ptr_out": pb.var(2),
      # This tracks the "carry" value form the previous iteration.
      "value_carry": pb.var(10),
      # This tracks the final output value for a given digit.
      "value_out": pb.var(10),
      # Static variables used as attention query.
      "one": pb.var(var_range=2, default=1),
  }
  return variables


def get_attention_heads():
  """Returns a dictionary of attention heads."""
  attention_heads = {
      # For these relative attention heads, we always want to attend to the
      # position immediately to the right.
      "token_right": pb.v_relative("token", 1),
      "ptr_a_right": pb.v_relative("ptr_a", 1),
      "ptr_b_right": pb.v_relative("ptr_b", 1),
      "ptr_out_right": pb.v_relative("ptr_out", 1),
      # For these attention heads, we want to attend to the positions associated
      # with the current pointers.
      "ptr_a_token": pb.qkv("one", "ptr_a", "token"),
      "ptr_b_token": pb.qkv("one", "ptr_b", "token"),
  }
  return attention_heads


def build_program_spec():
  """Returns a program spec for addition task."""
  variables = get_variables()
  attention_heads = get_attention_heads()
  return pb.program_spec(
      variables=variables,
      heads=attention_heads,
      ffn_fn=ffn_fn,
      output_name="value_out",
      input_range=INPUT_RANGE,
      position_range=None,
      halt=pb.halt_spec("step", halt_value=STEP_DONE),
      generate_rules=False,
  )


def add_init_rules(x):
  """Initializes pointers."""
  for token, token_right in x.get("token", "token_right"):
    if token != END_TOKEN:
      if token_right == END_TOKEN:
        x.set("ptr_b", 1)
        x.set("ptr_out", 1)
      if token_right == ADD_TOKEN:
        x.set("ptr_a", 1)


def add_iterate_rules(x, ptr_a_token):
  """Execute one step of addition."""
  # Compute sum.
  for value_carry, ptr_b_token in x.get("value_carry", "ptr_b_token"):
    raw_sum = value_carry + ptr_a_token + ptr_b_token
    for ptr_out in x.get("ptr_out"):
      if ptr_out:
        x.set("value_out", raw_sum % 10)
    x.set("value_carry", raw_sum // 10)

  # Move all pointers to the left.
  # Attention heads attending to the right will be undefined.
  for token in x.get("token"):
    if token != END_TOKEN:
      for ptr_out_right in x.get("ptr_out_right"):
        x.set("ptr_out", ptr_out_right)
      for ptr_a_right in x.get("ptr_a_right"):
        x.set("ptr_a", ptr_a_right)
      for ptr_b_right in x.get("ptr_b_right"):
        x.set("ptr_b", ptr_b_right)


def add_rules(x: pb.MLPBuilder):
  """Add transition rules."""
  for step in x.get("step"):
    if step == STEP_INIT:
      add_init_rules(x)
      x.set("step", STEP_ITERATE)
    elif step == STEP_ITERATE:
      for ptr_a_token in x.get("ptr_a_token"):
        if ptr_a_token == START_TOKEN:
          x.set("step", STEP_FINALIZE)
        else:
          add_iterate_rules(x, ptr_a_token)
    elif step == STEP_FINALIZE:
      for ptr_out in x.get("ptr_out"):
        if ptr_out:
          for value_carry in x.get("value_carry"):
            x.set("value_out", value_carry)
      x.set("step", STEP_DONE)


def build_sparse_program_spec():
  """Returns a sparse program spec."""
  variables = get_variables()
  attention_heads = get_attention_heads()
  x = pb.MLPBuilder(variables, attention_heads)
  add_rules(x)
  return pb.program_spec_from_rules(
      variables=variables,
      heads=attention_heads,
      rules=x.rules,
      output_name="value_out",
      input_range=INPUT_RANGE,
      position_range=None,
      halt=pb.halt_spec("step", halt_value=STEP_DONE),
  )
