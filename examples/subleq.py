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

"""Implements SUBLEQ in ALTA.

SUBLEQ is a single instruction language. It has the following instruction:

Instruction subleq a, b, c
  mem[b] = mem[b] - mem[a]
  if (mem[b] <= 0)
    if c < 0:
      break
    else:
      goto c
  else:
    goto a + 3
"""

from framework import program
from framework import program_builder as pb
from framework.mlp import simple_mlp


# Number of memory registers.
NUM_POSITIONS = 16

# Maximum and minimum register values.
MIN_VALUE = -16
MAX_VALUE = 16


def encode(value: int) -> int:
  """Encodes a register value as positive integer."""
  return value - MIN_VALUE


def decode(value: int) -> int:
  """Decodes a register value from positive integer."""
  return value + MIN_VALUE


def encode_inputs(inputs: list[int]) -> list[int]:
  return [encode(x) for x in inputs]


def decode_outputs(outputs: list[int]) -> list[int]:
  return [decode(x) for x in outputs]


STATE_1 = 0
STATE_2 = 1
STATE_3 = 2
STATE_DONE = 3
NUM_STATES = 4


def _update_position(z, position_a):
  z["position_a"] = position_a
  z["position_b"] = position_a + 1
  z["position_c"] = position_a + 2


def _ffn_fn(z: simple_mlp.VarsWrapper):
  """Feed-forward function for SUBLEQ."""
  if z["state"] == STATE_1:
    if decode(z["a"]) < 0 or decode(z["b"]) < 0:
      # Invalid instruction because `a` or `b` are not valid register positions.
      z["state"] = STATE_DONE
      return
    # Otherwise, proceed to next state after `mem_a` and `mem_b` update.
    z["state"] = STATE_2
  elif z["state"] == STATE_2:
    # Update mem[b].
    # mem[b] = mem[b] - mem[a].
    mem_b = decode(z["mem_b"]) - decode(z["mem_a"])
    z["jump"] = int(mem_b <= 0)

    # Update memory value at position `b`.
    if z["position"] == z["b"]:
      z["mem"] = encode(mem_b)

    z["state"] = STATE_3
  elif z["state"] == STATE_3:
    # Determine next instruction.
    if z["jump"]:
      # Jump to instruction `c`.
      if decode(z["c"]) < 0:
        # Break if `c` is negative.
        z["state"] = STATE_DONE
      else:
        _update_position(z, z["c"])
        z["state"] = STATE_1
    else:
      # Proceed to next instruction.
      _update_position(z, z["position_a"] + 3)
      z["state"] = STATE_1


def _get_variables(mem_range, use_clipping=True):
  """Returns a dictionary of variables for SUBLEQ."""
  variables = {
      # Value of register.
      "mem": pb.input_var(mem_range),
      # Position of register.
      "position": pb.position_var(mem_range, init_fn=encode),
      # Position of current instruction.
      "position_a": pb.var(mem_range, default=encode(0)),
      "position_b": pb.var(mem_range, default=encode(1)),
      "position_c": pb.var(mem_range, default=encode(2)),
      # Program state.
      "state": pb.var(NUM_STATES),
      # Whether to jump at next instruction.
      "jump": pb.var(2),
  }
  if use_clipping:
    variables.update({
        # Values of registers at `a` and `b` clipped to positive values to avoid
        # undefined attention outputs.
        "a_clipped": pb.var(mem_range),
        "b_clipped": pb.var(mem_range),
    })
  return variables


def _get_attention_heads(use_clipping=False):
  """Returns a dictionary of attention heads for SUBLEQ."""
  attention_heads = {
      # Values of registers at `position_a`, `position_b`, and `position_c`,
      # which contain the current instruction to execute.
      "a": pb.qkv("position_a", "position", "mem"),
      "b": pb.qkv("position_b", "position", "mem"),
      "c": pb.qkv("position_c", "position", "mem"),
  }
  if use_clipping:
    attention_heads.update({
        # Value of registers at `a` and `b`.
        "mem_a": pb.qkv("a_clipped", "position", "mem"),
        "mem_b": pb.qkv("b_clipped", "position", "mem"),
    })
  else:
    attention_heads.update({
        # Value of registers at `a` and `b`.
        "mem_a": pb.qkv("a", "position", "mem"),
        "mem_b": pb.qkv("b", "position", "mem"),
    })
  return attention_heads


def build_program_spec() -> program.Program:
  """Returns a program spec for SUBLEQ."""

  mem_range = (MAX_VALUE - MIN_VALUE) + 1
  variables = _get_variables(mem_range)
  attention_heads = _get_attention_heads()

  return pb.program_spec(
      variables=variables,
      heads=attention_heads,
      ffn_fn=_ffn_fn,
      output_name="mem",
      input_range=mem_range,
      position_range=NUM_POSITIONS,
      generate_rules=False,
      halt=pb.halt_spec("state", halt_value=STATE_DONE),
  )


def _add_position_update_rules(x, position_a):
  x.set("position_a", position_a)
  x.set("position_b", position_a + 1)
  x.set("position_c", position_a + 2)


def _add_rules(x: pb.MLPBuilder):
  """Feed-forward function for SUBLEQ."""
  for a in x.get("a"):
    x.set("a_clipped", encode(max(0, decode(a))))
  for b in x.get("b"):
    x.set("b_clipped", encode(max(0, decode(b))))

  for state in x.get("state"):
    if state == STATE_1:
      # Still need to wait for `mem_a` and `mem_b` to update, but can check
      # if rule is valid.
      # If `a` or `b` are not valid register positions then instruction is
      # invalid.
      for a, b in x.get("a", "b"):
        if decode(a) < 0 or decode(b) < 0:
          x.set("state", STATE_DONE)
        else:
          x.set("state", STATE_2)
    elif state == STATE_2:
      # Update mem[b].
      # mem[b] = mem[b] - mem[a].
      # If mem[b] <= 0, then jump to mem[c] at next step.
      for mem_a, mem_b in x.get("mem_a", "mem_b"):
        should_jump = (decode(mem_b) - decode(mem_a)) <= 0
        x.set("jump", int(should_jump))

      # Update memory value at position `b`.
      for position, b in x.get("position", "b"):
        if position == b:
          for mem, mem_a in x.get("mem", "mem_a"):
            mem_b = decode(mem) - decode(mem_a)
            x.set("mem", encode(mem_b))

      x.set("state", STATE_3)
    elif state == STATE_3:
      # Determine next instruction.
      for jump in x.get("jump"):
        if jump:
          # Jump to instruction `c`.
          for c in x.get("c"):
            if decode(c) < 0:
              # Break if `c` is negative.
              x.set("state", STATE_DONE)
            else:
              _add_position_update_rules(x, c)
              x.set("state", STATE_1)
        else:
          # Proceed to next instruction.
          for position_a in x.get("position_a"):
            _add_position_update_rules(x, position_a + 3)
          x.set("state", STATE_1)


def build_program_spec_sparse() -> program.Program:
  """Returns a program spec for SUBLEQ."""

  mem_range = (MAX_VALUE - MIN_VALUE) + 1
  variables = _get_variables(mem_range, use_clipping=True)
  attention_heads = _get_attention_heads(use_clipping=True)
  x = pb.MLPBuilder(variables, attention_heads)
  _add_rules(x)
  return pb.program_spec_from_rules(
      variables=variables,
      heads=attention_heads,
      rules=x.rules,
      output_name="mem",
      input_range=mem_range,
      position_range=NUM_POSITIONS,
      halt=pb.halt_spec("state", halt_value=STATE_DONE),
  )
