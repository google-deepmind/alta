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

"""Parses all examples in SCAN dataset file."""

from absl import app
from absl import flags

from alta.examples.scan import data_utils
from alta.examples.scan import scan_sparse_program
from alta.examples.scan import scan_utils
from alta.framework.interpreter import interpreter_utils
from alta.framework.interpreter import program_utils


_INPUT = flags.DEFINE_string(
    "input",
    "",
    "Path to scan dataset file.",
)

_OFFSET = flags.DEFINE_integer(
    "offset", 0, "Example index to start processing at."
)

_LIMIT = flags.DEFINE_integer("limit", 10, "Number of examples to process.")

_POSITION_SHIFT = flags.DEFINE_integer(
    "position_shift", 0, "Offset for positional indexes."
)


def get_output_string(input_string):
  """Returns output string for given input string."""
  program_spec = scan_sparse_program.build_program_spec()
  print("input_string: %s" % str(input_string))
  input_ids = scan_utils.input_string_to_input_ids(input_string)
  print("input_ids: %s" % str(input_ids))
  activations_seq = program_utils.initialize_activations(
      program_spec,
      input_ids,
      position_shift=_POSITION_SHIFT.value,
  )
  outputs = interpreter_utils.run_transformer(
      program_spec,
      activations_seq,
      max_layers=128,
      logger=None,
  )
  output_tokens = scan_utils.decode_output(outputs)
  return " ".join(output_tokens)


def main(unused_argv):
  examples = data_utils.load_examples(_INPUT.value)
  print("len(examples): %s" % len(examples))
  if _OFFSET.value:
    examples = examples[_OFFSET.value :]
  if _LIMIT.value:
    examples = examples[: _LIMIT.value]

  for idx, (input_string, output_string) in enumerate(examples):
    print("idx: %s" % (idx + _OFFSET.value))
    print("input_string: %s" % input_string)
    print("output_string: %s" % output_string)
    predicted_string = get_output_string(input_string)
    print("predicted_string: %s" % predicted_string)
    if predicted_string != output_string:
      raise ValueError("Mismatch: %s %s" % (input_string, output_string))


if __name__ == "__main__":
  app.run(main)
