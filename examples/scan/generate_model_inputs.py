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

"""Write jsonl file for model inputs.

Input is the original SCAN dataset files from:
https://github.com/brendenlake/SCAN
"""

from absl import app
from absl import flags

from alta.examples.scan import data_utils
from alta.examples.scan import scan_utils
from alta.framework.common import io_utils


_INPUT = flags.DEFINE_string(
    "input",
    "",
    "Path to SCAN dataset file.",
)

_OUTPUT = flags.DEFINE_string(
    "output",
    "",
    "Path to write jsonl file.",
)

_SAMPLE = flags.DEFINE_integer(
    "sample",
    0,
    "Sample only this many examples if > 0.",
)

_NUM_PADDING = flags.DEFINE_integer(
    "num_padding",
    0,
    "Add variable length padding to model inputs if > 0.",
)


def get_paddings():
  if _NUM_PADDING.value > 0:
    return range(_NUM_PADDING.value)
  else:
    return [0]


def get_model_inputs(examples):
  """Get set of unique model inputs."""
  model_inputs_set = set()
  for input_string, _ in examples:
    for padding in get_paddings():
      model_input = scan_utils.input_string_to_input_ids(
          input_string, padding=padding
      )
      model_inputs_set.add(tuple(model_input))

  model_inputs = []
  for model_input_tuple in model_inputs_set:
    model_inputs.append(list(model_input_tuple))
  return model_inputs


def main(unused_argv):
  examples = data_utils.load_examples(_INPUT.value)
  if _SAMPLE.value > 0:
    examples = examples[: _SAMPLE.value]
  model_inputs = get_model_inputs(examples)
  io_utils.write_jsonl(_OUTPUT.value, model_inputs)


if __name__ == "__main__":
  app.run(main)
