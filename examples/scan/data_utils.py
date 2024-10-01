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

"""Utilities for loading and parsing SCAN data."""

from framework.common import io_utils


TARGET_MAP = {
    "I_TURN_RIGHT": "RTURN",
    "I_TURN_LEFT": "LTURN",
    "I_WALK": "WALK",
    "I_LOOK": "LOOK",
    "I_RUN": "RUN",
    "I_JUMP": "JUMP",
}


def parse_example(row: str):
  """Parse a single example from SCAN data file."""
  splits = row.split("OUT:")
  # Trim "IN:" prefix.
  input_string = splits[0].removeprefix("IN:").strip()
  output_string = splits[1].strip()
  # Replace tokens with shorter versions.
  for original, replacement in TARGET_MAP.items():
    output_string = output_string.replace(original, replacement)
  return input_string, output_string


def load_examples(path):
  """Load examples from SCAN data file."""
  rows = io_utils.read_txt(path)
  examples = []
  for row in rows:
    input_string, output_string = parse_example(row)
    examples.append((input_string, output_string))
  return examples
