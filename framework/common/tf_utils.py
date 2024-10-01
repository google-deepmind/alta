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

"""Some utilities for TF data files."""

from collections.abc import Sequence

import tensorflow as tf


INPUT_FEATURE = "inputs"
OUTPUT_FEATURE = "targets"


def add_bytes_feature(
    example: tf.train.Example, key: str, value: bytes
) -> None:
  example.features.feature[key].bytes_list.value.append(value)


def add_text_feature(example: tf.train.Example, key: str, value: str) -> None:
  add_bytes_feature(example, key, value.encode("utf-8"))


def add_int_feature(example: tf.train.Example, key: str, value: int) -> None:
  """Appends int feature with given `key` from `example`."""
  example.features.feature[key].int64_list.value.append(value)


def add_int_list_feature(
    example: tf.train.Example, key: str, values: list[int]
) -> None:
  """Appends int `values` to `key` in `example`."""
  example.features.feature[key].int64_list.value.extend(values)


def add_float_list_feature(
    example: tf.train.Example, key: str, values: list[float]
) -> None:
  """Appends float `values` to `key` in `example`."""
  example.features.feature[key].float_list.value.extend(values)


def get_bytes_feature(example: tf.train.Example, key: str) -> bytes:
  return example.features.feature[key].bytes_list.value[0]


def get_text_feature(example: tf.train.Example, key: str) -> str:
  return get_bytes_feature(example, key).decode("utf-8")


def get_int_list_feature(example: tf.train.Example, key: str) -> Sequence[int]:
  """Returns int list feature with given `key` from `example`."""
  return example.features.feature[key].int64_list.value


def create_example(source, target) -> tf.train.Example:
  example = tf.train.Example()
  add_text_feature(example, INPUT_FEATURE, source)
  add_text_feature(example, OUTPUT_FEATURE, target)
  return example
