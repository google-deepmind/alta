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

"""Library for loading data."""

import functools
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def decode_fn(record_bytes, vector_length, include_debug=False):
  """Decodes a single example."""
  schema = {
      "input": tf.io.FixedLenFeature([vector_length], dtype=tf.float32),
      "output": tf.io.FixedLenFeature([vector_length], dtype=tf.float32),
  }
  if include_debug:
    schema["model_input"] = tf.io.VarLenFeature(tf.int64)

  input_example = tf.io.parse_single_example(
      record_bytes,
      schema,
  )

  output_example = {
      "input": input_example["input"],
      "output": input_example["output"],
  }
  if include_debug:
    output_example["model_input"] = tf.sparse.to_dense(
        input_example["model_input"]
    )

  return output_example


def get_batches(path, vector_length, batch_size, shuffle_buffer_size=10_000):
  """Returns batched data."""
  ds = tf.data.TFRecordDataset(tf.io.gfile.glob(path))
  ds = ds.map(functools.partial(decode_fn, vector_length=vector_length))
  ds = (
      ds.shuffle(shuffle_buffer_size)
      .batch(batch_size, drop_remainder=True)
      .prefetch(1)
  )
  return tfds.as_numpy(ds)


def get_all_data(path, vector_length, sample_size=None, include_debug=False):
  """Returns all data in at `path`."""
  ds = tf.data.TFRecordDataset(tf.io.gfile.glob(path))
  ds = ds.map(
      functools.partial(
          decode_fn, vector_length=vector_length, include_debug=include_debug
      )
  )
  if sample_size:
    ds = ds.take(sample_size)
  ds = tfds.as_numpy(ds)
  # TODO(jamesfcohan): This code does not work if dataset is shuffled. I.e.
  # input and output will come from different examples. Change it to traverse
  # `ds` just once, returning `input`, `output`, and `model_input` at the same
  # time.
  inputs = np.array([example["input"] for example in ds])
  outputs = np.array([example["output"] for example in ds])
  if include_debug:
    model_inputs = [example["model_input"] for example in ds]
    return inputs, outputs, model_inputs
  return inputs, outputs
