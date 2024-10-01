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

"""Utilties for reading and writing various file formats."""

import json
from typing import Iterable, Iterator

import tensorflow as tf


def write_txt(rows, filepath):
  """Write newline separated text file."""
  with tf.io.gfile.GFile(filepath, "w") as tsv_file:
    for row in rows:
      line = "%s\n" % row
      tsv_file.write(line)
  print("Wrote %s rows to %s." % (len(rows), filepath))


def read_txt(filepath):
  """Read newline separated text file."""
  rows = []
  with tf.io.gfile.GFile(filepath, "r") as tsv_file:
    for line in tsv_file:
      line = line.rstrip()
      rows.append(line)
  print("Loaded %s rows from %s." % (len(rows), filepath))
  return rows


def read_tsv(path):
  """Read tsv file to list of rows."""
  rows = []
  with tf.io.gfile.GFile(path, "r") as tsv_file:
    for line in tsv_file:
      line = line.rstrip()
      cols = line.split("\t")
      rows.append(cols)
  print("Loaded %s rows from %s." % (len(rows), path))
  return rows


def write_tsv(rows, filepath, delimiter="\t"):
  """Write rows to tsv file."""
  with tf.io.gfile.GFile(filepath, "w") as tsv_file:
    for row in rows:
      line = "%s\n" % delimiter.join([str(elem) for elem in row])
      tsv_file.write(line)
  print("Wrote %s rows to %s." % (len(rows), filepath))


def read_jsonl(filepath):
  """Read jsonl file to a List of Dicts."""
  data = []
  with tf.io.gfile.GFile(filepath, "r") as jsonl_file:
    for line in jsonl_file:
      data.append(json.loads(line))
  print("Loaded %s lines from %s." % (len(data), filepath))
  return data


def write_jsonl(filepath, rows):
  """Write a List of Dicts to jsonl file."""
  with tf.io.gfile.GFile(filepath, "w") as jsonl_file:
    for row in rows:
      line = "%s\n" % json.dumps(row)
      jsonl_file.write(line)
  print("Wrote %s lines to %s." % (len(rows), filepath))


def read_tfrecords(filepath: str) -> Iterator[tf.train.Example]:
  """Reads tfrecords from sharded files.

  Args:
    filepath: a sharded path with a 'tfrecord.*' suffix.

  Yields:
    A tfexample read from the file.
  """
  dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(filepath))
  for raw_record in dataset:
    example = tf.train.Example.FromString(raw_record.numpy())
    yield example


def write_tfrecords(
    tf_examples: Iterable[tf.train.Example], output_path: str
) -> None:
  """Writes tfrecords to sharded files.

  Args:
    tf_examples: The examples to write to file.
    output_path: The path to write the examples to.
  """
  with tf.io.TFRecordWriter(output_path) as tfrecord_writer:
    for example in tf_examples:
      tfrecord_writer.write(example.SerializeToString())  # pytype: disable=wrong-arg-types
