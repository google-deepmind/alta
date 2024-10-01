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

"""Common utilities for working with program variables."""

import dataclasses
import functools

from framework import program


@dataclasses.dataclass(frozen=True)
class Bucket:
  """Represents a discretized bucket for a numerical variable."""

  min_value: float
  max_value: float
  center: float  # Representative value in the middle of the bucket.
  first: bool = False  # Whether this is the "first" bucket.
  last: bool = False  # Whether this is the "last" bucket.


def _mean(value_a, value_b):
  return (value_a + value_b) / 2


def get_buckets(var_spec: program.NumericalVarSpec) -> tuple[Bucket, ...]:
  """Defines a sequence of discretized buckets for a numerical variable."""
  if var_spec.values is None:
    raise ValueError("NumericalVarSpec.values must be set for compilation.")
  num_values = len(var_spec.values)
  buckets = []

  for idx, value in enumerate(var_spec.values):
    if idx == 0:
      first = True
      bucket_min = None
    else:
      first = False
      bucket_min = _mean(value, var_spec.values[idx - 1])

    if idx == num_values - 1:
      last = True
      bucket_max = None
    else:
      last = False
      bucket_max = _mean(value, var_spec.values[idx + 1])

    buckets.append(
        Bucket(
            min_value=bucket_min,
            max_value=bucket_max,
            center=value,
            first=first,
            last=last,
        )
    )
  return tuple(buckets)


@functools.cache
def value_to_int(
    var_spec: program.VarSpec, var_value: program.VarValue
) -> int | None:
  """Return integer representation of a variable value."""
  if var_value is None:
    return None
  if isinstance(var_spec, program.CategoricalVarSpec):
    assert isinstance(var_value, int)
    return var_value
  elif isinstance(var_spec, program.NumericalVarSpec):
    for idx, bucket in enumerate(get_buckets(var_spec)):
      if bucket.min_value is None:
        if var_value <= bucket.max_value:
          return idx
      else:
        if bucket.max_value is None:
          if var_value > bucket.min_value:
            return idx
        else:
          if var_value > bucket.min_value and var_value <= bucket.max_value:
            return idx
    raise ValueError(f"Value {var_value} not in any bucket: {var_spec}")
  elif isinstance(var_spec, program.SetVarSpec):
    for idx, possible_value in enumerate(var_spec.values):
      if var_value == possible_value:
        return idx
    raise ValueError(f"Value {var_value} not in possible values: {var_spec}")
  else:
    raise ValueError(f"Unsupported var spec: {var_spec}")
