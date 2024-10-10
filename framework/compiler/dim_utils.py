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

"""Define variable mappings."""

import dataclasses

from framework import program
from framework import var_utils


@dataclasses.dataclass
class CategoricalVarDimMapping:
  """Mapping from a categorical variable to a range of dimensions."""

  start_idx: int
  end_idx: int


@dataclasses.dataclass
class SetVarDimMapping:
  """Mapping from a set variable to a range of dimensions."""

  start_idx: int
  end_idx: int


@dataclasses.dataclass
class NumericalVarDimMapping:
  """Mapping from a numerical variable to a single dimension."""

  idx: int


@dataclasses.dataclass
class ExpandedNumericalVarDimMapping:
  """This mapping is used internally to the implementation of the FFN.

  It represents a numerical variable that has been expanded to
  be represented as a one-hot vector where each dimension corresponds
  to a discretized bucket.
  """

  start_idx: int
  end_idx: int
  buckets: tuple[var_utils.Bucket, ...]


@dataclasses.dataclass
class ExpandedSetVarDimMapping:
  """This mapping is used internally to the implementation of the FFN.

  It represents a set variable as a one-hot vector where each dimension
  corresponds to a possible set.
  """

  start_idx: int
  end_idx: int
  values: tuple[frozenset[int], ...]


VarDimMapping = (
    CategoricalVarDimMapping
    | SetVarDimMapping
    | NumericalVarDimMapping
    | ExpandedNumericalVarDimMapping
    | ExpandedSetVarDimMapping
)


@dataclasses.dataclass
class VarDimMappings:
  """A mapping of all variables to dimensions of an embedding vector.

  This class is overloaded to represent either a mapping where
  numerical variables are represented as a floating point value in
  a single dimension, or where numerical variables are discretized
  to be represented as a one-hot vector, which is used internally
  in the FFN implementation.
  """

  var_mappings: dict[str, VarDimMapping]
  # Start and end dimension indices for overall mapping of all variables.
  start_idx: int
  end_idx: int


def _get_mapping_from_spec(
    var_spec: program.VarSpec, current_idx: int
) -> tuple[VarDimMapping, int]:
  """Get a variable mapping for a given spec."""
  if isinstance(var_spec, program.CategoricalVarSpec):
    start_idx = current_idx
    end_idx = current_idx + var_spec.range
    return (
        CategoricalVarDimMapping(start_idx=start_idx, end_idx=end_idx),
        end_idx,
    )
  elif isinstance(var_spec, program.SetVarSpec):
    start_idx = current_idx
    end_idx = current_idx + var_spec.range
    return (
        SetVarDimMapping(start_idx=start_idx, end_idx=end_idx),
        end_idx,
    )
  elif isinstance(var_spec, program.NumericalVarSpec):
    return NumericalVarDimMapping(idx=current_idx), current_idx + 1
  else:
    raise NotImplementedError()


def _get_expanded_mapping_from_spec(
    var_spec: program.VarSpec, current_idx: int
) -> tuple[VarDimMapping, int]:
  """Get extended mapping for a given spec."""
  if isinstance(var_spec, program.CategoricalVarSpec):
    return _get_mapping_from_spec(var_spec, current_idx)
  elif isinstance(var_spec, program.SetVarSpec):
    end_idx = current_idx + len(var_spec.values)
    return (
        ExpandedSetVarDimMapping(
            start_idx=current_idx,
            end_idx=end_idx,
            values=var_spec.values,
        ),
        end_idx,
    )
  elif isinstance(var_spec, program.NumericalVarSpec):
    buckets = var_utils.get_buckets(var_spec)
    end_idx = current_idx + len(buckets)
    return (
        ExpandedNumericalVarDimMapping(
            start_idx=current_idx, end_idx=end_idx, buckets=buckets
        ),
        end_idx,
    )
  else:
    raise NotImplementedError()


def get_var_mapping(
    program_spec: program.Program, start_idx: int = 0
) -> VarDimMappings:
  """Returns VarDimMappings for `program_spec`."""
  current_idx = start_idx
  var_mappings = {}
  # TODO(petershaw): Make this ordering explicit rather than relying on
  # insertion ordering present in Python 3.6+.
  for var_name, var_spec in program_spec.variables.items():
    var_mappings[var_name], current_idx = _get_mapping_from_spec(
        var_spec, current_idx
    )
  return VarDimMappings(
      var_mappings=var_mappings, start_idx=start_idx, end_idx=current_idx
  )


def get_expanded_var_mapping(
    program_spec: program.Program, start_idx: int = 0
) -> VarDimMappings:
  """Returns mapping with "one-hot" representation for all variables."""
  current_idx = start_idx
  var_mappings = {}
  # TODO(petershaw): Make this ordering explicit rather than relying on
  # insertion ordering present in Python 3.6+.
  for var_name, var_spec in program_spec.variables.items():
    var_mappings[var_name], current_idx = _get_expanded_mapping_from_spec(
        var_spec, current_idx
    )
  return VarDimMappings(
      var_mappings=var_mappings, start_idx=start_idx, end_idx=current_idx
  )
