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

"""Defines a program specification."""

import dataclasses
from typing import Callable, Dict, Optional, Union

from alta.framework.mlp import mlp_logger
from alta.framework.mlp import mlp_rules


# Categorical variables are represented as one-hot vectors.
@dataclasses.dataclass(frozen=True)
class CategoricalVarSpec:
  # Range of possible values (for allocating dimensions).
  range: int = 32
  # Default value. Attention output variables are initialized to None regardless
  # of this value.
  default: int = 0
  # If specified, used to initialize default value as function of input id.
  input_init_fn: Callable[[int], int] | None = None
  # If specified, used to initialize default value as function of position.
  position_init_fn: Callable[[int], int] | None = None


@dataclasses.dataclass(frozen=True)
class NumericalVarSpec:
  """Numerical variables are represented as a float."""
  # These determine how numerical variables are discretized for compilation.
  values: tuple[float, ...] | None = None  # Should be ordered.
  # Default value. Attention output variables are initialized to None regardless
  # of this value.
  default: float = 0.0
  # If specified, used to initialize default value as function of input id.
  input_init_fn: Callable[[int], int] | None = None
  # If specified, used to initialize default value as function of position.
  position_init_fn: Callable[[int], int] | None = None


@dataclasses.dataclass(frozen=True)
class SetVarSpec:
  """Special type of variable used to represent attention head queries."""

  # Default value.
  default: frozenset[int] = frozenset()
  # Conceptually these represent a set of categorical values.
  # Range of possible values for members of the set.
  range: int = 32
  # Possible set assignments.
  values: tuple[frozenset[int], ...] | None = None
  # If specified, used to initialize default value as function of input id.
  input_init_fn: Callable[[int], frozenset[int]] | None = None
  # If specified, used to initialize default value as function of position.
  position_init_fn: Callable[[int], frozenset[int]] | None = None


VarSpec = Union[CategoricalVarSpec, NumericalVarSpec, SetVarSpec]

NumericalVarValue = float
CategoricalVarValue = int
SetVarValue = frozenset[int]

# None corresponds to zero vector in compiled programs.
# TODO(petershaw): Differentiate between `undefined` and `null`.
VarValue = Union[CategoricalVarValue, NumericalVarValue, SetVarValue, None]


@dataclasses.dataclass(frozen=True)
class CategoricalAttentionHeadSpec:
  """Categorical attention head spec.

  For convience, we define two types of attention heads. A "categorical"
  attention head aggregates over categorical values, and will throw an exception
  if more than one value is "selected" by an attention head.

  Attributes:
    query: Name of var with CategoricalVarSpec or SetVarSpec.
    key: Name of var with CategoricalVarSpec where `range` equals that of the
      query.
    value: Name of var with CategoricalVarSpec.
    output: Name of var with CategoricalVarSpec and `range` equal to that of the
      value.
    relative_position_mask: Optional set of relative positions that can be
      attended to. E.g. if `relative_position_mask` is {-1, 1} then it can
      attend to the previous and next position. If unspecified, all positions
      are attended to.
  """
  query: str
  key: str
  value: str
  output: str
  relative_position_mask: frozenset[int] = frozenset()


@dataclasses.dataclass(frozen=True)
class NumericalAttentionHeadSpec:
  """Numerical attention head spec.

  A numerical attention head will output the mean of the "selected" values.

  Attributes:
    query: Name of var with CategoricalVarSpec or SetVarSpec.
    key: Name of var with CategoricalVarSpec where `range` equals that of the
      query.
    value: Name of var with NumericalVarSpec.
    output: Name of var with NumericalVarSpec.
    relative_position_mask: Optional set of relative positions that can be
      attended to. E.g. if `relative_position_mask` is {-1, 1} then it can
      attend to the previous and next position. If unspecified, all positions
      are attended to.
  """
  query: str
  key: str
  value: str
  output: str
  relative_position_mask: frozenset[int] = frozenset()


AttentionHeadSpec = Union[
    CategoricalAttentionHeadSpec, NumericalAttentionHeadSpec
]


# Activations represent the state of a single element before and after
# self-attention and the element-wise feed-forward network.
# This maps conceptually to an embedding, where different dimensions of the
# embedding conceptually relate to different variables.
# A sequence of Activations relates to the Transformer state between layers.
Activations = Dict[str, VarValue]

# Map of variables to their specifications.
VariablesMap = Dict[str, VarSpec]

# List of attention head specifications.
HeadSpecs = list[AttentionHeadSpec]


@dataclasses.dataclass(frozen=True)
class HaltSpec:
  """Spec for dynamic halting of computation."""

  # Name of categorical variable that indicates halting.
  halt_var: str
  # Value of the variable that indicates halting. All activations must have this
  # value for program to halt.
  halt_value: int = 1


class BaseMLP:
  """Base class for MLP specification."""
  var_specs: VariablesMap
  head_specs: HeadSpecs

  def run_layer(
      self,
      activations: Activations,
      logger: Optional[mlp_logger.MLPLogger] = None,
  ) -> None:
    raise NotImplementedError()

  def get_rules(self) -> mlp_rules.RuleSet:
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class Program:
  """Defines a program specification."""
  variables: VariablesMap
  input_range: int
  position_range: int | None
  # Outputs can be either categorical or numeric.
  outputs: str
  # Defines set of attention heads.
  head_specs: HeadSpecs
  # Defines function of MLP sub-layer.
  mlp: BaseMLP
  # Optional spec for dynamic halting.
  halt_spec: HaltSpec | None = None
