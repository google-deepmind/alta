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

"""Defines transitions rules for MLP layer."""

import dataclasses

# TODO(petershaw): Reduce duplication with `program.py` without adding
# circular dependencies.
VarValue = int | float | frozenset[int]


@dataclasses.dataclass(frozen=True)
class LHSAtom:
  """Represents a tuple of program variable and value."""

  variable: str
  # Bucketed value of the variable.
  value_idx: int


@dataclasses.dataclass(frozen=True)
class RHS:
  """Represents an update to a program variable."""

  variable: str
  old_value: VarValue
  # None corresponds to `null` used to reset attention outputs.
  new_value: VarValue | None


# Represents a conjunction of atoms.
LHS = tuple[LHSAtom, ...]


@dataclasses.dataclass(frozen=True)
class Rule:
  """Represents a definite clause for computing a value."""

  # The antecdent of the implication, which is a conjunction of atoms.
  lhs: LHS
  # The consequent of the implication.
  rhs: RHS


# Represents a set of rules.
RuleSet = list[Rule]
