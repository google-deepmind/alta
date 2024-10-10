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

"""MLP specificied by a set of rules."""

from collections.abc import Callable

from framework import program
from framework import var_utils
from framework.mlp import mlp_logger
from framework.mlp import mlp_rules


def _get_lhs_variables(rule: mlp_rules.Rule) -> tuple[str, ...]:
  return tuple([atom.variable for atom in rule.lhs])


def _get_lhs_values(rule: mlp_rules.Rule) -> tuple[int, ...]:
  return tuple([atom.value_idx for atom in rule.lhs])


def _get_activations_values(
    var_specs: program.VariablesMap,
    activations: program.Activations,
    variables: tuple[str, ...],
) -> tuple[int | None, ...]:
  """Returns values of variables in activations."""
  # Convert set and numeric variables to integers based on their possible
  # values.
  values = []
  for variable in variables:
    var_spec = var_specs[variable]
    var_value = activations[variable]
    values.append(var_utils.value_to_int(var_spec, var_value))
  return tuple(values)


def get_ffn_fn(
    var_specs: program.VariablesMap,
    rules: mlp_rules.RuleSet,
) -> Callable[[program.Activations, mlp_logger.MLPLogger | None], None]:
  """Returns a MLP function that implements the given rules."""

  # Map of variable names to map of variable values to rules.
  lookup_table = {}
  for rule in rules:
    lhs_variables = _get_lhs_variables(rule)
    lhs_values = _get_lhs_values(rule)
    if lhs_variables not in lookup_table:
      lookup_table[lhs_variables] = {}
    if lhs_values not in lookup_table[lhs_variables]:
      lookup_table[lhs_variables][lhs_values] = []
    lookup_table[lhs_variables][lhs_values].append(rule)

  def ffn_fn(
      activations: program.Activations,
      logger: mlp_logger.MLPLogger | None = None,
  ) -> None:
    # Determine which rules match the given input.
    updates = {}
    for variables in lookup_table:
      activations_values = _get_activations_values(
          var_specs, activations, variables
      )
      rule_list = lookup_table[variables].get(activations_values)
      if rule_list is not None:
        for rule in rule_list:
          rhs = rule.rhs
          if logger:
            logger.add(rule)
          if rhs.variable in updates:
            raise RuntimeError(
                f"Variable {rhs.variable} is updated multiple times."
            )
          updates[rhs.variable] = rhs.new_value

    # Apply updates for matched rules.
    for var, value in updates.items():
      # TODO(petershaw): Consider applying some additional type checking.
      activations[var] = value

  return ffn_fn


class SparseMLP(program.BaseMLP):
  """MLP behavior specified as a set of transition rules."""

  def __init__(
      self,
      var_specs: program.VariablesMap,
      head_specs: program.HeadSpecs,
      rules: mlp_rules.RuleSet,
  ):
    self.var_specs = var_specs
    self.head_specs = head_specs
    self.rules = rules
    self.ffn_fn = get_ffn_fn(var_specs, rules)

  def run_layer(
      self,
      activations: program.Activations,
      logger: mlp_logger.MLPLogger | None = None,
  ) -> None:
    self.ffn_fn(activations, logger)

  def get_rules(self) -> mlp_rules.RuleSet:
    return self.rules
