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

"""Class for MLP specified by a simple Python function.

This class does not support logging of transition rules or compilation,
but can be convienent and more computationally efficient for prototyping
programs with the interpreter.
"""

from typing import Callable, Optional

from alta.framework import program
from alta.framework.mlp import mlp_logger
from alta.framework.mlp import mlp_rules


class VarsWrapper:
  """Represents a set of input variables to MLP layer."""

  def __init__(
      self,
      variables: program.VariablesMap,
      head_specs: program.HeadSpecs,
      activations: program.Activations,
  ):
    self.variables = variables
    self.head_specs = head_specs
    self.activations = activations
    self.updates = {}

  def __getitem__(self, var_name: str) -> program.VarValue:
    if var_name not in self.variables:
      raise ValueError(f"Variable {var_name} not found in program spec.")
    var_value = self.activations[var_name]
    if var_value is None:
      raise ValueError(f"Variable {var_name} is undefined.")
    return var_value

  def __setitem__(self, var_name: str, value: int):
    if var_name not in self.variables:
      raise ValueError(f"Variable {var_name} not found in program spec.")
    if var_name in self.head_specs:
      raise ValueError(f"Variable {var_name} is an attention head output.")
    # TODO(petershaw): Include other type checking on this assignment.
    self.updates[var_name] = value


MLPFunc = Callable[[VarsWrapper], None]


class SimpleMLP(program.BaseMLP):
  """MLP behavior specified as a Python function."""

  def __init__(
      self,
      var_specs: program.VariablesMap,
      head_specs: program.HeadSpecs,
      ffn_fn: MLPFunc,
  ):
    self.var_specs = var_specs
    self.head_specs = head_specs
    self.ffn_fn = ffn_fn

  def run_layer(
      self,
      activations: program.Activations,
      logger: Optional[mlp_logger.MLPLogger] = None,
  ) -> None:
    if logger is not None:
      raise NotImplementedError("Logging not yet implemented for SimpleMLP.")
    wrapper = VarsWrapper(self.var_specs, self.head_specs, activations)
    self.ffn_fn(wrapper)
    for var_name, value in wrapper.updates.items():
      activations[var_name] = value

  def get_rules(self) -> mlp_rules.RuleSet:
    raise NotImplementedError()
