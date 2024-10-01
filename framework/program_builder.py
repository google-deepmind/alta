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

"""Defines API used to specify programs."""

import dataclasses
import itertools
from typing import Any, Iterator, Optional

from framework import program
from framework import var_utils
from framework.mlp import mlp_rules
from framework.mlp import simple_mlp
from framework.mlp import sparse_mlp


# Special variable used with relative position attention heads.
RELATIVE_QK = "relative_qk"


@dataclasses.dataclass
class AttentionHead:
  """Helper class for defining attention heads."""

  # Query must be a categorical or set variables.
  query: str
  # Key must be a categorical variable.
  key: str
  # Value can be a categorical or numerical variable.
  value: str
  # Optional mask for relative position attention.
  relative_position_mask: frozenset[int] = frozenset()
  # If None, will infer the output spec from the value spec.
  output_spec: Optional[program.VarSpec] = None


def qkv(
    query: str,
    key: str,
    value: str,
    relative_position_mask=frozenset(),
    output_spec=None,
) -> AttentionHead:
  """Returns an AttentionHead instance."""
  return AttentionHead(
      query=query,
      key=key,
      value=value,
      relative_position_mask=relative_position_mask,
      output_spec=output_spec,
  )


def v_relative(value, position):
  """Convience function for defining relative position attention heads."""
  # Will always attend to the element at the given relative position.
  return qkv(
      query=RELATIVE_QK,
      key=RELATIVE_QK,
      value=value,
      relative_position_mask=frozenset({position}),
  )


def get_specs(
    variables: dict[str, program.VarSpec],
    heads: dict[str, AttentionHead],
):
  """Returns a list of AttentionHeadSpec for the given variables and heads."""
  # Assume this name is not already defined.
  if RELATIVE_QK in variables:
    raise ValueError("Variable %s already defined." % RELATIVE_QK)

  head_specs = []
  var_specs = variables.copy()

  for output_name, head in heads.items():
    if output_name in variables:
      raise ValueError("Variable %s already defined." % output_name)

    # Add a variable used with relative position heads if needed.
    if (
        head.query == RELATIVE_QK or head.key == RELATIVE_QK
    ) and RELATIVE_QK not in variables:
      var_specs[RELATIVE_QK] = program.CategoricalVarSpec(range=1)

    # Add a new variable for output.
    value_spec = var_specs[head.value]
    output_spec = value_spec if head.output_spec is None else head.output_spec
    var_specs[output_name] = output_spec

    # Add a new attention head.
    if isinstance(value_spec, program.NumericalVarSpec):
      head_specs.append(
          program.NumericalAttentionHeadSpec(
              key=head.key,
              value=head.value,
              query=head.query,
              output=output_name,
              relative_position_mask=head.relative_position_mask,
          )
      )
    elif isinstance(value_spec, program.CategoricalVarSpec):
      head_specs.append(
          program.CategoricalAttentionHeadSpec(
              key=head.key,
              value=head.value,
              query=head.query,
              output=output_name,
              relative_position_mask=head.relative_position_mask,
          )
      )
    else:
      raise ValueError("Unsupported value spec: %s" % value_spec)
  return var_specs, head_specs


def get_rules_from_ffn_fn(
    variables: dict[str, program.VarSpec],
    heads: dict[str, AttentionHead],
    ffn_fn: simple_mlp.MLPFunc,
):
  """Naively generate a rule for every possible variable combination."""
  # Note that this can lead to a combinatorial explosion and may be infeasible
  # for programs with many possible variable combinations. In that case,
  # consider using `MLPBuilder` directly when defining the program.
  # You can alternatively use SimpleMLP for running such programs with
  # interpreter, but that does not support compilation.
  # TODO(petershaw): It may be possible to optimize the resulting number of
  # rules by using a more sophisticated approach to rule generation.
  builder = MLPBuilder(variables, heads)
  var_names = builder.var_specs.keys()
  for var_values in builder.get(*var_names):
    activations = {key: value for key, value in zip(var_names, var_values)}
    wrapper = simple_mlp.VarsWrapper(
        builder.var_specs, builder.head_specs, activations
    )
    ffn_fn(wrapper)
    for var_name, var_value_new in wrapper.updates.items():
      builder.set(var_name, var_value_new)
  return builder.rules


def program_spec(
    variables: dict[str, program.VarSpec],
    heads: dict[str, AttentionHead],
    ffn_fn: simple_mlp.MLPFunc,
    output_name: str,
    input_range: int,
    position_range: int | None = None,
    halt: program.HaltSpec | None = None,
    generate_rules: bool = True,
) -> program.Program:
  """Returns program spec given Python function for MLP."""
  var_specs, head_specs = get_specs(variables, heads)
  if generate_rules:
    # Generate a rule for every possible variable combination.
    rules = get_rules_from_ffn_fn(variables, heads, ffn_fn)
    return program_spec_from_rules(
        variables=variables,
        heads=heads,
        rules=rules,
        output_name=output_name,
        input_range=input_range,
        position_range=position_range,
        halt=halt,
    )
  else:
    # Use SimpleMLP for fast iteration with interpreter. Does not support
    # compilation.
    mlp = simple_mlp.SimpleMLP(var_specs, head_specs, ffn_fn)
    return program.Program(
        variables=var_specs,
        input_range=input_range,
        position_range=position_range,
        outputs=output_name,
        head_specs=head_specs,
        mlp=mlp,
        halt_spec=halt,
    )


def program_spec_from_rules(
    variables: dict[str, program.VarSpec],
    heads: dict[str, AttentionHead],
    rules: mlp_rules.RuleSet,
    output_name: str,
    input_range: int,
    position_range: int | None = None,
    halt: program.HaltSpec | None = None,
) -> program.Program:
  """Returns program spec given MLP rules."""
  var_specs, head_specs = get_specs(variables, heads)
  all_rules = rules + get_attention_transition_rules(var_specs, head_specs)
  mlp = sparse_mlp.SparseMLP(var_specs, head_specs, all_rules)
  return program.Program(
      variables=var_specs,
      input_range=input_range,
      position_range=position_range,
      outputs=output_name,
      head_specs=head_specs,
      mlp=mlp,
      halt_spec=halt,
  )


def var(var_range: int, **kwargs):
  """Returns a CategoricalVarSpec with the given range and default value."""
  return program.CategoricalVarSpec(range=var_range, **kwargs)


def input_var(var_range: int, init_fn=None, **kwargs):
  """Returns a CategoricalVarSpec initialized from input token."""
  if init_fn is None:
    init_fn = lambda x: x
  return program.CategoricalVarSpec(
      range=var_range,
      input_init_fn=init_fn,
      **kwargs,
  )


def position_var(var_range: int, init_fn=None, **kwargs):
  """Returns a CategoricalVarSpec initialized from position."""
  if init_fn is None:
    init_fn = lambda x: x
  return program.CategoricalVarSpec(
      range=var_range,
      position_init_fn=init_fn,
      **kwargs,
  )


def numeric_var(**kwargs) -> program.NumericalVarSpec:
  """Returns a NumericalVarSpec with the given range and default value."""
  return program.NumericalVarSpec(**kwargs)


def set_var(**kwargs):
  """Returns a SetVarSpec with the given range and default value."""
  return program.SetVarSpec(**kwargs)


def halt_spec(halt_var: str, halt_value: int = 1):
  """Returns a HaltSpec with the given halt variable and value."""
  return program.HaltSpec(halt_var=halt_var, halt_value=halt_value)


def get_attention_transition_rules(
    var_specs: program.VariablesMap, head_specs: program.HeadSpecs
) -> mlp_rules.RuleSet:
  """Returns rules for attention outputs."""
  # We want to set each attention output to null in the MLP sub-layer
  # so that they can be "written to" by the next attention sub-layer without
  # needing to consider the residual connection.
  rules = []
  for head_spec in head_specs:
    output_spec = var_specs[head_spec.output]
    if isinstance(head_spec, program.CategoricalAttentionHeadSpec):
      assert isinstance(output_spec, program.CategoricalVarSpec)
      for value in range(output_spec.range):
        rules.append(
            mlp_rules.Rule(
                (mlp_rules.LHSAtom(head_spec.output, value),),
                mlp_rules.RHS(
                    head_spec.output, old_value=value, new_value=None
                ),
            )
        )
    elif isinstance(head_spec, program.NumericalAttentionHeadSpec):
      assert isinstance(output_spec, program.NumericalVarSpec)
      for value_idx, value in enumerate(output_spec.values):
        rules.append(
            mlp_rules.Rule(
                (mlp_rules.LHSAtom(head_spec.output, value_idx),),
                mlp_rules.RHS(
                    head_spec.output, old_value=value, new_value=None
                ),
            )
        )
    else:
      raise ValueError("Unsupported head spec: %s" % head_spec)
  return rules


class MLPBuilder:
  """Interface to help build rule set for MLP."""

  var_specs: program.VariablesMap
  head_specs: program.HeadSpecs
  context: dict[str, int]
  rules: mlp_rules.RuleSet

  def __init__(
      self,
      variables: dict[str, program.VarSpec],
      heads: dict[str, AttentionHead],
  ):
    self.var_specs, self.head_specs = get_specs(variables, heads)
    self.rules = []
    self.context = {}

  def _assert_not_in_context(self, var_name: str):
    if var_name in self.context:
      raise ValueError(f"Variable {var_name} is already in context.")

  def _update_context(self, var_name: str, var_value: program.VarValue):
    var_spec = self.var_specs[var_name]
    var_value_idx = var_utils.value_to_int(var_spec, var_value)
    self.context[var_name] = (var_value_idx, var_value)

  def _remove_from_context(self, var_name: str):
    if var_name not in self.context:
      raise ValueError(f"Variable {var_name} is not in context.")
    del self.context[var_name]

  def get(self, *var_names: str) -> Iterator[Any]:
    """Returns a generator that iterates over possible values of variables."""
    for var_name in var_names:
      self._assert_not_in_context(var_name)

    variable_values_list = []
    for var_name in var_names:
      var_spec = self.var_specs[var_name]
      if isinstance(var_spec, program.CategoricalVarSpec):
        variable_values_list.append(range(var_spec.range))
      elif isinstance(var_spec, program.SetVarSpec):
        variable_values_list.append(list(var_spec.values))
      elif isinstance(var_spec, program.NumericalVarSpec):
        variable_values_list.append(list(var_spec.values))
      else:
        raise ValueError("Unsupported var spec: %s" % var_spec)

    for variable_values in itertools.product(*variable_values_list):
      for var_name, var_value in zip(var_names, variable_values):
        self._update_context(var_name, var_value)
      if len(variable_values) == 1:
        yield variable_values[0]
      else:
        yield variable_values

    for var_name in var_names:
      self._remove_from_context(var_name)

  def set(self, var_name: str, new_value: program.VarValue):
    """Sets the value of a variable conditioned on values in current context."""
    if var_name not in self.context:
      # An update must depend on the current value of a variable.
      # Add the current variable to the context if it is missing, so that
      # the antecedent of the rule contains the variable.
      for _ in self.get(var_name):
        self.set(var_name, new_value)
    else:
      _, old_value = self.context[var_name]
      if old_value == new_value:
        return
      lhs = []
      for lhs_var_name, (lhs_var_value_idx, _) in self.context.items():
        lhs.append(mlp_rules.LHSAtom(lhs_var_name, lhs_var_value_idx))
      # Use alphabetical ordering.
      lhs.sort(key=lambda atom: atom.variable)
      lhs = tuple(lhs)
      rhs = mlp_rules.RHS(var_name, old_value, new_value)
      self.rules.append(mlp_rules.Rule(lhs, rhs))
