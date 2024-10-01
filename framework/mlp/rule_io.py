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

"""Utilities for reading and writing sets of rules."""

import dataclasses
import json
from typing import Any
import tensorflow as tf
from alta.framework.common import io_utils
from alta.framework.mlp import mlp_rules


def rule_to_json(rule: mlp_rules.Rule) -> str:
  """Serializes a Rule object to a JSON string."""

  def _serialize_var_value(value: mlp_rules.VarValue) -> Any:
    if isinstance(value, frozenset):
      return list(value)
    return value

  return json.dumps({
      "lhs": [dataclasses.asdict(atom) for atom in rule.lhs],
      "rhs": {
          "variable": rule.rhs.variable,
          "old_value": _serialize_var_value(rule.rhs.old_value),
          "new_value": _serialize_var_value(rule.rhs.new_value),
      },
  })


def rule_from_dict(data: dict[str, Any]) -> mlp_rules.Rule:
  """Deserializes a Rule object from a dictionary."""
  def _deserialize_var_value(value: Any) -> mlp_rules.VarValue:
    if isinstance(value, list):
      return frozenset(value)
    return value

  return mlp_rules.Rule(
      lhs=tuple(mlp_rules.LHSAtom(**atom) for atom in data["lhs"]),
      rhs=mlp_rules.RHS(
          variable=data["rhs"]["variable"],
          old_value=_deserialize_var_value(data["rhs"]["old_value"]),
          new_value=_deserialize_var_value(data["rhs"]["new_value"]),
      ),
  )


def rule_from_json(json_str: str) -> mlp_rules.Rule:
  """Deserializes a Rule object from a JSON string."""
  data = json.loads(json_str)
  return rule_from_dict(data)


def write_rules(rules: mlp_rules.RuleSet, output_path: str):
  """Writes a set of rules to a JSON file."""
  io_utils.write_jsonl(
      output_path,
      [rule_to_json(rule) for rule in rules],
  )


def read_rules(input_path: str) -> mlp_rules.RuleSet:
  """Reads a set of rules from a JSON file."""
  return [
      rule_from_dict(rules_dict)
      for rules_dict in io_utils.read_jsonl(input_path)
  ]


def read_sharded_rules(input_path: str) -> mlp_rules.RuleSet:
  paths = tf.io.gfile.glob(input_path)
  if not paths:
    raise ValueError(f"No files found matching {input_path}")
  rules = []
  for path in paths:
    rules.extend(read_rules(path))
  return rules
