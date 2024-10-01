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

"""Utilities for logging activations in interpeter."""


class ActivationsLogger:
  """Logs activations in interpeter."""

  def __init__(self):
    self.initial_activations = None
    # Tuples of (mlp_input, mlp_output).
    self.layer_activations = []

  def set_initial_activations(self, activations):
    self.initial_activations = activations

  def add_layer_activations(self, mlp_in, mlp_out):
    self.layer_activations.append((mlp_in, mlp_out))

  def get_variable_values(self, element_idx, variable_name):
    values = []
    assert self.initial_activations is not None
    values.append(self.initial_activations[element_idx][variable_name])
    for mlp_input, mlp_output in self.layer_activations:
      values.append(mlp_input[element_idx][variable_name])
      values.append(mlp_output[element_idx][variable_name])
    return values

  def get_activations_table(
      self, elements_to_include=None, variables_to_include=None
  ):
    """Returns list of lists."""
    assert self.initial_activations is not None
    num_elements = len(self.initial_activations)
    # Assume all layers and elements have same variables.
    variables = self.initial_activations[0].keys()
    if variables_to_include is not None:
      variables = [x for x in variables if x in variables_to_include]

    table = []
    # Add header row.
    header = ["Element", "Variable", "Input"]
    for layer_idx in range(len(self.layer_activations)):
      header.append(f"L{layer_idx}a")
      header.append(f"L{layer_idx}b")
    table.append(header)

    # Add a row for every element and variable.
    for element_idx in range(num_elements):
      if (
          elements_to_include is not None
          and element_idx not in elements_to_include
      ):
        continue
      for variable_name in variables:
        if variables is not None and variable_name not in variables:
          continue
        row = [element_idx, variable_name]
        row.extend(self.get_variable_values(element_idx, variable_name))
        table.append(row)

    return table

  def _format_value(self, value):
    if callable(value):
      return "Q"
    elif value is None:
      return "-"
    else:
      return str(value)

  def print_activations_table(
      self, sep=",", elements_to_include=None, variables_to_include=None
  ):
    table = self.get_activations_table(
        elements_to_include=elements_to_include,
        variables_to_include=variables_to_include,
    )
    for row in table:
      row_values = [self._format_value(x) for x in row]
      print(sep.join(row_values))

  def get_layer_activations(self):
    return self.layer_activations
