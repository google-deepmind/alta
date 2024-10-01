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

from absl.testing import absltest
import numpy as np

from framework import program_builder as pb
from framework.compiler import dim_utils
from framework.compiler import ffn_lookup_utils


class FfnExpansionUtilsTest(absltest.TestCase):

  def test_build_lookup_params(self):
    # The FFN is a simple function that simply sets the value of `foo` to 1.
    def ffn_fn(activations):
      activations["foo"] = 1

    program_spec = pb.program_spec(
        variables={
            "foo": pb.var(2),
            "bar": pb.var(2),
        },
        heads={},
        ffn_fn=ffn_fn,
        output_name="foo",
        input_range=2,
    )
    rules = program_spec.mlp.get_rules()
    print("len(rules): %s" % len(rules))
    for rule in rules:
      print(rule)
    dim_mappings = dim_utils.get_var_mapping(program_spec)
    expanded_dim_mappings = dim_utils.get_expanded_var_mapping(program_spec)

    lookup_params = ffn_lookup_utils.build_lookup_params(
        program_spec=program_spec,
        dim_mappings=dim_mappings,
        expanded_dim_mappings=expanded_dim_mappings,
    )
    print(lookup_params)

    # Initial variable values are:
    # `foo` is 0, so it is represented as [1.0, 0.0].
    # `bar` is 0, so it is represented as [1.0, 0.0].
    input_arr = np.array([1.0, 0.0, 1.0, 0.0])
    output = np.matmul(input_arr, lookup_params.weights_1)
    output += lookup_params.bias_1
    output = np.minimum(1, np.maximum(0, output))
    print("hidden")
    print(output)
    output = np.matmul(output, lookup_params.weights_2)
    output += lookup_params.bias_2
    print("output")
    print(output)
    # Residual connection.
    output += input_arr

    # After running `ffn_fn` we expect:
    # `foo` is now 1, so it is represented as [0.0, 1.0].
    # `bar` is still 0, so it is represented as [1.0, 0.0].
    expected_output = np.array([0.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

  def test_build_lookup_params_numeric(self):
    def ffn_fn(activations):
      activations["bar"] = activations["bar"] + 0.1
      activations["xyz"] = activations["bar"]

    program_spec = pb.program_spec(
        variables={
            "foo": pb.var(2),
            "bar": pb.numeric_var(values=(0.0, 0.1, 0.2), default=0.1),
            "xyz": pb.numeric_var(values=(0.0, 0.1, 0.2), default=0.1),
        },
        heads={},
        ffn_fn=ffn_fn,
        output_name="foo",
        input_range=2,
    )

    rules = program_spec.mlp.get_rules()
    print("len(rules): %s" % len(rules))
    for rule in rules:
      print(rule)
    dim_mappings = dim_utils.get_var_mapping(program_spec)
    expanded_dim_mappings = dim_utils.get_expanded_var_mapping(program_spec)

    lookup_params = ffn_lookup_utils.build_lookup_params(
        program_spec=program_spec,
        dim_mappings=dim_mappings,
        expanded_dim_mappings=expanded_dim_mappings,
    )
    print(lookup_params)

    # Initial variable values are:
    # `foo` is 0, so it is represented as [1.0, 0.0].
    # `bar` is 0.1, the discretized representation is [0.0, 1.0, 0.0].
    # `xyz` is 0.1, the discretized representation is [0.0, 1.0, 0.0].
    input_arr = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    output = np.matmul(input_arr, lookup_params.weights_1)
    output += lookup_params.bias_1
    output = np.minimum(1, np.maximum(0, output))
    print("hidden")
    print(output)
    output = np.matmul(output, lookup_params.weights_2)
    output += lookup_params.bias_2
    print("output")
    print(output)
    # Residual connection. This adds the input prior to being expanded.
    output += [1.0, 0.0, 0.1, 0.1]

    # After running `ffn_fn` we expect:
    # `foo` is unchanged.
    # `bar` is now 0.2.
    expected_output = np.array([1.0, 0.0, 0.2, 0.1])
    np.testing.assert_array_almost_equal(output, expected_output, decimal=2)


if __name__ == "__main__":
  absltest.main()
