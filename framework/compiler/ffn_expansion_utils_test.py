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

from alta.framework import program_builder as pb
from alta.framework.compiler import compiler_config
from alta.framework.compiler import dim_utils
from alta.framework.compiler import ffn_expansion_utils


def expand_activations(program_spec, config, activations):
  """Applies expansion layers to `activations`."""
  dim_mappings = dim_utils.get_var_mapping(program_spec)
  expanded_dim_mappings = dim_utils.get_expanded_var_mapping(program_spec)
  expansion_params = ffn_expansion_utils.build_expansion_params(
      program_spec=program_spec,
      dim_mappings=dim_mappings,
      expanded_dim_mappings=expanded_dim_mappings,
      config=config,
  )

  output = np.matmul(activations, expansion_params.weights_1)
  output += expansion_params.bias_1
  output = np.minimum(1, np.maximum(0, output))
  output = np.matmul(output, expansion_params.weights_2)
  output += expansion_params.bias_2
  return np.minimum(1, np.maximum(0, output))


class FfnExpansionUtilsTest(absltest.TestCase):

  def test_build_expansion_params_simple(self):
    program_spec = pb.program_spec(
        variables={
            "foo": pb.var(2),
            "bar": pb.var(2),
            "xyz": pb.numeric_var(values=tuple([0.0, 0.25, 0.5, 0.75, 1.0])),
        },
        heads={},
        ffn_fn=lambda x: x,
        output_name="foo",
        input_range=2,
    )
    config = compiler_config.Config()
    # Here we assume the following variable values:
    # `foo` is 0, so it is represented as [1.0, 0.0].
    # `bar` is 0, so it is also represented as [1.0, 0.0].
    # `xyz` is 0.6, so it is represented as [0.6].
    activations = np.array([[1.0, 0.0, 1.0, 0.0, 0.6]])

    output = expand_activations(program_spec, config, activations)
    # In the output, the categorical variables `foo` and `bar` have not changed,
    # so the first four dimensions should still be [1.0, 0.0, 1.0, 0.0].
    # However, the variable `xyz` has been expanded to a one-hot representation,
    # according to the buckets defined in its specification above. The value
    # 0.6 maps to the 3rd bucket (out of 5 bucket), so its representation
    # is now [0.0, 0.0, 1.0, 0.0, 0.0].
    expected_output = np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

  def test_build_expansion_params_with_set(self):
    set_range = 4
    possible_values = [frozenset(range(idx + 1)) for idx in range(4)]
    default = frozenset(range(3))
    program_spec = pb.program_spec(
        variables={
            "foo": pb.var(2),
            "bar": pb.set_var(
                range=set_range,
                values=tuple(possible_values),
                default=default,
            ),
        },
        heads={},
        ffn_fn=lambda x: x,
        output_name="foo",
        input_range=2,
    )
    config = compiler_config.Config()
    # Set `bar` to be `{0, 1, 2}`.
    activations = [[1.0, 0.0, 1.0, 1.0, 1.0, 0.0]]

    output = expand_activations(program_spec, config, activations)
    # There are four possible values, and `{0, 1, 2}` is the 3rd possible value.
    expected_output = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

  def test_build_expansion_params_higher_expansion_scalar(self):
    """Increasing `expansion_scalar_1` handles smaller tolerances."""
    program_spec = pb.program_spec(
        variables={
            "foo": pb.var(2),
            "bar": pb.var(2),
            "x": pb.numeric_var(
                default=0.33,
                values=tuple([x / 50 for x in range(50)]),
            ),
        },
        heads={},
        ffn_fn=lambda x: x,
        output_name="foo",
        input_range=2,
    )
    config = compiler_config.Config(expansion_scalar_1=1000)
    activations = np.array([[1.0, 0.0, 1.0, 0.0, 0.33]])

    output = expand_activations(program_spec, config, activations)

    # In the output, the categorical variables `foo` and `bar` have not changed,
    # so the first four dimensions should still be [1.0, 0.0, 1.0, 0.0].
    # However, the variable `x` has been expanded to a one-hot representation,
    # according to the buckets defined in its specification above. The value
    # 0.33 maps to the 16th bucket (out of 50 buckets).
    one_hot = np.zeros(50, dtype=int)
    one_hot[16] = 1
    expected_output = np.expand_dims(
        np.concatenate([np.array([1, 0, 1, 0]), one_hot]), axis=0
    )
    np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

  def test_build_expansion_params_threshold_gt_one(self):
    """Tests that the expansion layer works when buckets are greater than 1."""
    program_spec = pb.program_spec(
        variables={
            "foo": pb.var(2),
            "bar": pb.var(2),
            "x": pb.numeric_var(
                default=500,
                values=tuple(range(1, 1000, 40)),
            ),
        },
        heads={},
        ffn_fn=lambda x: x,
        output_name="foo",
        input_range=2,
    )
    config = compiler_config.Config(expansion_scalar_1=100_000)
    activations = np.array([[1.0, 0.0, 1.0, 0.0, 500]])

    output = expand_activations(program_spec, config, activations)

    # In the output, the categorical variables `foo` and `bar` have not changed,
    # so the first four dimensions should still be [1.0, 0.0, 1.0, 0.0].
    # However, the variable `x` has been expanded to a one-hot representation,
    # according to the buckets defined in its specification above. The value
    # 500 maps to the 12th bucket (out of 25 buckets).
    one_hot = np.zeros(25, dtype=int)
    one_hot[12] = 1
    expected_output = np.expand_dims(
        np.concatenate([np.array([1, 0, 1, 0]), one_hot]), axis=0
    )
    np.testing.assert_array_almost_equal(output, expected_output, decimal=3)


if __name__ == "__main__":
  absltest.main()
