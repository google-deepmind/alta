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
from framework.compiler import projection_utils


class ProjectionUtilsTest(absltest.TestCase):

  def test_projection_utils(self):
    program_spec = pb.program_spec(
        variables={
            "foo": pb.var(2),
            "bar": pb.var(2),
        },
        heads={},
        ffn_fn=lambda x: x,
        output_name="foo",
        input_range=2,
    )
    var_mappings = dim_utils.get_var_mapping(program_spec)

    select_transform = projection_utils.select_variable(var_mappings, "foo")
    output_transform = projection_utils.project_variable(var_mappings, "bar")

    # Without broadcasting.
    activations = np.array([[1.0, 0.0, 0.0, 1.0]])

    print(select_transform)
    select_vector = np.matmul(activations, select_transform)
    expected_select_vector = np.array([[1.0, 0.0]])
    np.testing.assert_equal(select_vector, expected_select_vector)

    output_vector = np.matmul(select_vector, output_transform)
    expected_output_vector = np.array([[0.0, 0.0, 1.0, 0.0]])
    np.testing.assert_equal(output_vector, expected_output_vector)

    # With implicit broadcasting.
    activations = np.array([1.0, 0.0, 0.0, 1.0])

    print(select_transform)
    select_vector = np.matmul(activations, select_transform)
    expected_select_vector = np.array([1.0, 0.0])
    np.testing.assert_equal(select_vector, expected_select_vector)

    output_vector = np.matmul(select_vector, output_transform)
    expected_output_vector = np.array([0.0, 0.0, 1.0, 0.0])
    np.testing.assert_equal(output_vector, expected_output_vector)


if __name__ == "__main__":
  absltest.main()
