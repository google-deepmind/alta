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
from alta.framework.compiler import dim_utils
from alta.framework.compiler import embedding_utils


class EmbeddingUtilsTest(absltest.TestCase):

  def test_get_embedding_parameters(self):
    program_spec = pb.program_spec(
        variables={
            "inputs": pb.input_var(4),
            "indices": pb.position_var(4),
            "xyz": pb.numeric_var(values=range(10), default=5.0),
        },
        heads={},
        ffn_fn=lambda x: x,
        output_name="xyz",
        input_range=4,
        position_range=4,
    )
    var_mapping = dim_utils.get_var_mapping(program_spec)
    embeddings = embedding_utils.get_embedding_parameters(
        program_spec, var_mapping
    )

    expected_input_embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0],
    ])
    expected_index_embeddings = np.array([
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ])
    np.testing.assert_equal(
        embeddings.input_embeddings, expected_input_embeddings
    )
    np.testing.assert_equal(
        embeddings.index_embeddings, expected_index_embeddings
    )

    # Try a simple example use of the embedding tables.
    input_embeddings = np.take(
        embeddings.input_embeddings, np.array([1, 0]), axis=0
    )
    position_embeddings = np.take(
        embeddings.index_embeddings, np.array([0, 1]), axis=0
    )
    embeddings = input_embeddings + position_embeddings
    expected_embeddings = np.array([
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0],
    ])
    np.testing.assert_equal(embeddings, expected_embeddings)


if __name__ == "__main__":
  absltest.main()
