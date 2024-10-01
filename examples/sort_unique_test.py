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

"""Tests interpreter for "sort unique" task."""

from absl.testing import absltest

from examples import sort_unique
from framework.interpreter import interpreter_utils
from framework.interpreter import logger_utils
from framework.interpreter import program_utils


class SortUniqueTest(absltest.TestCase):

  def test_sort_unique(self):
    program_spec = sort_unique.build_program_spec()

    input_ids = [sort_unique.BOS_VALUE, 3, 5, 4, 2]
    activations_seq = program_utils.initialize_activations(
        program_spec, input_ids
    )
    logger = logger_utils.ActivationsLogger()
    outputs = interpreter_utils.run_transformer(
        program_spec,
        activations_seq,
        logger=logger,
        max_layers=4,
    )
    logger.print_activations_table(variables_to_include=["target_pos"])

    self.assertSequenceAlmostEqual(
        outputs, [sort_unique.BOS_VALUE, 2, 3, 4, 5], places=1
    )


if __name__ == "__main__":
  absltest.main()
