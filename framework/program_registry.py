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

"""Registry of programs."""

from examples import parity
from examples import parity_sparse
from examples.scan import scan_sparse_program
from framework import program


def get_program(
    program_name: str, dynamic_halting: bool = False
) -> program.Program:
  """Returns program specification."""
  name_to_spec = {
      "parity_sequential_absolute": parity.build_sequential_program_absolute(
          dynamic_halting=dynamic_halting
      ),
      "parity_sequential_relative": parity.build_sequential_program_relative(
          dynamic_halting=dynamic_halting
      ),
      "parity_sum_mod_2": (
          parity.build_intermediate_variable_sum_mod_2_program_spec()
      ),
      "sparse_parity_sequential_absolute": (
          parity_sparse.build_sequential_program_absolute(
              dynamic_halting=dynamic_halting
          )
      ),
      "sparse_parity_sequential_relative": (
          parity_sparse.build_sequential_program_relative(
              dynamic_halting=dynamic_halting
          )
      ),
      "sparse_parity_sum_mod_2": parity_sparse.build_sum_mod_2_program_spec(),
      "scan": scan_sparse_program.build_program_spec(),
  }
  return name_to_spec[program_name]
