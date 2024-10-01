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

"""Defines configuration for compiler."""

import dataclasses


@dataclasses.dataclass
class Config:
  expansion_scalar_1: float = 100.0
  expansion_scalar_2: float = 100.0
  # attention_scalar ** 2 must be significantly smaller than 1e9 for relative
  # position masking to work.
  attention_scalar: float = 100.0
