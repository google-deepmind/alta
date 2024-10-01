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

"""Defines set of parameters for Transformer with weight-sharing."""

import dataclasses
import numpy.typing as npt


@dataclasses.dataclass
class AttentionHeadParameters:
  query_transform: npt.ArrayLike
  key_transform: npt.ArrayLike
  value_transform: npt.ArrayLike
  output_transform: npt.ArrayLike
  relative_position_mask: frozenset[int]


@dataclasses.dataclass
class FeedForwardLayerParams:
  weights: npt.ArrayLike
  biases: npt.ArrayLike


@dataclasses.dataclass
class EmbeddingParameters:
  input_embeddings: npt.ArrayLike
  index_embeddings: npt.ArrayLike | None


@dataclasses.dataclass
class Parameters:
  attenion_heads: list[AttentionHeadParameters]
  feed_forward_layers: list[FeedForwardLayerParams] | None
  embeddings: EmbeddingParameters
  output_transform: npt.ArrayLike
