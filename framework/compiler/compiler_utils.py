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

"""Implements a conversion from program spec to parameter values."""

from alta.framework import program
from alta.framework.compiler import compiler_config
from alta.framework.compiler import dim_utils
from alta.framework.compiler import embedding_utils
from alta.framework.compiler import ffn_expansion_utils
from alta.framework.compiler import ffn_lookup_utils
from alta.framework.compiler import projection_utils
from alta.framework.transformer import parameters


def _get_feed_forward_params(
    program_spec, config, dim_mappings, expanded_dim_mappings
):
  """Get parameters for feed forward layers."""
  expansion_params = ffn_expansion_utils.build_expansion_params(
      program_spec,
      dim_mappings,
      expanded_dim_mappings,
      config,
  )
  lookup_params = ffn_lookup_utils.build_lookup_params(
      program_spec,
      dim_mappings,
      expanded_dim_mappings,
  )

  ffn_params = [
      parameters.FeedForwardLayerParams(
          weights=expansion_params.weights_1, biases=expansion_params.bias_1
      ),
      parameters.FeedForwardLayerParams(
          weights=expansion_params.weights_2, biases=expansion_params.bias_2
      ),
      parameters.FeedForwardLayerParams(
          weights=lookup_params.weights_1, biases=lookup_params.bias_1
      ),
      parameters.FeedForwardLayerParams(
          weights=lookup_params.weights_2, biases=lookup_params.bias_2
      ),
  ]

  return ffn_params


def compile_transformer(
    program_spec: program.Program,
    config: compiler_config.Config,
    compile_ffn: bool = True,
    verbose=False,
) -> parameters.Parameters:
  """Generate parameters given a program specification."""
  # Generate variable to dimension mapping.
  dim_mappings = dim_utils.get_var_mapping(program_spec)
  expanded_dim_mappings = dim_utils.get_expanded_var_mapping(program_spec)
  if verbose:
    print("dim_mappings: %s" % dim_mappings)
    print("expanded_dim_mappings: %s" % expanded_dim_mappings)

  embeddings = embedding_utils.get_embedding_parameters(
      program_spec, dim_mappings
  )
  attention_heads = projection_utils.get_attention_params(
      program_spec, dim_mappings, config
  )

  feed_forward_layers = None
  if compile_ffn:
    feed_forward_layers = _get_feed_forward_params(
        program_spec, config, dim_mappings, expanded_dim_mappings
    )

  output_transform = projection_utils.get_output_transform(
      program_spec, dim_mappings
  )

  return parameters.Parameters(
      attenion_heads=attention_heads,
      feed_forward_layers=feed_forward_layers,
      embeddings=embeddings,
      output_transform=output_transform,
  )
