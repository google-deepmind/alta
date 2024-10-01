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

"""Implements minimalist numpy Transformer forward pass."""

from typing import List, Union

import jax.numpy as jnp
import numpy as np
import scipy

from alta.framework.traces.ffn import activation
from alta.framework.traces.ffn import inference
from alta.framework.transformer import parameters


def initialize_embeddings(params, input_ids):
  """Initializes embeddings for input tokens."""
  input_embeddings = np.take(
      params.embeddings.input_embeddings, np.array(input_ids), axis=0
  )

  if params.embeddings.index_embeddings is None:
    return input_embeddings

  indices = range(len(input_ids))
  position_embeddings = np.take(
      params.embeddings.index_embeddings, np.array(indices), axis=0
  )
  return input_embeddings + position_embeddings


def get_output(params, embeddings):
  """Runs the output transformation."""
  output = np.matmul(embeddings, params.output_transform)
  output = np.argmax(output, axis=1)
  return output.tolist()


def get_relative_position_embeddings(
    relative_position_mask: frozenset[int], num_inputs: int
) -> np.ndarray:
  """Returns relative position embeddings.

  Returns T5 style relative position embeddings based on mask. I.e. embeddings
  are 0 if unmasked and -1e9 if masked.

  E.g. given `relative_position_mask` of {-1} and `num_inputs` of 3, this
  function will return:
  [[-1e9, -1e9, -1e9],
   [0, -1e9, -1e9],
   [-1e9, 0, -1e9]]

  Args:
    relative_position_mask: Set of relative positions that should be masked. If
      unset, no positions are masked.
    num_inputs: Number of input tokens.

  Returns:
    Relative position embedding matrix.
  """
  relative_position_embeddings = np.zeros((num_inputs, num_inputs))
  if not relative_position_mask:
    return relative_position_embeddings

  for i in range(num_inputs):
    for j in range(num_inputs):
      if j - i not in relative_position_mask:
        # Use -1e9 and not -inf to avoid softmax returning nan if all positions
        # are masked. Unnormalized score will always be 0 or
        # `attention_scalar`** 2 because it's the dot product of queries and
        # keys, both of which have values of either 0 or `attention_scalar` and
        # key is a one-hot vector. So guaranteed to mask positions as long as
        # `attention_scalar`** 2 is significantly less than 1e-9.
        relative_position_embeddings[i, j] = -1e9

  return relative_position_embeddings


def multihead_attention(params, embeddings):
  """Runs multihead attention."""
  output = np.zeros(embeddings.shape)
  for attention_head_params in params.attenion_heads:
    queries = np.matmul(embeddings, attention_head_params.query_transform)
    keys = np.matmul(embeddings, attention_head_params.key_transform)
    attn_weights = np.matmul(queries, np.transpose(keys))
    relative_position_embeddings = get_relative_position_embeddings(
        attention_head_params.relative_position_mask, embeddings.shape[0]
    )
    attn_weights += relative_position_embeddings
    attn_weights = scipy.special.softmax(attn_weights, axis=-1)
    values = np.matmul(embeddings, attention_head_params.value_transform)
    aggr_values = np.matmul(attn_weights, values)
    # We use the reparameterization of the output matrix used by Tracr:
    # https://arxiv.org/abs/2301.05062
    output += np.matmul(aggr_values, attention_head_params.output_transform)
  return output


def clipped_relu(x):
  """Clipped ReLU activation function."""
  return np.minimum(1, np.maximum(0, x))


def run_ffn(params, embeddings, verbose=False):
  """Runs MLP sub-layer."""
  final_layer_idx = len(params.feed_forward_layers) - 1
  activations = embeddings
  # TODO(jamesfcohan): Consider raising error if a numeric variable is not
  # expanded to a one-hot. Solutions for this error are increasing
  # `expansion_scalar_1` or, if the scalar variable is on a bucket boundary,
  # changing the bucket boundaries.
  for layer_idx, layer_params in enumerate(params.feed_forward_layers):
    activations = np.matmul(activations, layer_params.weights)
    activations += layer_params.biases
    if layer_idx != final_layer_idx:
      activations = clipped_relu(activations)
    if verbose and layer_idx == 1:
      print("FFN layer 1: %s" % activations)
    if verbose and layer_idx == 2:
      for element_idx, element in enumerate(activations):
        print("element %s FFN layer 2" % element_idx)
        for idx, value in enumerate(element):
          if value != 0:
            print("FFN layer 2@%s: %s" % (idx, value))

  return activations


def run_layer(
    params, learned_ffn_params, embeddings, activation_fn_name, verbose=False
):
  """Runs one Transformer layer."""
  attn_output = multihead_attention(params, embeddings)
  if verbose:
    print("attn_output: %s" % attn_output)
  # Residual connection.
  attn_output += embeddings
  if verbose:
    print("attn_output + residual: %s" % attn_output)

  if learned_ffn_params:
    # TODO(b/347699354): Make activation fn configurable.
    ffn_output = inference.batched_predict(
        learned_ffn_params,
        attn_output,
        activation.get_activation_fn(activation_fn_name),
    )
  else:
    ffn_output = run_ffn(params, attn_output, verbose=verbose)
  if verbose:
    print("ffn_output: %s" % ffn_output)

  # Residual connection.
  ffn_output += attn_output
  if verbose:
    print("ffn_output + residual: %s" % ffn_output)
  return ffn_output


def run_transformer(
    params: parameters.Parameters,
    learned_ffn_params: jnp.ndarray | None,
    input_ids: List[int],
    max_layers: int = 100,
    activation_fn_name: str = "sigmoid",
    verbose: bool = False,
) -> List[Union[int, float]]:
  """Runs a Transformer forward pass.

  Args:
    params: Compiled transformer parameters.
    learned_ffn_params: Optional. Parameters of a learned FFN. If None, uses
      compiled FFN params.
    input_ids: List of input ids.
    max_layers: Maximum number of layers to run.
    activation_fn_name: Name of activation function to use. Must be sigmoid if
      using compiled FFN params.
    verbose: Whether to print debugging information.

  Returns:
    List of outputs.
  """
  if not learned_ffn_params and activation_fn_name != "sigmoid":
    raise ValueError(
        "If using compiled FFN, activation function must be sigmoid."
    )
  embeddings = initialize_embeddings(params, input_ids)
  if verbose:
    np.set_printoptions(precision=2, floatmode="fixed", suppress=True)
    print("embeddings: %s" % embeddings)
  for layer_idx in range(max_layers):
    if verbose:
      print("layer_idx: %s" % layer_idx)
    embeddings = run_layer(
        params,
        learned_ffn_params,
        embeddings,
        activation_fn_name,
        verbose=verbose,
    )
    # TODO(petershaw): Implement dynamic halting mechanism.

  return get_output(params, embeddings)
