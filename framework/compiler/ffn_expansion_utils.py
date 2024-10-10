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

"""Utilities for FFN layers that expand numeric variables.

We generate 4-layer FNNs. The first 2 layers "expand" numeric variables
to a discretized one-hot representation. The utilities in this module generate
the parameters for these first 2 layers.
"""

import dataclasses

import numpy as np

from framework import program
from framework import var_utils
from framework.compiler import compiler_config
from framework.compiler import dim_utils


@dataclasses.dataclass(frozen=True)
class ExpansionParams:
  """Parameters for expanding variables to one-hot vectors."""

  weights_1: np.ndarray
  bias_1: np.ndarray
  weights_2: np.ndarray
  bias_2: np.ndarray


@dataclasses.dataclass(frozen=True)
class VarExpansionParams:
  """Parameters for expanding a single variable to one-hot vectors."""

  weights_1: np.ndarray
  bias_1: np.ndarray
  weights_2: np.ndarray
  bias_2: np.ndarray
  input_start_idx: int
  input_end_idx: int
  output_start_idx: int
  output_end_idx: int


def _build_categorical_expansion_params(
    input_mapping: dim_utils.CategoricalVarDimMapping,
    output_mapping: dim_utils.CategoricalVarDimMapping,
) -> VarExpansionParams:
  """Categorical representations are unchanged through these layers."""
  dims = input_mapping.end_idx - input_mapping.start_idx
  identity = np.identity(dims)
  bias = np.zeros(dims)
  return VarExpansionParams(
      weights_1=identity,
      bias_1=bias,
      weights_2=identity,
      bias_2=bias,
      input_start_idx=input_mapping.start_idx,
      input_end_idx=input_mapping.end_idx,
      output_start_idx=output_mapping.start_idx,
      output_end_idx=output_mapping.end_idx,
  )


def _build_set_expansion_params(
    input_mapping: dim_utils.SetVarDimMapping,
    output_mapping: dim_utils.ExpandedSetVarDimMapping,
) -> VarExpansionParams:
  """Builds a lookup table mapping from set values to one-hot vectors."""
  input_dims = input_mapping.end_idx - input_mapping.start_idx
  output_dims = output_mapping.end_idx - output_mapping.start_idx

  weights_1_stack = []
  bias_1_stack = []

  for possible_value in output_mapping.values:
    vector = np.zeros(input_dims)
    for idx in range(input_dims):
      if idx in possible_value:
        vector[idx] = 1.0
      else:
        vector[idx] = -1.0
    weights_1_stack.append(vector)
    # Ensure that if any value is missing from the set, the input vector
    # will be all zeros.
    bias_1_stack.append(1 - len(possible_value))

  weights_1 = np.stack(weights_1_stack, axis=0)
  weights_1 = np.transpose(weights_1)
  bias_1 = np.stack(bias_1_stack, axis=0)
  weights_2 = np.identity(output_dims)
  bias_2 = np.zeros(output_dims)

  return VarExpansionParams(
      weights_1,
      bias_1,
      weights_2,
      bias_2,
      input_start_idx=input_mapping.start_idx,
      input_end_idx=input_mapping.end_idx,
      output_start_idx=output_mapping.start_idx,
      output_end_idx=output_mapping.end_idx,
  )


def _get_weight_and_bias(
    expansion_scalar_1: float, threshold: float
) -> tuple[float, float]:
  """Returns the weight and bias to implement a "step" at `threshold`.

  Approximates a step function at `threshold` with the parameterization `y =
  a(w * x + b)`, where `a` is a clipped ReLU activation function.
  The "step" takes place where `x = - b / w`.

  The approach is inspired by:
  http://neuralnetworksanddeeplearning.com/chap4.html

  Args:
    expansion_scalar_1: How large to make `weight` or `bias`. For larger values
      of `expansion_scalar_1` the approximation will be less smooth and closer
      to a true "step" function.
    threshold: The threshold at which to make the step.

  Returns:
    A weight and bias that approximate a step function at `threshold`.
  """
  # For numerical stability, we ensure that
  # `max(bias, weight) <= expansion_scalar_1`.
  if threshold is None:
    # threshold = -inf
    weight = 0.0
    bias = expansion_scalar_1
    return (weight, bias)
  elif abs(threshold) > 1.0:
    # bias > weight.
    # weight = - bias / threshold.
    bias = -expansion_scalar_1
    weight = expansion_scalar_1 / threshold
    return (weight, bias)
  elif threshold == 0.0:
    # - bias / weight -> 0.0
    bias = 0.0
    weight = expansion_scalar_1
    return (weight, bias)
  else:
    # 0 < threshold < 1.
    # bias < weight.
    # bias = - weight * threshold.
    weight = expansion_scalar_1
    bias = -expansion_scalar_1 * threshold
    return (weight, bias)


def _build_numeric_layer_1_params(
    buckets: tuple[var_utils.Bucket, ...],
    expansion_scalar: float,
):
  """Returns params for implementing step function at each bucket threshold.

  Returns a list of weights and a list of biases such that weights[i] and
  biases[i] implement a step function at buckets[i].min_value. When the weights
  and biases are multiplied by a scalar, the output will be of the form
  [1,1,...,1,0,0,...,0], since the scalar will be higher than the first j step
  function thresholds and smaller than the final n-j step function thresholds.

  Args:
    buckets: The list of buckets whose min values are used as thresholds.
    expansion_scalar: How large to make `weight` or `bias`. For larger values of
      `expansion_scalar` the approximation will be less smooth and  closer to a
      true "step" function.

  Returns:
    List of weights and list of biases implementing step functions at each
    bucket threshold.
  """
  output_dims = len(buckets)

  weights = np.zeros([1, output_dims])
  biases = np.zeros([output_dims])

  for i, bucket in enumerate(buckets):
    threshold = None if bucket.first else bucket.min_value
    weight, bias = _get_weight_and_bias(expansion_scalar, threshold)
    weights[0, i] = weight
    biases[i] = bias

  return weights, biases


def _build_numeric_layer_2_params(
    dims: int,
    expansion_scalar: float,
):
  """Build params for mapping [1,1,...,1,0,0,...,0] to one-hot with 1 at last 1.

  Returns parameters for mapping a vector of the form [1,1,...1,0,0,...0] to a
  one-hot vector with a 1 where the input vector has its last 1.

  Args:
    dims: Number of dimensions in the input / output vectors.
    expansion_scalar: How large to make weights.

  Returns:
    List of weights and list of biases for converting to a one-hot.
  """
  weights = np.zeros([dims, dims])
  biases = np.zeros([dims])

  # `weights` will have 1's on the diagonal and -1's below the diagonal.
  # Multiplying the input by `weights` takes the dot product of the
  # input with each column of `weights`. Since the input vector is of the form
  # [1,1,...1,0,0,...0], where the last 1 is at index j, the dot product will
  # always be zero except when multiplied by column j.
  i = 0
  j = 0
  for _ in range(dims - 1):
    weights[j, i] = expansion_scalar
    weights[j + 1, i] = -expansion_scalar
    i += 1
    j += 1
  # Special case for values that exceed final threshold.
  weights[j, i] = expansion_scalar
  biases -= 0.5 * expansion_scalar

  return weights, biases


def _build_numeric_expansion_params(
    input_mapping: dim_utils.NumericalVarDimMapping,
    output_mapping: dim_utils.ExpandedNumericalVarDimMapping,
    expansion_scalar_1: float,
    expansion_scalar_2: float,
) -> VarExpansionParams:
  """Returns parameters for mapping from a scalar to one-hot vector."""
  buckets = output_mapping.buckets
  weights_1, bias_1 = _build_numeric_layer_1_params(buckets, expansion_scalar_1)
  weights_2, bias_2 = _build_numeric_layer_2_params(
      len(buckets), expansion_scalar_2
  )

  return VarExpansionParams(
      weights_1,
      bias_1,
      weights_2,
      bias_2,
      input_start_idx=input_mapping.idx,
      input_end_idx=input_mapping.idx + 1,
      output_start_idx=output_mapping.start_idx,
      output_end_idx=output_mapping.end_idx,
  )


def _get_expansion_params(
    input_mapping: dim_utils.VarDimMapping,
    output_mapping: dim_utils.VarDimMapping,
    config: compiler_config.Config,
) -> VarExpansionParams:
  """Returns parameters for mapping from a scalar to one-hot vector."""
  if isinstance(input_mapping, dim_utils.CategoricalVarDimMapping):
    assert isinstance(output_mapping, dim_utils.CategoricalVarDimMapping)
    return _build_categorical_expansion_params(input_mapping, output_mapping)
  elif isinstance(input_mapping, dim_utils.NumericalVarDimMapping):
    assert isinstance(output_mapping, dim_utils.ExpandedNumericalVarDimMapping)
    return _build_numeric_expansion_params(
        input_mapping,
        output_mapping,
        config.expansion_scalar_1,
        config.expansion_scalar_2,
    )
  elif isinstance(input_mapping, dim_utils.SetVarDimMapping):
    assert isinstance(output_mapping, dim_utils.ExpandedSetVarDimMapping)
    return _build_set_expansion_params(input_mapping, output_mapping)
  else:
    raise ValueError(f"Unsupported input mapping: {input_mapping}")


def build_expansion_params(
    program_spec: program.Program,
    dim_mappings: dim_utils.VarDimMappings,
    expanded_dim_mappings: dim_utils.VarDimMappings,
    config: compiler_config.Config,
) -> ExpansionParams:
  """Builds the parameters for first two FFN layers."""

  # Initialize empty matrices.
  input_dims = dim_mappings.end_idx
  output_dims = expanded_dim_mappings.end_idx
  weights_1 = np.zeros([input_dims, output_dims])
  bias_1 = np.zeros([output_dims])
  weights_2 = np.zeros([output_dims, output_dims])
  bias_2 = np.zeros([output_dims])

  for var_name in program_spec.variables.keys():
    input_mapping = dim_mappings.var_mappings[var_name]
    output_mapping = expanded_dim_mappings.var_mappings[var_name]
    params = _get_expansion_params(input_mapping, output_mapping, config)
    weights_1[
        params.input_start_idx : params.input_end_idx,
        params.output_start_idx : params.output_end_idx,
    ] = params.weights_1
    bias_1[params.output_start_idx : params.output_end_idx] = params.bias_1
    weights_2[
        params.output_start_idx : params.output_end_idx,
        params.output_start_idx : params.output_end_idx,
    ] = params.weights_2
    bias_2[params.output_start_idx : params.output_end_idx] = params.bias_2

  return ExpansionParams(weights_1, bias_1, weights_2, bias_2)
