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

"""Runs compiled transformer with learned FFN."""

from collections.abc import Sequence

from absl import app
from absl import flags
import tensorflow as tf

from alta.framework import program_registry
from alta.framework.common import io_utils
from alta.framework.common import tf_utils
from alta.framework.compiler import compiler_config
from alta.framework.compiler import compiler_utils
from alta.framework.traces.ffn import serialize
from alta.framework.transformer import transformer_utils

_INPUT_PATH = flags.DEFINE_string(
    "input_path", None, "Path to jsonl file of model inputs.", required=True
)

_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    "Model outputs will be written as TFRecords to this path.",
    required=True,
)

_PROGRAM = flags.DEFINE_string(
    "program",
    None,
    "Name of program to compile.",
    required=True,
)

_FFN_PARAMS_PATH = flags.DEFINE_string(
    "ffn_params_path",
    None,
    "Optional. If set, uses learned FFN at given path. Otherwise, uses compiled"
    " FFN.",
)

_ACTIVATION_FN = flags.DEFINE_enum(
    "activation_fn", "relu", ["sigmoid", "relu", "tanh"], "Activation function."
)

_MAX_LAYERS = flags.DEFINE_integer(
    "max_layers", None, "Number of inference layers.", required=True
)

_ATTENTION_SCALAR = flags.DEFINE_integer(
    "attention_scalar", 100, "Attention scalar."
)

_VERBOSE = flags.DEFINE_bool("verbose", False, "Whether to out debug logs.")


def compile_and_run_transformer(
    input_path: str,
    output_path: str,
    ffn_params_path: str | None,
    program: str,
    activation_fn_name: str,
    max_layers: int,
    attention_scalar: int,
    verbose: bool,
):
  """Compiles transformer for program and runs with learned FFN."""
  program_spec = program_registry.get_program(program)
  config = compiler_config.Config(
      expansion_scalar_1=1000, attention_scalar=attention_scalar
  )
  parameters = compiler_utils.compile_transformer(
      program_spec, config, compile_ffn=(not ffn_params_path)
  )

  learned_ffn_params = None
  if ffn_params_path:
    learned_ffn_params = serialize.load_params(ffn_params_path)

  inputs = io_utils.read_jsonl(input_path)
  outputs = [
      transformer_utils.run_transformer(
          parameters,
          learned_ffn_params,
          model_input,
          activation_fn_name=activation_fn_name,
          max_layers=max_layers,
          verbose=verbose,
      )
      for model_input in inputs
  ]

  examples = []
  for model_input, model_output in zip(inputs, outputs):
    example = tf.train.Example()
    tf_utils.add_int_list_feature(example, "model_input", model_input)
    tf_utils.add_int_list_feature(example, "model_output", model_output)
    examples.append(example)
  io_utils.write_tfrecords(examples, output_path)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  compile_and_run_transformer(
      input_path=_INPUT_PATH.value,
      output_path=_OUTPUT_PATH.value,
      ffn_params_path=_FFN_PARAMS_PATH.value,
      program=_PROGRAM.value,
      activation_fn_name=_ACTIVATION_FN.value,
      max_layers=_MAX_LAYERS.value,
      attention_scalar=_ATTENTION_SCALAR.value,
      verbose=_VERBOSE.value,
  )


if __name__ == "__main__":
  app.run(main)
