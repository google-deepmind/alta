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

"""Trains simple MLP from traces."""

from collections.abc import Sequence
import functools
import json
import os
import time

from absl import app
from absl import flags
import jax
import numpy as np
import optax
import tensorflow as tf

from alta.framework.traces.ffn import activation
from alta.framework.traces.ffn import data
from alta.framework.traces.ffn import inference
from alta.framework.traces.ffn import initialization
from alta.framework.traces.ffn import metrics
from alta.framework.traces.ffn import serialize
from alta.framework.traces.ffn import train


_TRAIN_EXAMPLES_PATH = flags.DEFINE_string(
    "train_examples_path",
    None,
    "Path to TFRecords containing trace tf.Examples.",
    required=True,
)

_TEST_EXAMPLES_PATH = flags.DEFINE_string(
    "test_examples_path",
    None,
    "Path to TFRecords containing test tf.Examples.",
    required=True,
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Directory to write checkpoints and metrics to.",
    required=True,
)


_VECTOR_LENGTH = flags.DEFINE_integer(
    "vector_length",
    None,
    "Length of trace input and output vectors.",
    required=True,
)

_NUM_STEPS = flags.DEFINE_integer(
    "num_steps",
    1000000,
    "Number of training steps.",
)

_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate",
    1e-2,
    "Learning rate.",
)

_DECAY_FACTOR = flags.DEFINE_float(
    "decay_factor",
    1e-2,
    "Exponential decay factor.",
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    256,
    "Batch size.",
)

_NUM_HIDDEN_LAYERS = flags.DEFINE_integer(
    "num_hidden_layers",
    2,
    "Number of hidden layers.",
)

_HIDDEN_LAYER_SIZE = flags.DEFINE_integer(
    "hidden_layer_size",
    128,
    "Size of each hidden layer.",
)

_ACTIVATION_FN = flags.DEFINE_enum(
    "activation_fn", "relu", ["sigmoid", "relu", "tanh"], "Activation function."
)

_OPTIMIZATION_FN = flags.DEFINE_enum(
    "optimization_fn",
    "adafactor",
    ["adafactor", "adam"],
    "Optimization function.",
)

_INITIALIZATION_FN = flags.DEFINE_enum(
    "initialization_fn",
    "he_layer_params",
    ["he_layer_params", "random_layer_params", "xavier_normal_layer_params"],
    "Initialization function.",
)

_NOISE_STDDEV = flags.DEFINE_float(
    "noise_stddev",
    None,
    "Optional. If set adds noise with mean zero and this standard deviation to"
    " the input vectors.",
)


_EVAL_PERIOD = flags.DEFINE_integer(
    "eval_period",
    100,
    "Number of steps between evaluations.",
)

_CHECKPOINT_PERIOD = flags.DEFINE_integer(
    "checkpoint_period",
    None,
    "Number of steps between writing checkpoints. "
    "If None, only writes final checkpoint.",
)

_STATS_PERIOD = flags.DEFINE_integer(
    "stats_period",
    10,
    "Write statistics related to params and loss every N steps.",
)

_WARMUP_STEPS = flags.DEFINE_integer(
    "warmup_steps",
    100,
    "Number of warmup steps.",
)

_EVAL_SIZE = flags.DEFINE_integer(
    "eval_size",
    None,
    "Number of examples in evaluation set to evaluate on. If None, evaluate on"
    " all examples.",
)

_GRAD_ACCUMULATION_STEPS = flags.DEFINE_integer(
    "grad_accumulation_steps",
    1,
    "Number of steps to accumulate gradients over.",
)

_SUBTRACT_RESIDUAL = flags.DEFINE_bool(
    "subtract_residual",
    False,
    "Whether to set target to output - input (to accomodate the residual"
    " connection around the FFN in a Transformer). This must be done at"
    " training time if adding noise at training time, since the noise should"
    " also be subtracte from the output.",
)


def get_layer_sizes(
    vector_length: int, num_hidden_layers: int, hidden_layer_size: int
):
  """Returns layer sizes for FFN."""
  return (
      [vector_length]
      + [hidden_layer_size for _ in range(num_hidden_layers)]
      + [vector_length]
  )


def write_metrics(
    output_dir,
    steps,
    train_vector_element_accuracies,
    test_vector_element_accuracies,
):
  """Writes metrics as json to output directory."""
  metrics_dict = {
      "steps": steps,
      "train_vector_element_accuracies": train_vector_element_accuracies,
      "test_vector_element_accuracies": test_vector_element_accuracies,
  }
  with tf.io.gfile.GFile(
      os.path.join(output_dir, "metrics.json"),
      "w",
  ) as f:
    f.write(json.dumps(metrics_dict))


def get_summary_writer():
  return tf.summary.create_file_writer(os.path.join(_OUTPUT_DIR.value, "train"))


def write_metric(writer, name, metric, step):
  with writer.as_default():
    tf.summary.scalar(name, metric, step=step)


@functools.partial(
    jax.jit,
    # Pass immutable objects as static arguments.
    static_argnames="activation_fn",
)
def evaluate_model(params, activation_fn, inputs, targets):
  """Returns accuracy and element accuracy given inputs."""
  predictions = inference.batched_predict(params, inputs, activation_fn)
  vector_accuracy = metrics.get_vector_accuracy(predictions, targets)
  vector_element_accuracy = metrics.get_vector_element_accuracy(
      predictions, targets
  )
  return vector_accuracy, vector_element_accuracy


def write_params_stats(writer, params, step):
  """Writes statistics about parameters to summary writer."""
  with writer.as_default():
    projection_means = []
    projection_stdevs = []
    bias_means = []
    bias_stdevs = []
    for mat, bias in params:
      projection_means.append(np.mean(mat))
      projection_stdevs.append(np.std(mat))
      bias_means.append(np.mean(bias))
      bias_stdevs.append(np.std(bias))
  write_metric(writer, "projection_means", np.mean(projection_means), step)
  write_metric(writer, "projection_stdevs", np.mean(projection_stdevs), step)
  write_metric(writer, "bias_means", np.mean(bias_means), step)
  write_metric(writer, "bias_stdevs", np.mean(bias_stdevs), step)


def write_grad_stats(writer, grads, step):
  """Writes statistics about gradients to summary writer."""
  with writer.as_default():
    projection_grad_norms = []
    bias_grad_norms = []
    for mat_grad, bias_grad in grads:
      projection_grad_norms.append(np.mean(np.abs(mat_grad)))
      bias_grad_norms.append(np.mean(np.abs(bias_grad)))
  write_metric(
      writer, "projection_grad_norms", np.mean(projection_grad_norms), step
  )
  write_metric(writer, "bias_grad_norms", np.mean(bias_grad_norms), step)


def get_optimization_fn(optimization_fn_name: str):
  if optimization_fn_name == "adam":
    return optax.adam
  elif optimization_fn_name == "adafactor":
    return optax.adafactor


def get_optimizer(training_config: train.TrainingConfig):
  """Returns optimizer."""
  lr_fn = optax.warmup_exponential_decay_schedule(
      init_value=0.0,
      peak_value=training_config.learning_rate,
      warmup_steps=training_config.warmup_steps,
      transition_steps=training_config.num_steps,
      decay_rate=_DECAY_FACTOR.value,
      end_value=training_config.learning_rate * _DECAY_FACTOR.value,
  )
  optimizer = training_config.optimization_fn(lr_fn)
  if _GRAD_ACCUMULATION_STEPS.value > 1:
    optimizer = optax.MultiSteps(
        optimizer, every_k_schedule=_GRAD_ACCUMULATION_STEPS.value
    )
  return optimizer


def train_model(training_config: train.TrainingConfig):
  """Trains model using given config."""
  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.set_visible_devices([], "GPU")

  test_inputs, test_outputs = data.get_all_data(
      training_config.test_examples_path,
      training_config.vector_length,
      sample_size=training_config.eval_size,
  )
  if _SUBTRACT_RESIDUAL.value:
    test_outputs -= test_inputs
  params = initialization.init_network_params(
      training_config.layer_sizes,
      jax.random.key(0),
      training_config.initialization_fn,
  )
  optimizer = get_optimizer(training_config)
  opt_state = optimizer.init(params)
  key = jax.random.PRNGKey(0)
  writer = get_summary_writer()

  step = 0
  epoch = 0
  while True:
    if step >= training_config.num_steps:
      break

    start_time = time.time()

    for batch in data.get_batches(
        training_config.train_examples_path,
        training_config.vector_length,
        training_config.batch_size,
    ):
      vector_input = batch["input"]
      vector_output = batch["output"]

      if training_config.noise_stddev is not None:
        key, subkey = jax.random.split(key)
        noise = (
            jax.random.normal(
                subkey,
                (training_config.batch_size, vector_input.shape[1]),
            )
            * training_config.noise_stddev
        )
        vector_input += noise
      # Must wait to subtract `vector_input` from `vector_output` until after
      # adding noise.
      if _SUBTRACT_RESIDUAL.value:
        vector_output -= vector_input
      # Write checkpoint every `checkpoint_period` steps.
      if (training_config.checkpoint_period is not None and
          step % training_config.checkpoint_period == 0):
        serialize.save_params(os.path.join(training_config.output_dir,
                                           f"params-{step}.pkl"), params)

      # Run evaluation every `eval_period` steps.
      if step % training_config.eval_period == 0:
        train_vector_accuracy, train_vector_element_accuracy = evaluate_model(
            params, training_config.activation_fn, vector_input, vector_output
        )
        test_vector_accuracy, test_vector_element_accuracy = evaluate_model(
            params, training_config.activation_fn, test_inputs, test_outputs
        )

        print(
            "Step {} Train vector accuracy {}".format(
                step, train_vector_accuracy.item()
            )
        )
        print(
            "Step {} Train vector element accuracy {}".format(
                step, train_vector_element_accuracy.item()
            )
        )
        print(
            "Step {} Test vector accuracy {}".format(
                step, test_vector_accuracy.item()
            )
        )
        print(
            "Step {} Test vector element accuracy {}".format(
                step, test_vector_element_accuracy.item()
            )
        )

        write_metric(
            writer, "train_vector_accuracy", train_vector_accuracy.item(), step
        )
        write_metric(
            writer,
            "train_vector_element_accuracy",
            train_vector_element_accuracy.item(),
            step,
        )
        write_metric(
            writer, "test_vector_accuracy", test_vector_accuracy.item(), step
        )
        write_metric(
            writer,
            "test_vector_element_accuracy",
            test_vector_element_accuracy.item(),
            step,
        )

      loss, grads, params, opt_state = train.update(
          params,
          training_config.activation_fn,
          optimizer,
          opt_state,
          vector_input,
          vector_output,
      )

      if step % training_config.stats_period == 0:
        write_metric(writer, "loss", loss, step)
        write_params_stats(writer, params, step)
        write_grad_stats(writer, grads, step)

      step += 1
    epoch_time = time.time() - start_time
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    epoch += 1
    write_metric(writer, "epoch", epoch, step)

  return params


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if tf.io.gfile.exists(_OUTPUT_DIR.value):
    print("Warning: --output_dir {} already exists.".format(_OUTPUT_DIR.value))
  else:
    tf.io.gfile.makedirs(_OUTPUT_DIR.value)

  training_config = train.TrainingConfig(
      layer_sizes=get_layer_sizes(
          vector_length=_VECTOR_LENGTH.value,
          num_hidden_layers=_NUM_HIDDEN_LAYERS.value,
          hidden_layer_size=_HIDDEN_LAYER_SIZE.value,
      ),
      learning_rate=_LEARNING_RATE.value,
      num_steps=_NUM_STEPS.value,
      batch_size=_BATCH_SIZE.value,
      train_examples_path=_TRAIN_EXAMPLES_PATH.value,
      test_examples_path=_TEST_EXAMPLES_PATH.value,
      vector_length=_VECTOR_LENGTH.value,
      activation_fn=activation.get_activation_fn(_ACTIVATION_FN.value),
      initialization_fn=initialization.get_initialization_fn(
          _INITIALIZATION_FN.value
      ),
      optimization_fn=get_optimization_fn(_OPTIMIZATION_FN.value),
      noise_stddev=_NOISE_STDDEV.value,
      eval_period=_EVAL_PERIOD.value,
      eval_size=_EVAL_SIZE.value,
      checkpoint_period=_CHECKPOINT_PERIOD.value,
      output_dir=_OUTPUT_DIR.value,
      stats_period=_STATS_PERIOD.value,
      warmup_steps=_WARMUP_STEPS.value,
  )
  params = train_model(training_config)

  # Write final checkpoint.
  serialize.save_params(os.path.join(_OUTPUT_DIR.value, "params.pkl"), params)


if __name__ == "__main__":
  app.run(main)
