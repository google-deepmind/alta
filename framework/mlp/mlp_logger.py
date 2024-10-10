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

"""Class for logging satisfied transition rules."""

from framework.mlp import mlp_rules


class MLPLogger:
  """Logs seen rules."""

  seen: set[mlp_rules.Rule]

  def __init__(self):
    self.seen = set()

  def add(self, rule: mlp_rules.Rule):
    self.seen.add(rule)

  def reset(self):
    self.seen.clear()
