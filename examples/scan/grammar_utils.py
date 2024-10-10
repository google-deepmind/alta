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

"""Defines SCAN grammar and related utilities."""

import dataclasses
import functools


@dataclasses.dataclass(frozen=True)
class QCFGRule:
  """A Quasi-synchronous Context-Free Grammar (QCFG) rule."""

  # RHS list of nonterminal or terminal symbols for source and target.
  source: tuple[str, ...]
  target: tuple[str, ...]
  # LHS nonterminal symbol.
  lhs: str = "S"
  # For each target nonterminal, a mapping to corresponding source nonterminal.
  mapping: tuple[int, ...] = tuple()

  def key(self) -> str:
    return " ".join(self.source)


NONTERMINALS = ("S",)


SOURCE_TERMINALS = (
    "and",
    "after",
    "opposite",
    "turn",
    "around",
    "left",
    "right",
    "twice",
    "thrice",
    "walk",
    "look",
    "run",
    "jump",
)


TARGET_TERMINALS = (
    "WALK",
    "LOOK",
    "RUN",
    "JUMP",
    "LTURN",
    "RTURN",
)

RULES = (
    QCFGRule(
        source=("S", "and", "S"),
        target=("S", "S"),
        mapping=(0, 1),
    ),
    QCFGRule(
        source=("S", "after", "S"),
        target=("S", "S"),
        mapping=(1, 0),
    ),
    QCFGRule(
        source=("S", "twice"),
        target=("S", "S"),
        mapping=(0, 0),
    ),
    QCFGRule(
        source=("S", "thrice"),
        target=("S", "S", "S"),
        mapping=(0, 0, 0),
    ),
    QCFGRule(
        source=("walk",),
        target=("WALK",),
    ),
    QCFGRule(
        source=("look",),
        target=("LOOK",),
    ),
    QCFGRule(
        source=("run",),
        target=("RUN",),
    ),
    QCFGRule(
        source=("jump",),
        target=("JUMP",),
    ),
    QCFGRule(
        source=("turn", "left"),
        target=("LTURN",),
    ),
    QCFGRule(
        source=("turn", "right"),
        target=("RTURN",),
    ),
    QCFGRule(
        source=("turn", "opposite", "left"),
        target=("LTURN", "LTURN"),
    ),
    QCFGRule(
        source=("turn", "opposite", "right"),
        target=("RTURN", "RTURN"),
    ),
    QCFGRule(
        source=("S", "opposite", "left"),
        target=("LTURN", "LTURN", "S"),
        mapping=(0,),
    ),
    QCFGRule(
        source=("S", "opposite", "right"),
        target=("RTURN", "RTURN", "S"),
        mapping=(0,),
    ),
    QCFGRule(
        source=("S", "around", "left"),
        target=("LTURN", "S", "LTURN", "S", "LTURN", "S", "LTURN", "S"),
        mapping=(0, 0, 0, 0),
    ),
    QCFGRule(
        source=("S", "around", "right"),
        target=("RTURN", "S", "RTURN", "S", "RTURN", "S", "RTURN", "S"),
        mapping=(0, 0, 0, 0),
    ),
    QCFGRule(
        source=("turn", "around", "left"),
        target=("LTURN", "LTURN", "LTURN", "LTURN"),
    ),
    QCFGRule(
        source=("turn", "around", "right"),
        target=("RTURN", "RTURN", "RTURN", "RTURN"),
    ),
    QCFGRule(
        source=("S", "left"),
        target=("LTURN", "S"),
        mapping=(0,),
    ),
    QCFGRule(
        source=("S", "right"),
        target=("RTURN", "S"),
        mapping=(0,),
    ),
)


class Vocab:
  """Defines a mapping between tokens and integers."""

  def __init__(self, tokens):
    self.token_to_idx = {}
    self.idx_to_token = {}
    for idx, token in enumerate(tokens):
      self.token_to_idx[token] = idx
      self.idx_to_token[idx] = token

  def encode_tokens(self, tokens):
    return [self.token_to_idx[token] for token in tokens]

  def decode_tokens(self, token_ids):
    return [self.idx_to_token[idx] for idx in token_ids]


@functools.cache
def get_symbol_vocab():
  symbols = ("PAD",) + SOURCE_TERMINALS + TARGET_TERMINALS + NONTERMINALS
  return Vocab(symbols)


def get_symbol_id(symbol_token: str) -> int | None:
  if symbol_token is None:
    return None
  symbol_vocab = get_symbol_vocab()
  return symbol_vocab.token_to_idx.get(symbol_token)


def get_symbol_token(symbol_id: int) -> str | None:
  if symbol_id is None:
    return None
  symbol_vocab = get_symbol_vocab()
  return symbol_vocab.idx_to_token.get(symbol_id)
