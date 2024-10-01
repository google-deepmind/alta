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

"""Implements a program for parsing and decoding SCAN inputs."""

from alta.examples.scan import grammar_utils
from alta.examples.scan import scan_utils
from alta.framework import program_builder as pb


# Processing steps.
STEP_INIT_PARSE = 0
# Set `matched_rule_id`, `child_0_stack_position`, `child_1_stack_position`.
STEP_PARSE_1 = 1
# Set `child_0_position`, `child_1_position`.
STEP_PARSE_2 = 2
# Set `is_child_0`, `is_child_1`, `is_next_nt`, `is_tree_pointer`.
STEP_PARSE_3 = 3
# Execute `parse_action`.
STEP_PARSE_4 = 4
STEP_INIT_DECODE = 5
# Set `decode_action`, `decode_symbol` and `current_node_is_root`.
STEP_DECODE_1 = 6
# Determine if decoding is complete before executing action.
STEP_DECODE_2 = 7
# Execute decoding action.
STEP_DECODE_3 = 8
STEP_DONE = 9
NUM_STEPS = 10

# Decode actions.
DECODE_ACTION_NONE = 0
DECODE_ACTION_COPY = 1
DECODE_ACTION_CHILD_0 = 2
DECODE_ACTION_CHILD_1 = 3
DECODE_ACTION_PARENT = 4
NUM_DECODE_ACTIONS = 5

# Positions in stack.
STACK_POSITION_1 = 0
STACK_POSITION_2 = 1
STACK_POSITION_3 = 2
STACK_POSITION_NONE = 3
NUM_STACK_POSITIONS = 4


def eq(m: pb.MLPBuilder, var_name: str, var_value: int):
  """Helper method that iterates once at a specific value."""
  for tmp_value in m.get(var_name):
    if var_value == tmp_value:
      yield


def set_from_zero(m: pb.MLPBuilder, var_name: str, new_value: int):
  """Helper method that assumes current variable value is 0."""
  for _ in eq(m, var_name, 0):
    m.set(var_name, new_value)


def update_stack_pointers(m: pb.MLPBuilder, difference: int):
  for stack_pointer in m.get("stack_pointer"):
    m.set("stack_pointer", stack_pointer + difference)
  for stack_pointer_1 in m.get("stack_pointer_1"):
    m.set("stack_pointer_1", stack_pointer_1 + difference)
  for stack_pointer_2 in m.get("stack_pointer_2"):
    m.set("stack_pointer_2", stack_pointer_2 + difference)
  for stack_pointer_3 in m.get("stack_pointer_3"):
    m.set("stack_pointer_3", stack_pointer_3 + difference)


def step_init_parse(m: pb.MLPBuilder):
  """Initialize parsing state."""
  # Initialize pointers.
  for start_pointer in m.get("start_pointer"):
    stack_pointer = start_pointer + scan_utils.STACK_OFFSET
    set_from_zero(m, "stack_pointer", stack_pointer)
    set_from_zero(m, "stack_pointer_1", stack_pointer - 1)
    set_from_zero(m, "stack_pointer_2", stack_pointer - 2)
    set_from_zero(m, "stack_pointer_3", stack_pointer - 3)
    set_from_zero(m, "tree_pointer", start_pointer + scan_utils.TREE_OFFSET)
    set_from_zero(m, "input_pointer", start_pointer + scan_utils.INPUT_OFFSET)
  # Update step for next layer.
  m.set("processing_step", STEP_PARSE_1)


def get_stack_position(rule, nonterminal_index):
  """Returns position of nonterminal in stack."""
  num_nonterminals = 0
  rule_len = len(rule.source)
  for idx, symbol in enumerate(rule.source):
    if symbol in grammar_utils.NONTERMINALS:
      if num_nonterminals == nonterminal_index:
        if rule_len - idx == 1:
          return STACK_POSITION_1
        elif rule_len - idx == 2:
          return STACK_POSITION_2
        elif rule_len - idx == 3:
          return STACK_POSITION_3
        else:
          raise ValueError("Unsupported rule: %s" % rule)
      num_nonterminals += 1
  return STACK_POSITION_NONE


def set_matched_rule_vars(
    m: pb.MLPBuilder,
    rule_id: int,
    rule: grammar_utils.QCFGRule,
):
  set_from_zero(m, "matched_rule_id", rule_id)
  set_from_zero(m, "matched_rule_len", len(rule.source))
  set_from_zero(m, "matched_rule", 1)
  child_0_stack_position = get_stack_position(rule, nonterminal_index=0)
  set_from_zero(m, "child_0_stack_position", child_0_stack_position)
  child_1_stack_position = get_stack_position(rule, nonterminal_index=1)
  set_from_zero(m, "child_1_stack_position", child_1_stack_position)


def step_parse_1(m: pb.MLPBuilder):
  """Sets matched rule ID and child stack positions."""
  for rule_id, rule in enumerate(grammar_utils.RULES):
    source_symbols = [
        grammar_utils.get_symbol_id(symbol) for symbol in rule.source]
    rule_len = len(rule.source)
    if rule_len == 1:
      for _ in eq(m, "stack_symbol_1", source_symbols[0]):
        set_matched_rule_vars(m, rule_id, rule)
    elif rule_len == 2:
      for _ in eq(m, "stack_symbol_2", source_symbols[0]):
        for _ in eq(m, "stack_symbol_1", source_symbols[1]):
          set_matched_rule_vars(m, rule_id, rule)
    elif rule_len == 3:
      for _ in eq(m, "stack_symbol_3", source_symbols[0]):
        for _ in eq(m, "stack_symbol_2", source_symbols[1]):
          for _ in eq(m, "stack_symbol_1", source_symbols[2]):
            if rule.source in (("S", "and", "S"), ("S", "after", "S")):
              # These rules can only be applied if the input pointer is at EOS.
              for _ in eq(m, "input_pointer_token",
                          scan_utils.get_input_id("eos")):
                set_matched_rule_vars(m, rule_id, rule)
            else:
              set_matched_rule_vars(m, rule_id, rule)
    else:
      raise ValueError("Invalid rule: %s" % rule)
  # Update step.
  m.set("processing_step", STEP_PARSE_2)


def set_child_position(m: pb.MLPBuilder, child_idx: int):
  """Sets position of child node."""
  for child_stack_position in m.get(f"child_{child_idx}_stack_position"):
    if child_stack_position == STACK_POSITION_1:
      for position in m.get("stack_1_node_position"):
        set_from_zero(m, f"child_{child_idx}_position", position)
    elif child_stack_position == STACK_POSITION_2:
      for position in m.get("stack_2_node_position"):
        set_from_zero(m, f"child_{child_idx}_position", position)
    elif child_stack_position == STACK_POSITION_3:
      for position in m.get("stack_3_node_position"):
        set_from_zero(m, f"child_{child_idx}_position", position)
    else:
      set_from_zero(m, f"child_{child_idx}_position", 0)


def step_parse_2(m: pb.MLPBuilder):
  """Sets parse action and child node positions."""
  set_child_position(m, child_idx=0)
  set_child_position(m, child_idx=1)
  m.set("processing_step", STEP_PARSE_3)


def shift(m: pb.MLPBuilder, input_pointer_token: int):
  """Executes shift action."""
  # Increment input pointer.
  for input_pointer in m.get("input_pointer"):
    m.set("input_pointer", input_pointer + 1)
  # Add token at current input position to stack.
  for position, stack_pointer in m.get("position", "stack_pointer"):
    if position == stack_pointer:
      token_symbol_id = grammar_utils.get_symbol_id(
          scan_utils.get_input_token(input_pointer_token)
      )
      m.set("symbol_id", token_symbol_id)
  # Update stack pointers.
  update_stack_pointers(m, difference=1)


def reduce(m: pb.MLPBuilder):
  """Executes reduce action."""

  # Reset nodes being removed from stack.
  for _ in eq(m, "should_reset", 1):
    m.set("symbol_id", 0)
    m.set("tree_node_position", 0)

  # Add LHS nonterminal to stack.
  for _ in eq(m, "is_next_nt", 1):
    for matched_rule_id in m.get("matched_rule_id"):
      matched_rule = grammar_utils.RULES[matched_rule_id]
      lhs_id = grammar_utils.get_symbol_id(matched_rule.lhs)
      m.set("symbol_id", lhs_id)
    # The tree node associated with this nonterminal on the stack is going
    # to be located at the position identified by `tree_pointer`.
    for tree_pointer in m.get("tree_pointer"):
      m.set("tree_node_position", tree_pointer)

  # Update stack pointers.
  for matched_rule_len in m.get("matched_rule_len"):
    update_stack_pointers(m, 1 - matched_rule_len)

  # Add node to tree.
  for _ in eq(m, "is_tree_pointer", 1):
    for matched_rule_id in m.get("matched_rule_id"):
      # Use 1-indexing to reserve 0 for no rule.
      set_from_zero(m, "rule_id", matched_rule_id + 1)
    # Set pointers to child nodes.
    for child_0_position in m.get("child_0_position"):
      set_from_zero(m, "tree_child_0", child_0_position)
    for child_1_position in m.get("child_1_position"):
      set_from_zero(m, "tree_child_1", child_1_position)

  # Set pointer to parent node.
  for tree_pointer in m.get("tree_pointer"):
    for is_child_0, is_child_1 in m.get("is_child_0", "is_child_1"):
      if is_child_0 or is_child_1:
        set_from_zero(m, "tree_parent", tree_pointer)

  # Increment tree pointer.
  for tree_pointer in m.get("tree_pointer"):
    m.set("tree_pointer", tree_pointer + 1)


def step_parse_3(m: pb.MLPBuilder):
  """Sets indicator variables for certain positions used for reduce op."""
  # Set `is_next_nt`, `is_child_0`, `is_child_1`, `is_tree_pointer`.
  for matched_rule_len, position, stack_pointer in m.get(
      "matched_rule_len", "position", "stack_pointer"
  ):
    if position == (stack_pointer - matched_rule_len):
      set_from_zero(m, "is_next_nt", 1)
    if position < stack_pointer and position > (
        stack_pointer - matched_rule_len
    ):
      set_from_zero(m, "should_reset", 1)

  for position, tree_pointer in m.get("position", "tree_pointer"):
    if position == tree_pointer:
      set_from_zero(m, "is_tree_pointer", 1)

  for position in m.get("position"):
    for child_0_position in m.get("child_0_position"):
      if child_0_position > 0 and position == child_0_position:
        set_from_zero(m, "is_child_0", 1)
    for child_1_position in m.get("child_1_position"):
      if child_1_position > 0 and position == child_1_position:
        set_from_zero(m, "is_child_1", 1)

  m.set("processing_step", STEP_PARSE_4)


def step_parse_4(m: pb.MLPBuilder):
  """Executes parse action."""
  for matched_rule in m.get("matched_rule"):
    if matched_rule == 1:
      # A rule matched, so execute a reduce action.
      reduce(m)
      m.set("processing_step", STEP_PARSE_1)
    else:
      # No rule matched, so execute shift action or finish.
      for input_pointer_token in m.get("input_pointer_token"):
        if input_pointer_token == scan_utils.get_input_id("eos"):
          # We are finished with parsing. Start decoding.
          m.set("processing_step", STEP_INIT_DECODE)
        else:
          # Execute a shift action.
          shift(m, input_pointer_token)
          m.set("processing_step", STEP_PARSE_1)

  # Reset variables used for parsing.
  m.set("matched_rule", 0)
  m.set("matched_rule_id", 0)
  m.set("matched_rule_len", 0)
  m.set("is_next_nt", 0)
  m.set("is_child_0", 0)
  m.set("is_child_1", 0)
  m.set("is_tree_pointer", 0)
  m.set("should_reset", 0)
  m.set("child_0_stack_position", 0)
  m.set("child_1_stack_position", 0)
  m.set("child_0_position", 0)
  m.set("child_1_position", 0)


def step_init_decode(m: pb.MLPBuilder):
  """Initializes decoding state."""
  # Initialize output pointer.
  for start_pointer in m.get("start_pointer"):
    for _ in eq(m, "output_pointer", 0):
      m.set("output_pointer", start_pointer + scan_utils.OUTPUT_OFFSET)

  # Set tree pointer to root node.
  for tree_pointer in m.get("tree_pointer"):
    m.set("tree_pointer", tree_pointer - 1)
    # Set current node pointer to root also.
    m.set("current_node_pointer", tree_pointer - 1)

  # Update processing step.
  m.set("processing_step", STEP_DECODE_1)


def get_child_nonterminal_idx(rule, symbol_index):
  nonterminal_count = 0
  for idx in range(symbol_index):
    if rule.target[idx] in grammar_utils.NONTERMINALS:
      nonterminal_count += 1
  return rule.mapping[nonterminal_count]


def step_decode_1(m: pb.MLPBuilder):
  """Set `decode_action`, `decode_symbol` and `current_node_is_root`."""
  for current_node_pointer, tree_pointer in m.get(
      "current_node_pointer", "tree_pointer"
  ):
    if current_node_pointer == tree_pointer:
      m.set("current_node_is_root", 1)

  for current_node_rule_id, current_node_symbol_index in m.get(
      "current_node_rule_id", "current_node_symbol_index"
  ):
    # Use 1-indexing to reserve 0 for no rule.
    rule_id = current_node_rule_id - 1
    rule = grammar_utils.RULES[rule_id]
    num_symbols = len(rule.target)
    if num_symbols == current_node_symbol_index:
      set_from_zero(m, "decode_action", DECODE_ACTION_PARENT)
    elif num_symbols < current_node_symbol_index:
      # Invalid input value.
      continue
    else:
      decode_symbol = rule.target[current_node_symbol_index]
      if decode_symbol in grammar_utils.TARGET_TERMINALS:
        set_from_zero(m, "decode_action", DECODE_ACTION_COPY)
        set_from_zero(
            m, "decode_symbol", grammar_utils.get_symbol_id(decode_symbol)
        )
      elif decode_symbol in grammar_utils.NONTERMINALS:
        child_nonterminal_idx = get_child_nonterminal_idx(
            rule, current_node_symbol_index
        )
        if child_nonterminal_idx == 0:
          set_from_zero(m, "decode_action", DECODE_ACTION_CHILD_0)
        elif child_nonterminal_idx == 1:
          set_from_zero(m, "decode_action", DECODE_ACTION_CHILD_1)
        else:
          raise ValueError(
              "Unsupported child nonterminal index: %s" % child_nonterminal_idx
          )
      else:
        raise ValueError("Unsupported decode symbol: %s" % decode_symbol)

  m.set("processing_step", STEP_DECODE_2)


def step_decode_2(m: pb.MLPBuilder):
  """Determine if decoding is complete."""
  for decode_action, current_node_is_root in m.get(
      "decode_action", "current_node_is_root"
  ):
    if decode_action == DECODE_ACTION_PARENT and current_node_is_root:
      m.set("processing_step", STEP_DONE)
    else:
      m.set("processing_step", STEP_DECODE_3)
  # Reset `current_node_is_root`.
  m.set("current_node_is_root", 0)


def copy_terminal(m: pb.MLPBuilder):
  """Copy terminal symbol from current tree node to output buffer."""
  for decode_symbol in m.get("decode_symbol"):
    # Add symbol to output and increment symbol index.
    for output_pointer, position in m.get("output_pointer", "position"):
      if position == output_pointer:
        set_from_zero(m, "output_symbol_id", decode_symbol)
  # Increment output pointer.
  for output_pointer in m.get("output_pointer"):
    m.set("output_pointer", output_pointer + 1)
  # Increment symbol index.
  for position, current_node_pointer in m.get(
      "position", "current_node_pointer"
  ):
    if position == current_node_pointer:
      for symbol_index in m.get("symbol_index"):
        m.set("symbol_index", symbol_index + 1)


def go_to_parent(m: pb.MLPBuilder):
  """Set current tree node to be parent of current node."""
  # First, reset symbol index of child.
  for position, current_node_pointer in m.get(
      "position", "current_node_pointer"
  ):
    if position == current_node_pointer:
      m.set("symbol_index", 0)
  for current_node_parent in m.get("current_node_parent"):
    m.set("current_node_pointer", current_node_parent)


def go_to_child(m: pb.MLPBuilder, child_idx: int):
  """Set current tree node to be child of current node."""
  for position, current_node_pointer in m.get(
      "position", "current_node_pointer"
  ):
    if position == current_node_pointer:
      for symbol_index in m.get("symbol_index"):
        m.set("symbol_index", symbol_index + 1)
  for current_node_child_idx in m.get(f"current_node_child_{child_idx}"):
    m.set("current_node_pointer", current_node_child_idx)


def step_decode_3(m: pb.MLPBuilder):
  """Execute decoding action."""
  for decode_action in m.get("decode_action"):
    if decode_action == DECODE_ACTION_COPY:
      copy_terminal(m)
    elif decode_action == DECODE_ACTION_PARENT:
      go_to_parent(m)
    elif decode_action == DECODE_ACTION_CHILD_0:
      go_to_child(m, child_idx=0)
    elif decode_action == DECODE_ACTION_CHILD_1:
      go_to_child(m, child_idx=1)
    elif decode_action == DECODE_ACTION_NONE:
      pass
    else:
      raise ValueError("Unsupported decode action: %s" % decode_action)
  m.set("processing_step", STEP_DECODE_1)

  # Reset some values.
  m.set("decode_action", 0)
  m.set("decode_symbol", 0)
  m.set("current_node_is_root", 0)


def get_rules(m: pb.MLPBuilder):
  """Returns rules for the MLP."""
  for processing_step in m.get("processing_step"):
    if processing_step == STEP_INIT_PARSE:
      step_init_parse(m)
    elif processing_step == STEP_PARSE_1:
      step_parse_1(m)
    elif processing_step == STEP_PARSE_2:
      step_parse_2(m)
    elif processing_step == STEP_PARSE_3:
      step_parse_3(m)
    elif processing_step == STEP_PARSE_4:
      step_parse_4(m)
    elif processing_step == STEP_INIT_DECODE:
      step_init_decode(m)
    elif processing_step == STEP_DECODE_1:
      step_decode_1(m)
    elif processing_step == STEP_DECODE_2:
      step_decode_2(m)
    elif processing_step == STEP_DECODE_3:
      step_decode_3(m)
  return m.rules


def build_program_spec(max_num_padding: int = 0):
  """Returns a program spec for SCAN task."""
  num_positions = scan_utils.get_num_positions(max_num_padding)

  variables = {}
  heads = {}

  variables["token"] = pb.input_var(scan_utils.NUM_INPUT_TOKENS)
  variables["position"] = pb.position_var(num_positions)
  # Add variable to represent current parsing state.
  variables["processing_step"] = pb.var(NUM_STEPS, default=STEP_INIT_PARSE)

  # Add variable to represent matched rule ID for reduce operation.
  variables["matched_rule_id"] = pb.var(scan_utils.NUM_RULES)
  variables["matched_rule_len"] = pb.var(4)
  variables["matched_rule"] = pb.var(2)

  # Index of nonterminals in matched rule.
  variables["child_0_stack_position"] = pb.var(NUM_STACK_POSITIONS)
  variables["child_1_stack_position"] = pb.var(NUM_STACK_POSITIONS)
  # Next symbol to decode.
  variables["decode_symbol"] = pb.var(scan_utils.NUM_SYMBOLS)
  # Whether current tree node is root.
  variables["current_node_is_root"] = pb.var(2)
  # Add variable to represent current decode action.
  variables["decode_action"] = pb.var(NUM_DECODE_ACTIONS)

  # Represent positions of child nodes of nonterminals in stack.
  variables["child_0_position"] = pb.var(num_positions)
  variables["child_1_position"] = pb.var(num_positions)

  # Add attention head that returns position of start token.
  variables["start_token_id"] = pb.var(
      scan_utils.NUM_INPUT_TOKENS, default=scan_utils.get_input_id("start")
  )

  heads["start_pointer"] = pb.qkv("start_token_id", "token", "position")

  variables["stack_pointer"] = pb.var(num_positions)
  variables["tree_pointer"] = pb.var(num_positions)
  variables["input_pointer"] = pb.var(num_positions)
  # Stores grammar rule associated with tree node.
  variables["rule_id"] = pb.var(scan_utils.NUM_RULES + 1)
  # Stores symbol ID associated with stack element.
  variables["symbol_id"] = pb.var(scan_utils.NUM_SYMBOLS)
  # Get token at input pointer.
  heads["input_pointer_token"] = pb.qkv("input_pointer", "position", "token")
  # Get top 3 symbols on stack.
  variables["stack_pointer_1"] = pb.var(num_positions)
  variables["stack_pointer_2"] = pb.var(num_positions)
  variables["stack_pointer_3"] = pb.var(num_positions)

  heads["stack_symbol_1"] = pb.qkv("stack_pointer_1", "position", "symbol_id")
  heads["stack_symbol_2"] = pb.qkv("stack_pointer_2", "position", "symbol_id")
  heads["stack_symbol_3"] = pb.qkv("stack_pointer_3", "position", "symbol_id")

  # Track tree node position associated with nonterminal on stack.
  variables["tree_node_position"] = pb.var(num_positions)
  # Get tree node positions associated with top 3 symbols on stack.
  heads["stack_1_node_position"] = pb.qkv(
      "stack_pointer_1", "position", "tree_node_position"
  )
  heads["stack_2_node_position"] = pb.qkv(
      "stack_pointer_2", "position", "tree_node_position"
  )
  heads["stack_3_node_position"] = pb.qkv(
      "stack_pointer_3", "position", "tree_node_position"
  )

  # Track child nodes for nodes in tree.
  variables["tree_child_0"] = pb.var(num_positions)
  variables["tree_child_1"] = pb.var(num_positions)
  variables["tree_parent"] = pb.var(num_positions)
  # Positional indicator variables used during parsing.
  variables["is_child_0"] = pb.var(2)
  variables["is_child_1"] = pb.var(2)
  variables["is_next_nt"] = pb.var(2)
  variables["should_reset"] = pb.var(2)
  variables["is_tree_pointer"] = pb.var(2)

  # Tracks output pointer.
  variables["output_pointer"] = pb.var(num_positions)
  # Tracks output symbol ID.
  variables["output_symbol_id"] = pb.var(scan_utils.NUM_SYMBOLS)
  # Tracks index of target symbol.
  # Longest rule has 8 symbols.
  variables["symbol_index"] = pb.var(9)
  # Track current node.
  variables["current_node_pointer"] = pb.var(num_positions)
  # Attenion heads related to current node.
  heads["current_node_rule_id"] = pb.qkv(
      "current_node_pointer", "position", "rule_id"
  )
  heads["current_node_child_0"] = pb.qkv(
      "current_node_pointer", "position", "tree_child_0"
  )
  heads["current_node_child_1"] = pb.qkv(
      "current_node_pointer", "position", "tree_child_1"
  )
  heads["current_node_parent"] = pb.qkv(
      "current_node_pointer", "position", "tree_parent"
  )
  heads["current_node_symbol_index"] = pb.qkv(
      "current_node_pointer", "position", "symbol_index"
  )

  m = pb.MLPBuilder(variables, heads)
  get_rules(m)
  return pb.program_spec_from_rules(
      variables=variables,
      heads=heads,
      rules=m.rules,
      output_name="output_symbol_id",
      input_range=scan_utils.NUM_INPUT_TOKENS,
      position_range=num_positions,
  )
