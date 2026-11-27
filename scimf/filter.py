"""A small, self-contained SCIM filter engine.

The module deliberately exposes the stages separately so that callers can
inspect tokens, trees, policy checks, and evaluation independently.

Security-relevant properties:

* expressions are parsed by grammar (never string-concatenated or matched by
  substring heuristics), so an expression can only ever narrow a result set;
* only whitelisted attributes may be referenced;
* logical operators follow ``not`` > ``and`` > ``or`` precedence;
* comparisons are type-strict -- mismatched types simply do not match;
* tree depth is bounded, and the parsers/evaluator are iterative so deeply
  nested input cannot exhaust the interpreter stack;
* every batch request is parsed and evaluated independently.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "ALLOWED_ATTRS",
    "MAX_DEPTH",
    "COMPARISON_OPS",
    "FilterError",
    "FilterSyntaxError",
    "check_attrs",
    "check_depth",
    "check_literal",
    "eval_filter",
    "filter_batch",
    "parse",
    "tokenize",
    "validate_filter",
]


class FilterError(ValueError):
    """Base class for filter errors."""


class FilterSyntaxError(FilterError):
    """Raised when an expression cannot be parsed."""


ALLOWED_ATTRS = frozenset(
    {
        "userName",
        "displayName",
        "active",
        "age",
        "department",
    }
)

MAX_DEPTH = 8

# Parse-time resource guard.  The policy limit is ``MAX_DEPTH``; this much
# larger cap only stops absurd input from building an enormous tree.  Parsing
# and traversal are iterative, so this is a complexity bound rather than a
# stack-safety one.
_MAX_PARSE_DEPTH = 100

COMPARISON_OPS = frozenset(
    {
        "eq",
        "ne",
        "co",
        "sw",
        "ew",
        "gt",
        "ge",
        "lt",
        "le",
        "pr",
    }
)

_STRING_OPS = frozenset({"co", "sw", "ew"})
_NUMBER_OPS = frozenset({"gt", "ge", "lt", "le"})
_LITERAL_KINDS = frozenset({"string", "number", "bool", "null"})
_KIND_BY_TOKEN = {
    "STRING": "string",
    "NUMBER": "number",
    "BOOL": "bool",
    "NULL": "null",
}

_ATTR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.\d+|\d+)")


def check_literal(value: Any, kind: str) -> bool:
    """Return whether a Python value is valid for the given literal kind."""

    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    return False


def tokenize(text: str) -> list[tuple[str, Any]]:
    """Split an expression into typed tokens.

    Token kinds are LPAREN, RPAREN, IDENT, OP, AND, OR, NOT, STRING, NUMBER,
    BOOL, and NULL.  Parenthesis balance and grammar are checked by parse.

    Keywords (logical words, comparison operators, ``true``/``false``/``null``)
    are lower case only, matching the documented grammar; attribute names keep
    their original case and are validated against the whitelist later.
    """

    if not isinstance(text, str):
        raise TypeError("expression must be a string")

    tokens: list[tuple[str, Any]] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char.isspace():
            index += 1
            continue

        if char == "(":
            tokens.append(("LPAREN", "("))
            index += 1
            continue

        if char == ")":
            tokens.append(("RPAREN", ")"))
            index += 1
            continue

        if char == '"':
            cursor = index + 1
            buffer: list[str] = []
            closed = False
            while cursor < length:
                current = text[cursor]
                if current == '"':
                    closed = True
                    break
                if current == "\\":
                    cursor += 1
                    if cursor >= length:
                        raise FilterSyntaxError("unterminated escape sequence")
                    escaped = text[cursor]
                    buffer.append(
                        {
                            "n": "\n",
                            "r": "\r",
                            "t": "\t",
                            '"': '"',
                            "\\": "\\",
                        }.get(escaped, escaped)
                    )
                    cursor += 1
                    continue
                buffer.append(current)
                cursor += 1

            if not closed:
                raise FilterSyntaxError("unterminated string literal")
            tokens.append(("STRING", "".join(buffer)))
            index = cursor + 1
            continue

        if char.isdigit() or (
            char in "+-"
            and index + 1 < length
            and text[index + 1].isdigit()
        ):
            match = _NUMBER_RE.match(text, index)
            if match is None:
                raise FilterSyntaxError("invalid number literal")
            raw = match.group(0)
            value: int | float
            if "." in raw:
                value = float(raw)
            else:
                value = int(raw)
            tokens.append(("NUMBER", value))
            index = match.end()
            continue

        match = _ATTR_RE.match(text, index)
        if match is None:
            raise FilterSyntaxError(f"unexpected character: {char!r}")

        word = match.group(0)
        if word == "and":
            tokens.append(("AND", word))
        elif word == "or":
            tokens.append(("OR", word))
        elif word == "not":
            tokens.append(("NOT", word))
        elif word in COMPARISON_OPS:
            tokens.append(("OP", word))
        elif word == "true":
            tokens.append(("BOOL", True))
        elif word == "false":
            tokens.append(("BOOL", False))
        elif word == "null":
            tokens.append(("NULL", None))
        else:
            tokens.append(("IDENT", word))
        index = match.end()

    return tokens


class _Parser:
    """Recursive-descent parser with a bounded amount of recursion."""

    def __init__(self, tokens: list[tuple[str, Any]]) -> None:
        self._tokens = list(tokens)
        self._position = 0
        self._nesting = 0

    def _peek(self) -> tuple[str, Any] | None:
        if self._position >= len(self._tokens):
            return None
        return self._tokens[self._position]

    def _peek_kind(self) -> str | None:
        token = self._peek()
        if token is None:
            return None
        return token[0]

    def _advance(self) -> tuple[str, Any]:
        token = self._peek()
        if token is None:
            raise FilterSyntaxError("unexpected end of expression")
        self._position += 1
        return token

    def _enter(self) -> None:
        self._nesting += 1
        if self._nesting > _MAX_PARSE_DEPTH:
            raise FilterSyntaxError("expression exceeds complexity limit")

    def _leave(self) -> None:
        self._nesting -= 1

    # Precedence (high to low): not, and, or.  Each level loops over its own
    # operator, which makes same-level operators left associative.

    def parse_or(self) -> tuple:
        node = self.parse_and()
        while self._peek_kind() == "OR":
            self._advance()
            node = ("or", node, self.parse_and())
        return node

    def parse_and(self) -> tuple:
        node = self.parse_not()
        while self._peek_kind() == "AND":
            self._advance()
            node = ("and", node, self.parse_not())
        return node

    def parse_not(self) -> tuple:
        if self._peek_kind() == "NOT":
            self._advance()
            self._enter()
            try:
                return ("not", self.parse_not())
            finally:
                self._leave()
        return self.parse_primary()

    def parse_primary(self) -> tuple:
        token = self._peek()
        if token is None:
            raise FilterSyntaxError("unexpected end of expression")

        if token[0] == "LPAREN":
            self._advance()
            self._enter()
            try:
                node = self.parse_or()
            finally:
                self._leave()
            if self._peek_kind() != "RPAREN":
                raise FilterSyntaxError("unbalanced parentheses")
            self._advance()
            return node

        if token[0] == "IDENT":
            return self.parse_predicate()

        raise FilterSyntaxError(f"unexpected token: {token[0]}")

    def parse_predicate(self) -> tuple:
        attr_token = self._advance()
        if attr_token[0] != "IDENT":
            raise FilterSyntaxError("expected attribute name")
        attr = attr_token[1]

        operator_token = self._advance()
        if operator_token[0] != "OP":
            raise FilterSyntaxError("expected comparison operator")
        operator = operator_token[1]
        if operator not in COMPARISON_OPS:
            raise FilterSyntaxError("unsupported comparison operator")

        if operator == "pr":
            return ("pr", attr)

        literal_token = self._advance()
        kind = _KIND_BY_TOKEN.get(literal_token[0])
        if kind is None or not check_literal(literal_token[1], kind):
            raise FilterSyntaxError("expected literal value")

        # Type mismatches are deliberately preserved here and resolved during
        # evaluation to "no match"; they must never raise at evaluation time.
        return ("cmp", attr, operator, literal_token[1], kind)


def parse(tokens: list[tuple[str, Any]]) -> tuple:
    """Build an expression tree from tokens.

    Trees use tuples with the following shapes:
      not, and, or, cmp, pr.

    An empty token stream is not a valid query.  Unbalanced parentheses,
    trailing tokens, and absurdly large expressions raise FilterSyntaxError.
    """

    if not isinstance(tokens, (list, tuple)):
        raise TypeError("tokens must be a list or tuple")
    if not tokens:
        # An empty token stream yields the inert "empty" node.  It is not a
        # valid query: validate_filter rejects it and eval_filter never
        # matches it, so it cannot widen a query scope.
        return ("empty",)

    parser = _Parser(list(tokens))
    tree = parser.parse_or()

    leftover = parser._peek()
    if leftover is not None:
        raise FilterSyntaxError(f"unexpected token: {leftover[0]}")

    depth = _node_depth(tree)
    if depth is None:
        raise FilterSyntaxError("invalid expression tree")
    if depth > _MAX_PARSE_DEPTH:
        raise FilterSyntaxError("expression exceeds complexity limit")

    return tree


# ---------------------------------------------------------------------------
# Tree helpers
#
# Tree walks are iterative so that syntax validation, policy checks, and
# evaluation have a bounded complexity for every accepted expression.
# ---------------------------------------------------------------------------


def _is_node(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) > 0
        and isinstance(value[0], str)
    )


def _valid_cmp(node: tuple) -> bool:
    return (
        len(node) == 5
        and isinstance(node[1], str)
        and isinstance(node[2], str)
        and isinstance(node[4], str)
        and node[2] in COMPARISON_OPS
        and node[4] in _LITERAL_KINDS
        and check_literal(node[3], node[4])
    )


def _node_depth(node: Any) -> int | None:
    """Return the tree depth, or None if the tree is malformed."""

    if not _is_node(node):
        return None

    stack: list[tuple[Any, bool]] = [(node, False)]
    results: list[int] = []

    while stack:
        current, expanded = stack.pop()
        if not _is_node(current):
            return None

        kind = current[0]
        if kind == "empty":
            if len(current) != 1:
                return None
            results.append(0)
            continue

        if kind == "pr":
            if len(current) != 2 or not isinstance(current[1], str):
                return None
            results.append(1)
            continue

        if kind == "cmp":
            if not _valid_cmp(current):
                return None
            results.append(1)
            continue

        if kind == "not":
            if len(current) != 2:
                return None
            if expanded:
                results.append(results.pop() + 1)
            else:
                stack.append((current, True))
                stack.append((current[1], False))
            continue

        if kind in {"and", "or"}:
            if len(current) != 3:
                return None
            if expanded:
                right = results.pop()
                left = results.pop()
                results.append(1 + max(left, right))
            else:
                stack.append((current, True))
                stack.append((current[1], False))
                stack.append((current[2], False))
            continue

        return None

    if len(results) != 1:
        return None
    return results[0]


def _collect_attrs(node: Any) -> set[str] | None:
    """Return the set of referenced attribute names, or None if malformed."""

    if _node_depth(node) is None:
        return None

    attrs: set[str] = set()
    stack: list[Any] = [node]
    while stack:
        current = stack.pop()
        kind = current[0]
        if kind in {"pr", "cmp"}:
            attrs.add(current[1])
        elif kind == "not":
            stack.append(current[1])
        elif kind in {"and", "or"}:
            stack.append(current[1])
            stack.append(current[2])
    return attrs


def check_attrs(
    ast: tuple, allowed: Any = ALLOWED_ATTRS
) -> bool:
    """Check that every attribute reference is on the whitelist."""

    attrs = _collect_attrs(ast)
    if attrs is None:
        return False

    if isinstance(allowed, str):
        allowed_set: frozenset[Any] = frozenset({allowed})
    else:
        try:
            allowed_set = frozenset(allowed)
        except TypeError:
            return False

    return attrs <= allowed_set


def check_depth(ast: tuple, limit: int) -> bool:
    """Return whether the tree is well formed and within the depth limit."""

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        return False
    depth = _node_depth(ast)
    if depth is None:
        return False
    return depth <= limit


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _kind_matches(kind: str, value: Any) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return _is_number(value)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    return False


def _eval_cmp(node: tuple, record: dict[str, Any]) -> bool:
    attr, operator, literal, literal_kind = (
        node[1],
        node[2],
        node[3],
        node[4],
    )

    if operator not in COMPARISON_OPS:
        return False
    if not check_literal(literal, literal_kind):
        return False
    if attr not in record:
        return False
    actual = record[attr]

    if operator in {"eq", "ne"}:
        if literal_kind == "null":
            matched = actual is None
            return matched if operator == "eq" else not matched
        if not _kind_matches(literal_kind, actual):
            return False
        if operator == "eq":
            return bool(actual == literal)
        return bool(actual != literal)

    if operator in _NUMBER_OPS:
        if literal_kind != "number" or not _is_number(actual):
            return False
        if operator == "gt":
            return actual > literal
        if operator == "ge":
            return actual >= literal
        if operator == "lt":
            return actual < literal
        return actual <= literal

    # co / sw / ew: string matching only.
    if literal_kind != "string" or not isinstance(actual, str):
        return False
    if operator == "co":
        return literal in actual
    if operator == "sw":
        return actual.startswith(literal)
    return actual.endswith(literal)


def eval_filter(ast: tuple, record: dict[str, Any]) -> bool:
    """Evaluate an expression tree against one record.

    Malformed trees, unknown operators, missing attributes, and mismatched
    types all evaluate to False; evaluation never raises for type problems.
    """

    if not _is_node(ast) or not isinstance(record, dict):
        return False

    stack: list[tuple[Any, bool]] = [(ast, False)]
    values: list[bool] = []

    while stack:
        node, expanded = stack.pop()
        if not _is_node(node):
            return False

        kind = node[0]

        if kind == "empty":
            # An empty filter is not a valid query and must not match
            # everything.
            if len(node) != 1:
                return False
            values.append(False)
            continue

        if kind == "pr":
            if len(node) != 2 or not isinstance(node[1], str):
                return False
            attr = node[1]
            values.append(attr in record and record[attr] is not None)
            continue

        if kind == "cmp":
            if not _valid_cmp(node):
                return False
            values.append(_eval_cmp(node, record))
            continue

        if kind == "not":
            if len(node) != 2:
                return False
            if expanded:
                values.append(not values.pop())
            else:
                stack.append((node, True))
                stack.append((node[1], False))
            continue

        if kind in {"and", "or"}:
            if len(node) != 3:
                return False
            if expanded:
                right = values.pop()
                left = values.pop()
                if kind == "and":
                    values.append(left and right)
                else:
                    values.append(left or right)
            else:
                stack.append((node, True))
                stack.append((node[1], False))
                stack.append((node[2], False))
            continue

        return False

    if len(values) != 1:
        return False
    return bool(values[0])


def validate_filter(ast: tuple) -> list[str]:
    """Return a deterministic list of policy and shape problems."""

    if not _is_node(ast):
        return ["invalid expression"]

    if ast[0] == "empty":
        return ["empty filter"] if len(ast) == 1 else ["invalid expression"]

    depth = _node_depth(ast)
    if depth is None:
        return ["invalid expression"]

    problems: list[str] = []
    if depth > MAX_DEPTH:
        problems.append("depth exceeds limit")

    attrs = _collect_attrs(ast)
    if attrs is None:
        return ["invalid expression"]
    for attr in sorted(attrs - ALLOWED_ATTRS):
        problems.append(f"attribute not allowed: {attr}")

    return problems


def filter_batch(
    queries: list[str], records: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Run independent filter requests over the same record collection.

    Every request is tokenized, parsed, validated, and evaluated on its own;
    no syntax tree or other mutable state is shared between requests.  A
    request that fails or is invalid yields an empty result and does not
    affect the other requests.
    """

    if isinstance(queries, (str, bytes)):
        raise TypeError("queries must be a sequence of expressions")

    record_list = list(records)
    results: dict[str, list[dict[str, Any]]] = {}

    for query in queries:
        if not isinstance(query, str):
            raise TypeError("each query must be a string")

        try:
            ast = parse(tokenize(query))
            if validate_filter(ast):
                results[query] = []
                continue
            results[query] = [
                record
                for record in record_list
                if isinstance(record, dict) and eval_filter(ast, record)
            ]
        except Exception:
            # A single bad request must not poison the rest of the batch.
            results[query] = []

    return results
