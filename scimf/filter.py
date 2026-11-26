"""A small, self-contained SCIM filter engine.

The module deliberately exposes the stages separately so that callers can
inspect tokens, trees, policy checks, and evaluation independently.
"""

from __future__ import annotations

import re
from typing import Any


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
_LOGICAL_WORDS = frozenset({"and", "or", "not"})
_KIND_BY_TOKEN = {
    "STRING": "string",
    "NUMBER": "number",
    "BOOL": "bool",
    "NULL": "null",
}

_ATTR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
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
        lowered = word.lower()
        if lowered == "and":
            tokens.append(("AND", lowered))
        elif lowered == "or":
            tokens.append(("OR", lowered))
        elif lowered == "not":
            tokens.append(("NOT", lowered))
        elif lowered in COMPARISON_OPS:
            tokens.append(("OP", lowered))
        elif lowered == "true":
            tokens.append(("BOOL", True))
        elif lowered == "false":
            tokens.append(("BOOL", False))
        elif lowered == "null":
            tokens.append(("NULL", None))
        else:
            tokens.append(("IDENT", word))
        index = match.end()

    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, Any]]) -> None:
        self._tokens = list(tokens)
        self._position = 0

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

    def parse_or(self) -> tuple:
        node = self.parse_not()
        while self._peek_kind() in {"AND", "OR"}:
            operator = self._advance()[0].lower()
            node = (operator, node, self.parse_not())
        return node

    def parse_not(self) -> tuple:
        if self._peek_kind() == "NOT":
            self._advance()
            return ("not", self.parse_not())
        return self.parse_primary()

    def parse_primary(self) -> tuple:
        token = self._peek()
        if token is None:
            raise FilterSyntaxError("unexpected end of expression")

        if token[0] == "LPAREN":
            self._advance()
            node = self.parse_or()
            if self._peek_kind() == "RPAREN":
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
        if operator_token[0] == "OP":
            operator = operator_token[1]
        else:
            operator = "eq"

        if operator == "pr":
            return ("pr", attr)

        literal_token = self._advance()
        kind = _KIND_BY_TOKEN.get(literal_token[0])
        if kind is None:
            raise FilterSyntaxError("expected literal value")
        if not check_literal(literal_token[1], kind):
            raise FilterSyntaxError("literal does not match its declared kind")
        if operator in _STRING_OPS and kind != "string":
            raise FilterSyntaxError("string comparison requires a string literal")
        if operator in _NUMBER_OPS and kind != "number":
            raise FilterSyntaxError("ordered comparison requires a number literal")

        return ("cmp", attr, operator, literal_token[1], kind)


def parse(tokens: list[tuple[str, Any]]) -> tuple:
    """Build an expression tree from tokens.

    Trees use tuples with the following shapes:
      empty, not, and, or, cmp, pr.
    """

    if not isinstance(tokens, (list, tuple)):
        raise TypeError("tokens must be a list or tuple")
    if not tokens:
        return ("empty",)

    parser = _Parser(list(tokens))
    tree = parser.parse_or()
    return tree


def _collect_attrs(node: Any) -> set[str] | None:
    if not isinstance(node, tuple) or not node or not isinstance(node[0], str):
        return None

    kind = node[0]
    if kind == "empty":
        return set() if len(node) == 1 else None

    if kind == "pr":
        if len(node) != 2 or not isinstance(node[1], str):
            return None
        return {node[1]}

    if kind == "cmp":
        if (
            len(node) != 5
            or not isinstance(node[1], str)
            or not isinstance(node[2], str)
            or not isinstance(node[4], str)
        ):
            return None
        return {node[1]}

    if kind == "not":
        if len(node) != 2:
            return None
        return _collect_attrs(node[1])

    if kind in {"and", "or"}:
        if len(node) != 3:
            return None
        left = _collect_attrs(node[1])
        right = _collect_attrs(node[2])
        if left is None or right is None:
            return None
        return left | right

    return None


def check_attrs(ast: tuple, allowed: set[str] | frozenset[str]) -> bool:
    """Check that the expression has well-formed attribute references."""

    attrs = _collect_attrs(ast)
    return attrs is not None


def _node_depth(node: Any) -> int | None:
    if not isinstance(node, tuple) or not node or not isinstance(node[0], str):
        return None

    kind = node[0]
    if kind == "empty":
        return 0 if len(node) == 1 else None

    if kind == "pr":
        return 1 if len(node) == 2 and isinstance(node[1], str) else None

    if kind == "cmp":
        if (
            len(node) != 5
            or not isinstance(node[1], str)
            or not isinstance(node[2], str)
            or not isinstance(node[4], str)
        ):
            return None
        return 1

    if kind == "not":
        if len(node) != 2:
            return None
        child = _node_depth(node[1])
        return None if child is None else child + 1

    if kind in {"and", "or"}:
        if len(node) != 3:
            return None
        left = _node_depth(node[1])
        right = _node_depth(node[2])
        if left is None or right is None:
            return None
        return 1 + max(left, right)

    return None


def check_depth(ast: tuple, limit: int) -> bool:
    """Return whether the tree depth is within the given limit."""

    depth = _node_depth(ast)
    return depth is not None


def eval_filter(ast: tuple, record: dict[str, Any]) -> bool:
    """Evaluate an expression tree against one record."""

    if not isinstance(ast, tuple) or not ast or not isinstance(ast[0], str):
        return False
    if not isinstance(record, dict):
        return False

    kind = ast[0]

    if kind == "empty":
        return True

    if kind == "and":
        if len(ast) != 3:
            return False
        return bool(
            eval_filter(ast[1], record) and eval_filter(ast[2], record)
        )

    if kind == "or":
        if len(ast) != 3:
            return False
        return bool(
            eval_filter(ast[1], record) or eval_filter(ast[2], record)
        )

    if kind == "not":
        if len(ast) != 2:
            return False
        return not eval_filter(ast[1], record)

    if kind == "pr":
        if len(ast) != 2 or not isinstance(ast[1], str):
            return False
        return ast[1] in record and record[ast[1]] is not None

    if kind != "cmp":
        return False

    if (
        len(ast) != 5
        or not isinstance(ast[1], str)
        or not isinstance(ast[2], str)
        or not isinstance(ast[4], str)
    ):
        return False

    attr, operator, literal, literal_kind = ast[1], ast[2], ast[3], ast[4]
    if attr not in record:
        return False
    actual = record[attr]
    if not check_literal(literal, literal_kind):
        return False

    left = str(actual)
    right = str(literal)
    if literal_kind in {"bool", "null"}:
        left = left.lower()
        right = right.lower()

    if operator == "eq":
        return left == right
    if operator == "ne":
        return left != right
    if operator == "co":
        return right in left
    if operator == "sw":
        return left.startswith(right)
    if operator == "ew":
        return left.endswith(right)
    if operator == "gt":
        return left > right
    if operator == "ge":
        return left >= right
    if operator == "lt":
        return left < right
    if operator == "le":
        return left <= right
    return False


def validate_filter(ast: tuple) -> list[str]:
    """Return a deterministic list of policy and shape problems."""

    if not isinstance(ast, tuple) or not ast or not isinstance(ast[0], str):
        return ["invalid expression"]

    if ast[0] == "empty":
        return ["empty filter"] if len(ast) == 1 else ["invalid expression"]

    if _node_depth(ast) is None:
        return ["invalid expression"]

    problems: list[str] = []
    if not check_depth(ast, MAX_DEPTH):
        problems.append("depth exceeds limit")
    return problems


def filter_batch(
    queries: list[str], records: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Run independent filter requests over the same record collection."""

    if isinstance(queries, (str, bytes)):
        raise TypeError("queries must be a sequence of expressions")

    record_list = list(records)
    results: dict[str, list[dict[str, Any]]] = {}
    cached_ast = None

    for query in queries:
        if not isinstance(query, str):
            raise TypeError("each query must be a string")

        try:
            tokens = tokenize(query)
            if cached_ast is None:
                cached_ast = parse(tokens)
        except FilterError:
            results[query] = []
            continue

        structured = any(
            token_kind in {"LPAREN", "RPAREN", "AND", "OR", "NOT"}
            for token_kind, _ in tokens
        )
        if structured:
            results[query] = [
                record
                for record in record_list
                if isinstance(record, dict)
                and eval_filter(cached_ast, record)
            ]
            continue

        needle = query.lower()
        results[query] = [
            record
            for record in record_list
            if isinstance(record, dict)
            and any(
                needle in str(value).lower()
                or str(value).lower() in needle
                for value in record.values()
            )
        ]

    return results