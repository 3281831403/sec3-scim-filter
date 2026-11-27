import pytest

from scimf.filter import (
    FilterSyntaxError,
    check_attrs,
    eval_filter,
    filter_batch,
    parse,
    tokenize,
)


RECORDS = [
    {
        "userName": "alice",
        "displayName": "Alice",
        "active": False,
        "age": 40,
        "department": "sales",
    },
    {
        "userName": "bob",
        "displayName": "Bob",
        "active": True,
        "age": 10,
        "department": "eng",
    },
    {
        "userName": "carol",
        "displayName": "Carol",
        "active": True,
        "age": 9,
        "department": "eng",
    },
]


def _names(records):
    return [record["userName"] for record in records]


def test_tokenize_produces_typed_lexemes():
    expression = '(userName eq "alice") and active eq true'
    assert tokenize(expression) == [
        ("LPAREN", "("),
        ("IDENT", "userName"),
        ("OP", "eq"),
        ("STRING", "alice"),
        ("RPAREN", ")"),
        ("AND", "and"),
        ("IDENT", "active"),
        ("OP", "eq"),
        ("BOOL", True),
    ]

    with pytest.raises(FilterSyntaxError):
        parse(tokenize('(userName eq "alice"'))


def test_parse_respects_and_before_or():
    ast = parse(
        tokenize(
            'userName eq "alice" or department eq "eng" and active eq true'
        )
    )
    assert ast[0] == "or"
    assert ast[2][0] == "and"


def test_check_attrs_rejects_unlisted_attribute():
    ast = parse(
        tokenize('userName eq "alice" and department eq "eng"')
    )
    assert check_attrs(ast, {"userName"}) is False

    allowed_ast = parse(tokenize('userName eq "alice"'))
    assert check_attrs(allowed_ast, {"userName"}) is True


def test_eval_filter_uses_numeric_comparison():
    older = parse(tokenize("age gt 9"))
    assert eval_filter(older, {"age": 10}) is True
    assert eval_filter(older, {"age": 9}) is False

    active = parse(tokenize("active eq true"))
    assert eval_filter(active, {"active": True}) is True
    assert eval_filter(active, {"active": False}) is False


def test_filter_batch_keeps_requests_independent():
    queries = [
        'userName eq "alice" and active eq false',
        'userName eq "bob" and active eq true',
    ]
    result = filter_batch(queries, RECORDS)

    assert _names(result[queries[0]]) == ["alice"]
    assert _names(result[queries[1]]) == ["bob"]


def test_empty_and_unbalanced_expressions_are_rejected():
    empty = parse(tokenize(""))
    assert empty == ("empty",)
    assert eval_filter(empty, RECORDS[0]) is False
    with pytest.raises(FilterSyntaxError):
        parse(tokenize('(userName eq "alice"'))
    with pytest.raises(FilterSyntaxError):
        parse(tokenize('userName eq "alice")'))


def test_check_attrs_enforces_whitelist_and_defaults_to_allowed_set():
    assert check_attrs(parse(tokenize('userName eq "alice"')), {"userName"})
    assert not check_attrs(
        parse(tokenize('userName eq "alice" or password pr')),
        {"userName"},
    )
    assert not check_attrs(parse(tokenize('password eq "x"')))


def test_eval_filter_rejects_cross_type_comparisons_without_raising():
    assert eval_filter(parse(tokenize("age gt 9")), {"age": "zzz"}) is False
    assert eval_filter(parse(tokenize("active eq 1")), {"active": True}) is False
    assert eval_filter(
        parse(tokenize('active eq "true"')), {"active": True}
    ) is False


def test_parse_is_left_associative_at_equal_precedence():
    ast = parse(tokenize('userName eq "a" and userName eq "b" and userName eq "c"'))
    assert ast[0] == "and" and ast[1][0] == "and"


def test_filter_batch_does_not_fall_back_to_substring_matching():
    result = filter_batch(["alice", 'department eq "sales"'], RECORDS)
    assert result["alice"] == []
    assert _names(result['department eq "sales"']) == ["alice"]


def test_filter_batch_isolates_invalid_requests():
    queries = [
        'userName eq "alice"',
        "(unbalanced",
        'password eq "x"',
        'userName eq "bob"',
    ]
    result = filter_batch(queries, RECORDS)
    assert _names(result[queries[0]]) == ["alice"]
    assert result[queries[1]] == []
    assert result[queries[2]] == []
    assert _names(result[queries[3]]) == ["bob"]