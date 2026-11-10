import pytest
from model.benchmark.src import scoring


@pytest.mark.parametrize(
    ("source", "valid"),
    [
        ("result = sum(values)", True),
        ("print(missing_variable)", True),
        ("raise AssertionError('must not execute')", True),
        ("def answer():\n    return 42", True),
        ("", False),
        (" \n# only a comment\n", False),
        ("```python\nresult = 42\n```", False),
        ("Here is the Python code:\nresult = 42", False),
        ("result = 42\nThe answer is forty two.", False),
        ("result =", False),
        ("return 42", False),
        ("break", False),
        ("result = 42\x00", False),
        ("<|end_code|>", False),
    ],
)
def test_python_validity_checks_unmodified_source_without_executing_it(source, valid):
    result = scoring.score([{"id": "task", "status": "incorrect", "steps": [{"source": source}]}])
    assert result["valid_python"] == int(valid)
    assert result["valid_python_rate"] == float(valid)
    assert result["success_rate"] == 0


def test_syntax_scores_all_responses_and_task_success_scores_tasks():
    rows = [
        {"id": "a", "status": "passed", "steps": [{"source": "result ="}, {"source": "result = 1"}]},
        {"id": "b", "status": "call_limit", "steps": [{"source": "# empty"}, {"source": "print(missing)"}]},
        {"id": "c", "status": "boundary_error", "steps": [{"source": "result = 1"}]},
    ]
    result = scoring.score(rows)
    assert result == {"passed": 1, "total": 3, "generations": 5, "success_rate": 1 / 3, "valid_python": 3, "valid_python_rate": 0.6}
    assert scoring.summary(result) == "Task success: 33.33% (1/3 tasks)\nValid Python: 60.00% (3/5 responses)"


def test_older_results_without_generations_are_not_reported_as_zero_percent():
    result = scoring.score([{"id": "a", "status": "passed"}])
    assert result["valid_python_rate"] is None
    assert "Valid Python: not measured (0 responses)" in scoring.summary(result)


def test_nontext_source_is_rejected():
    with pytest.raises(ValueError, match="must be text"):
        scoring.score([{"id": "a", "status": "passed", "steps": [{"source": 42}]}])
