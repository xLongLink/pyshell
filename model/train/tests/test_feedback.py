import pytest

from model.train.src.generation import expanded, legacy


@pytest.fixture(params=[legacy, expanded], ids=["legacy", "expanded"])
def executor(request):
    return request.param


def test_last_completion_wins_and_resets_per_submission(executor):
    namespace = {}
    result = executor.execute("done('first')\ndone('  last  ')", namespace)
    assert result["exception"] is None
    assert result["done"] == "  last  "
    assert executor.execute("value = 42", namespace)["done"] is None


def test_completion_accepts_message_keyword_and_returns_none(executor):
    result = executor.execute(
        "returned = done(message='first')\nassert returned is None\ndone(message='last')",
        {},
    )
    assert result["exception"] is None
    assert result["done"] == "last"


@pytest.mark.parametrize("message", [None, False, 0, "", "   ", [], {}, [1]])
def test_invalid_completion_suppresses_prior_completion(executor, message):
    result = executor.execute("done('first')\ndone(message)", {"message": message})
    exception = (
        "AssertionError"
        if executor is legacy
        else "ValueError: done requires a nonempty message"
    )
    assert exception in result["exception"]
    assert result["done"] is None


@pytest.mark.parametrize("message", [None, False, 0, "", "   ", [], {}, [1]])
def test_caught_validation_failure_preserves_last_successful_completion(
    executor, message
):
    result = executor.execute(
        "done('first')\ntry:\n    done(message)\nexcept (AssertionError, ValueError):\n    pass",
        {"message": message},
    )
    assert result["exception"] is None
    assert result["done"] == "first"


def test_exception_suppresses_completion_without_rolling_back_state(executor):
    namespace = {"values": []}
    result = executor.execute(
        "values.append(1)\ndone('first')\ndone('last')\nraise RuntimeError('failed')",
        namespace,
    )
    assert "RuntimeError: failed" in result["exception"]
    assert result["done"] is None
    assert namespace["values"] == [1]
    assert executor.execute("pass", namespace)["done"] is None


def test_completion_retains_string_subclass_identity(executor):
    class Message(str):
        pass

    message = Message("last")
    message.values = []
    result = executor.execute(
        "done('first')\ndone(message)\nmessage.values.append(1)", {"message": message}
    )
    assert result["exception"] is None
    assert result["done"] is message
    assert result["done"].values == [1]
