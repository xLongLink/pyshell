import ast
from collections.abc import Sequence


def score(rows: Sequence[dict]) -> dict:
    """Score task outcomes and Python syntax across all generated responses."""
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Evaluation must contain unique, nonempty tasks")
    statuses = {"passed", "incorrect", "call_limit", "boundary_error"}
    if any(row["status"] not in statuses for row in rows):
        raise ValueError("Unknown evaluation status")
    generations = 0
    valid_python = 0
    for row in rows:
        for step in row.get("steps", []):
            source = step["source"]
            if not isinstance(source, str):
                raise ValueError("Generated source must be text")
            generations += 1
            try:
                tree = ast.parse(source)
                compile(tree, "<benchmark>", "exec")
            except (SyntaxError, ValueError):
                continue
            valid_python += bool(tree.body)
    passed = sum(row["status"] == "passed" for row in rows)
    return {
        "passed": passed,
        "total": len(rows),
        "generations": generations,
        "success_rate": passed / len(rows),
        "valid_python": valid_python,
        "valid_python_rate": valid_python / generations if generations else None,
    }


def summary(result: dict) -> str:
    """Format both aggregate scores with their distinct denominators."""
    python = "not measured (0 responses)"
    if result["valid_python_rate"] is not None:
        python = f"{result['valid_python_rate']:.2%} ({result['valid_python']}/{result['generations']} responses)"
    return f"Task success: {result['success_rate']:.2%} ({result['passed']}/{result['total']} tasks)\nValid Python: {python}"
