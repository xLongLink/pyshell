# Agents Instructions

- PyShell provides a persistent Python workspace that AI agents can inspect and control by executing Python code.
- The repository includes a FastAPI execution backend, a widget frontend, and a sample browser/voice application.
- The model’s (called workder) job is to understand a request, inspect live objects, execute Python, and return the correct result.

## Repository instructions

- AVOID extra comments that a human wouldn't add or is inconsistent with the rest of the file
- MINIMIZE comment to the bare minimun.
- DO NOT add content to markdown files, only fix what is outdated or wrong. DO NOT add tests for the sample application.

## Project layout

```bash
├── src/
│   ├── api/           # Execution backend project
│   │   ├── src/api/   # FastAPI endpoints, Python namespace, and widget library
│   │   ├── tests/     # Server tests
│   │   ├── Dockerfile # Execution container image
│   │   ├── pyproject.toml # API dependencies
│   │   └── uv.lock
│   ├── web/           # Container-owned widget frontend and tests
│   ├── .dockerignore
│   └── compose.yaml   # Execution environment configuration
├── sample/
│   ├── main.py        # Browser-facing FastAPI app and shared workspace
│   ├── assets/        # index.html and browser voice signaling
│   ├── src/
│   │   ├── agents/    # Execution agent loop and voice assistant
│   │   ├── shell.py   # Execution container client
│   │   └── settings.py # Shared application settings
│   ├── pyproject.toml # Client dependencies
│   └── uv.lock
├── README.md          # Setup and usage
└── CONTRIBUTING.md    # Development and operational details
```

## PRINCIPLES 

- KISS: Choose the simplest correct implementation.
- DRY: Avoid duplicated knowledge or business rules. AVOID abstractions that only remove superficial code similarity.
- YAGNI: Avoid features, abstractions, configuration, and extension points needed only for hypothetical future use.
- SRP / Separation of concerns: Keep responsibilities focused and ownership clear.
- High cohesion / Low coupling: Keep related behavior together and reduce unnecessary dependencies.

## General rules

- DO implement only the requested feature and changes strictly necessary for it. Do not add convenience commands (such as Makefile targets), tooling, configuration, or unrelated cleanup unless explicitly requested.
- PREFER simple, maintainable, conventional solutions.
- When they simplify implementation, PREFER the standard library or established libraries over handwritten implementations.
- DO validate inputs at system boundaries.
- DO test actual behavior instead of duplicating production logic in tests.
- When duplication improves clarity, PREFER explicit duplication to helpers or shared code.
- Where practical, PREFER real implementations and explicit dependency boundaries to mocks or global runtime-state changes.
- NEVER EVER EVER add any new comment in markdown files, only fix existing one.
- NEVER create new markdown files unless indicated from the user.
- ALWAYS run the cleanup skill at the end of the implementation

## Python Guidelines

- NEVER add module-level docstrings; start Python files with imports instead of descriptive triple-quoted headers.
- AVOID renaming imports.
- DO use clear domain names.
- PREFER single-word Python filenames.
- AVOID unnecessary try/except blocks.
- DO use exceptions for genuine errors.
- DO validate types at system boundaries.
- DO group Pydantic fields into commented sections.
- DO order Pydantic fields within each section from the shortest name to the longest name.
- PREFER the simplest correct implementation. 
- AVOID features needed only for hypothetical future use.
- DO construct class instances in separate, multi-line assignments before invoking their methods.
- PREFER idiomatic Python and readability over efficiency.
- DO construct class instances in separate, multi-line assignments before invoking their methods.
- DO use Protocol for behavioral interfaces and dependency contracts.
- DO represent application state with typed models, enums, or structured objects.
- PREFER namespaced module APIs to importing many related functions directly.
- DO store asynchronous query results in a named variable before calling .all(), .one_or_none(), or similar result methods.
- DO use collections.abc.Sequence instead of list for read-only query result return types.
- DO declare response_model on FastAPI routes and let FastAPI validate responses.
- Within each section, DO order fields from the shortest name to the longest name.
- DO add a descriptive # ... comment before each logic block, with one blank line before the comment.
- PREFER explicit duplication to a local helper, When duplication makes lifecycle code clearer.

## Commit Message Structure

DO use this structure for commit messages:

```
<commit-message>
	<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert</type>
	<scope>api|sdk|web (optional)</scope>
	<description>A short, imperative summary of the change</description>
</commit-message>
```

## Merge Request Structure

DO use this structure for merge requests:

```
<merge-request>
	<title>Use the commit message structure</title>
	<summary>What changed and why</summary>
	<validation>Checks run and results, or why not run</validation>
	<labels>Change-type labels from .github/release.yml and applicable area labels: api|sdk|web</labels>
</merge-request>
```
