# Repository instructions

- AVOID extra comments that a human wouldn't add or is inconsistent with the rest of the file
- MINIMIZE comment to the bare minimun.


## Project layout

```bash
├── api/
│   ├── main.py        # FastAPI application and endpoints
│   ├── src/           # Execution API and persistent Python namespace
│   │   └── ui/        # One module per UI element
│   ├── tests/         # Server tests
│   ├── Dockerfile     # Execution container image
│   ├── Dockerfile.dockerignore
│   ├── pyproject.toml # API dependencies
│   ├── uv.lock
│   └── compose.yaml   # Execution environment configuration
├── web/
│   ├── main.py        # Browser-facing application entry point
│   ├── src/
│   │   ├── script.py  # Agent loop and terminal client
│   │   ├── server.py  # Agent API and shared workspace
│   │   └── ui/        # Self-contained React frontend, npm configuration, and tests
│   ├── tests/         # Agent and server tests
│   ├── pyproject.toml # Client dependencies
│   └── uv.lock
├── README.md          # Setup and usage
└── CONTRIBUTING.md    # Development and operational details
```

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
