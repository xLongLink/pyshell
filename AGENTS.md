# Repository instructions

- AVOID extra comments that a human wouldn't add or is inconsistent with the rest of the file
- MINIMIZE comment to the bare minimun.
- DO NOT add content to markdown files, only fix what is outdated or wrong

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
