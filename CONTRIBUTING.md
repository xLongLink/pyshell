# Contributing to pyshell

pyshell runs model-generated Python in a persistent execution API. `api/src/api/` owns
FastAPI endpoints and Python execution; Docker configuration lives in `api/`.
`web/script.py` contains the local terminal UI,
model requests, conversation state, HTTP calls, and printing.

<br />

## Development

`api/` and `web/` are independent uv projects, each with its own `pyproject.toml`,
`uv.lock`, and `.venv`. Sync them with `uv sync --project api` and
`uv sync --project web`. Each project configures its own pytest settings and
keeps the same lint and formatting rules in its own `pyproject.toml`.
Client tests install the API as an editable development dependency.

Run from the repository root with Make, Docker, and Docker Compose v2.24+:

```bash
make up
```

Press Ctrl-C to exit the terminal client. The API keeps running until `make down`.
Use `/reset` for a fresh session; `/exit` and `/quit` are not local commands.

Cleanup:

```bash
make down
```

## Dependencies

Set `PYSHELL_REQUIREMENTS` to newline-separated requirements, or pass file contents
with `PYSHELL_REQUIREMENTS="$(cat requirements.txt)" make up`. No file is mounted.
At startup, the Dockerfile entrypoint writes the value to a temporary requirements
file, copies the base environment into writable temporary storage, installs the
requirements, checks compatibility, and launches the API. Installation failure
prevents launch; an unset or empty variable skips installation.

User packages are not baked into the image and are reinstalled on each container
start. Requirements are separate from `api/uv.lock`; pin versions where possible.
Prefer compatible wheels; system dependencies still need a custom image. Compose
provides 512 MiB of executable temporary storage so native extensions can load.
Larger environments need more temporary storage and memory.

The agent imports modules as needed. Imports and variables persist between
instructions; `/reset` clears the execution namespace and conversation. Reset
does not undo file writes, imported-module state, network activity, or subprocesses.
Use trusted packages: installation can execute code with access to the container's
environment, network, and workspace. Avoid secrets in dependency URLs.

## Workspace

Compose uses a Docker-managed volume at `/workspace`, with no host directory or
Docker socket mounts. Copy a result to the host explicitly:

```bash
docker compose -f api/compose.yaml run --rm --no-deps --entrypoint cat -T pyshell /workspace/result.txt > result.txt
```

`make down` preserves files. **`docker compose -f api/compose.yaml down --volumes` deletes
workspace data.** Do not add privileged mode or broad host mounts to work around
container restrictions.

Interactive programs and `input()` are unsupported. Capture subprocess output
explicitly and print it if the agent needs to inspect it.

## Testing

```bash
make test
```

Tests cover the API under `api/tests/` and the local client under `web/tests/`.
API tests use an in-process FastAPI client; model requests are mocked. No live
OpenAI requests or Docker containers are needed for the unit tests.

The API must run with one worker because the Python namespace is in memory.
Execution and reset are serialized. Interrupting the host client does not cancel
server execution; restart the container if code hangs. `/reset` waits for active
execution and does not reset the working directory or filesystem.

## Release

```bash
git fetch origin main
git tag vX.Y.Z origin/main
git push origin vX.Y.Z
```
