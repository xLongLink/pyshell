# Contributing to pyshell

pyshell runs model-generated Python in a persistent execution API. `api/` owns
FastAPI endpoints and Python execution. `script.py` contains the local terminal UI,
model requests, conversation state, HTTP calls, and printing.

<br />

## Development

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
start. Requirements are separate from `uv.lock`; pin versions where possible.
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
docker compose run --rm --no-deps --entrypoint cat -T pyshell /workspace/result.txt > result.txt
```

`docker compose down` preserves files. **`docker compose down --volumes` deletes
workspace data.** Do not add privileged mode or broad host mounts to work around
container restrictions.

Interactive programs and `input()` are unsupported. Capture subprocess output
explicitly and print it if the agent needs to inspect it.

## Testing

```bash
uv run --locked pytest -v
```

Tests cover the API under `tests/api/` and the local client under `tests/client/`.
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
