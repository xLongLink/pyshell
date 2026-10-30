# Contributing to pyshell

pyshell runs model-generated Python in a persistent execution API. `api/main.py` owns
FastAPI endpoints; `api/src/` owns Python execution. Docker configuration lives in `api/`.
`web/main.py` exposes the browser-facing app from `web/src/server.py`, which owns
the agent API and one shared workspace. `web/src/script.py` contains the reusable
agent loop and original terminal client. `web/src/ui/` is the self-contained React/shadcn
frontend project, including npm dependencies, build configuration, and frontend tests.
Vite proxies `/ws` to the agent API.
`api/src/ui/widget.py` defines the shared widget methods. The figure model
in `api/src/ui/plot/figure.py` inherits that base; `api/src/ui/session.py` owns its factory and
uses `a2ui-core` for catalog validation and protocol messages.
`web/src/ui/lib/a2ui.ts` validates incoming display messages, and
`web/src/ui/components/catalog.tsx` implements the approved React component catalog.
Plot models live in `api/src/ui/plot/`: a figure contains discriminated scatter,
bar, or pie traces and a typed layout. `ui.plot(data, layout=...)` returns the
figure handle; explicit `update()` replaces supplied top-level fields atomically.
`web/src/ui/components/plot/` renders Cartesian traces together or a single pie.
The renderers compose Recharts with the installed shadcn chart primitives in
`web/src/ui/components/ui/chart.tsx` for responsive sizing, tooltips, and legends.

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

Open `http://127.0.0.1:8010`. `make up` builds and starts the container, installs
frontend dependencies, builds the website, and starts the foreground web server
and agent API. For frontend development, run `npm --prefix web/src/ui run dev`
in another terminal and open
`http://127.0.0.1:5180`. The execution endpoint is hardcoded to
`http://127.0.0.1:8000`. Browser tabs share one conversation and namespace.
Use Reset for a fresh session. Closing a tab does not cancel submitted work.
Press Ctrl-C to stop the agent API; the container runs until `make down`.

Run `npm --prefix web/src/ui run typecheck` and `npm --prefix web/src/ui run build` to check the frontend.
`npm --prefix web/src/ui run format` formats frontend sources.

To add a trace type, define its strict model in `api/src/ui/plot/` and add it to
the discriminated trace union and compatibility checks in `figure.py`. Each
surface holds one `Figure`; no new factory or catalog entry is needed for a trace.
Mirror its schema and figure validation in `web/src/ui/lib/a2ui.ts`, then extend
the renderer in `web/src/ui/components/plot/`. Update the agent instructions in
`web/src/script.py`. Shared contract cases in `api/tests/fixtures/figures.json`
run through both Python and frontend validation; add rendering tests as well.
Keep catalog changes backward compatible or version the catalog ID on both sides.
Rebuild the execution container and frontend with `make up` after changes.

Widget content uses A2UI; pin/focus commands use separate workspace events.
The agent API owns placement state and broadcasts it to all tabs. Placement
records contain only `pinned`; unpinned widgets occupy the center, with at most
four centered at once. Placement events are broadcast live but not retained in
history: reconnect snapshots carry the current placements separately. Deploy
the frontend and backend together when changing this contract. Content updates
preserve placement, while reset and surface deletion clear it.
The browser processes appended A2UI messages incrementally and rebuilds its
processor when a reconnect snapshot replaces the history or the workspace resets.

Cleanup:

```bash
make down
```

## Dependencies

See [api/pyproject.toml](api/pyproject.toml) for default libraries and version constraints,
and [api/uv.lock](api/uv.lock) for exact locked versions. Plotting uses the provided
`ui` charts rather than a separate plotting library.

`make up` defaults `PYSHELL_REQUIREMENTS` to `swissparlpy` as an extra-library showcase.
Override `PYSHELL_REQUIREMENTS` with newline-separated requirements, or pass file contents
with `PYSHELL_REQUIREMENTS="$(cat requirements.txt)" make up`. No file is mounted.
At startup, the Dockerfile entrypoint writes the value to a temporary requirements
file, copies the base environment into writable temporary storage, installs the
requirements, checks compatibility, and launches the API. Installation failure
prevents launch. Use `PYSHELL_REQUIREMENTS='' make up` to skip installation of extras;
direct Compose invocations also skip extras when the variable is unset.

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

Tests cover the execution API under `api/tests/` and the agent/client under `web/tests/`.
API tests use an in-process FastAPI client; model requests are mocked. No live
OpenAI requests or Docker containers are needed for the unit tests.

Frontend tests in `web/src/ui/tests/` include real A2UI rendering under jsdom,
without browser automation.

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
