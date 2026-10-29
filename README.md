# Python agent shell

A terminal REPL: give the agent an instruction, it generates only Python, and
each complete line runs immediately. Stdout, stderr, and tracebacks are sent back to the
agent so it can inspect results and continue. Variables and imports persist.
Generated code streams into a persistent Python interactive console with `>>>`
and `...` prompts. Expressions display their values automatically. Multiline
statements need a blank line to finish, just like the normal Python REPL.
The final line can omit its newline when the response completes successfully.
If streaming fails or is interrupted, already executed lines remain in effect;
unfinished input is discarded. Output goes back to the model in the next API
request, not while the current response is still streaming.
After each response, timing is printed to stderr: time to the first nonempty code
delta and total elapsed time, including local Python execution. Failed or
interrupted responses also print timing; first-token time is `n/a` if no code arrived.

Requires Docker and an OpenAI account with access to the chosen model.

## Run the container

Pull a released image (requires GitHub access to the private package):

```bash
# Authenticate with a GitHub personal access token with read:packages access.
gh auth token | docker login ghcr.io -u "$(gh api user --jq .login)" --password-stdin
docker run --rm -it --env-file .env ghcr.io/xlonglink/pyshell:v0.1.0
```

The token must have `read:packages`; the default `gh` login scopes may not include it.
Configure `.env` from `.env.sample` before running. Alternatively, build locally:

```bash
docker build -t pyshell .
cp .env.sample .env
# Edit .env and replace the placeholder API key before running.
docker run --rm -it --env-file .env pyshell
```

The container opens the `pyshell>` prompt directly. Enter instructions, then use
`/exit` or Ctrl-D to quit. Keep `-it` for interactive use. No ports or server are needed.
The API key is supplied at runtime; `.env`, local virtual environments, and build
artifacts are excluded from the image build context.

The agent works in `/workspace` inside the container. Files disappear when the
container is removed unless you mount a directory. To work on files in the current
host directory and keep changes:

```bash
docker run --rm -it --env-file .env \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$(pwd),target=/workspace" pyshell
```

The image runs as a non-root user (UID 1000) by default. The `--user` option above
matches your host user's file permissions on Linux. Mount only directories the
agent should access; it can modify or delete mounted files. Do not mount the
Docker socket. Python state lasts only for the current process, even with a mount.

Run a single instruction by passing it after the image name:

```bash
docker run --rm --env-file .env pyshell "List files in the current directory"
```

Settings also load from `.env` in the working directory. Environment variables override
`.env` values. `.env.sample` lists `OPENAI_API_KEY`, `OPENAI_MODEL`, and
`OPENAI_SERVICE_TIER`. Set the tier to `priority` for faster Priority processing
(account/model support required; higher pricing may apply), or `auto` for the
normal default behavior. Priority is requested, not a latency guarantee.
Requests use `reasoning.effort="low"`, the lowest supported effort for GPT-6.1 Sol.
This model does not support disabling reasoning with `none`.
Priority processing is used for speed; no separate "ultrafast" API option is configured.
Interactive startup makes a small paid warm-up request using the same API client
as subsequent requests, allowing HTTP connection reuse before the first prompt.
Warm-up output is never executed or added to the conversation. This does not
reserve a model instance or guarantee lower latency. Single-instruction commands
skip warm-up to avoid an extra request. Warm-up failure is reported but does not
prevent using the prompt.
The API key is masked in settings representations.
The container is the distribution format; no host Python installation is needed.

## Releases

Push a `vX.Y.Z` tag matching `project.version` in `pyproject.toml` on a commit
on `main`. Update the version and refresh `uv.lock` with `uv lock` before tagging.
The release workflow validates the tag, runs the tests and container smoke checks,
publishes `ghcr.io/xlonglink/pyshell:vX.Y.Z` and `:latest` (Linux amd64), then
creates a GitHub release with generated notes. The GHCR package is private by
default; keep its visibility private in the organization's package settings.

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Development

For local development without Docker, install Python 3.10+ and uv:

```bash
uv sync
# If you haven't configured .env yet, copy .env.sample to .env and edit it.
uv run pyshell
```

Example session (generated code can vary):

```text
pyshell> print hello world

>>> print("hello world")
hello world
>>> done()

pyshell> calculate the sum of numbers 1 through 100

>>> total = sum(range(1, 101))
>>> print(total)
5050
>>> done()
```

Use `/exit` to quit or `/reset` to clear the Python namespace and conversation.
Reset does not undo filesystem changes or terminate subprocesses. Ctrl-C
interrupts ordinary Python execution; it cannot reliably stop native code or
every subprocess. Output is shown live for Python writes to stdout/stderr.
Subprocess output must be captured explicitly (for example, with
`subprocess.run(..., capture_output=True, text=True)` and `print(result.stdout)`)
to feed it back to the agent. Interactive programs and `input()` are not supported.

Single instruction:

```bash
uv run pyshell "List files in the current directory"
```

The agent continues until it calls `done()` or you interrupt it with Ctrl-C;
there is no step limit. Each continuation makes another API request. The Responses API
maintains conversation context using response IDs. A final execution's result
is included with the next user turn.

**Executes generated code:** in the container, code can access its filesystem,
network, environment (including API keys), and any mounted host files. A container
limits host access but is not a secure sandbox for untrusted code. When running
locally, code has your account's permissions. Only run trusted instructions in a
suitable environment. Executed code and captured output go to OpenAI; avoid
secrets in both. Your local `.env` is ignored by Git; never share it. Rotate any key pasted
into a conversation. Model availability depends on your account; an unavailable
model returns an API error rather than silently switching models.

Tests (no API calls):

```bash
uv run python -m unittest -v
```
