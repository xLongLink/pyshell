<div align="center">

# Python agent shell

Give instructions. Watch the agent write and run Python inside Docker.

</div>

<br />

## What it does

- ⚡ **Remote execution**: Generated Python runs through a local API, with code and output shown in the terminal.
- 🧠 **Persistent sessions**: Variables, imports, and conversation context carry over between instructions.
- 🔄 **Execution feedback**: Output and errors go back to the model so it can continue or correct mistakes.
- 🛠️ **Custom environment**: Install Python libraries at container startup; the agent imports what it needs.

<br />

## Getting started

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), Make, Docker,
Docker Compose v2.24+, and an OpenAI API key.

Run from the repository root:

```bash
cp .env.sample .env
# Edit .env and set OPENAI_API_KEY.
make up
```

`make up` builds and starts the execution API at `http://127.0.0.1:8000`, waits
until it is healthy, then runs `script.py` to open the local `pyshell>` prompt and communicate
with OpenAI and the API. Use `/exit` to quit the client or `/reset` to clear the
server's Python namespace and the client's conversation. The API keeps running
after the terminal client exits. Use `/reset` when reconnecting for a fresh session.

Stop project containers without deleting workspace data:

```bash
make down
```

<br />

## How it works

The local terminal client streams Python from the model, then sends the completed
code to the container's FastAPI server. Output and errors return as JSON and are
printed locally and sent back to the model. Variables and imports remain available
between requests. No generated Python executes on the host.

```text
pyshell> calculate the sum of numbers 1 through 100

total = sum(range(1, 101))
print(total)
done()
5050
```

The agent continues until it calls `done()`. Each continuation makes another model
request; there is no step limit. Ctrl-C stops the local client request, not Python
already running in the container. Restart the container to stop runaway execution.

### Execution API

`POST /` accepts Python source, not natural-language instructions:

```bash
curl http://127.0.0.1:8000/ -H 'Content-Type: application/json' \
  -d '{"code":"value = 42\nprint(value)"}'
```

```json
{"stdout":"42\n","stderr":"","exception":null,"done":false}
```

`POST /reset` clears the namespace; `GET /` reports health. One session is shared
by all requests to the container, and execution is serialized. Code runs as a
normal Python script: use `print()` for results. Use one server worker only.

<br />

## Customize the environment

Configure your API key, model, and processing tier in [`.env`](.env.sample).
`PYSHELL_API_URL` selects the execution server for the terminal client.
The selected model must be available to your OpenAI account; Priority processing
may cost more. Interactive startup makes a small paid warm-up request.

Optionally pass requirements as an environment variable, one package per line
(Bash syntax):

```bash
PYSHELL_REQUIREMENTS=$'pandas==2.3.3\nnumpy\nrequests' make up
```

Or pass the contents of an existing file without mounting it:

```bash
PYSHELL_REQUIREMENTS="$(cat requirements.txt)" make up
```

Packages are installed at container startup—not baked into the image.
Without `PYSHELL_REQUIREMENTS`, `make up` installs no extra packages.
The agent imports the modules it needs; imports persist until `/reset`.

Restart with the same option after changing requirements. Installation needs network
access and stops startup on failure. Packages are reinstalled in each new container.
Source changes still require an image rebuild.

With a published image:

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  -e PYSHELL_REQUIREMENTS=$'pandas==2.3.3\nnumpy\nrequests' \
  ghcr.io/xlonglink/pyshell:v0.1.1
```

An unset or empty variable starts the image with only PyShell's dependencies.

<br />

## Workspace and isolation

The execution API runs as non-root inside Docker, without host file, directory,
or Docker socket mounts.
Files in `/workspace` persist across runs; Python state does not.
Reset does not undo file changes or stop subprocesses.

> [!WARNING]
> The API executes arbitrary Python without authentication. Keep the published port
> bound to localhost and use only trusted local clients. Do not expose it publicly.
> Generated code can access the container's workspace and network. Docker is not a
> secure sandbox for hostile code. OpenAI credentials stay in the local terminal
> client, not the API container. Executed code and captured output are sent to OpenAI.

**`docker compose down --volumes` deletes workspace data.**

<br />

## Contribute to pyshell

See [Contributing](CONTRIBUTING.md) for development, testing, and releases.

<br />
<br />


---

<div align="center">
LongLink 2026

[License](./LICENSE) &nbsp; - &nbsp; [Contributing](./CONTRIBUTING.md) &nbsp; - &nbsp; [Code of Conduct](./CODE_OF_CONDUCT.md) &nbsp; - &nbsp; [Contact](mailto:info@longlink.dev)

</div>

---
