<div align="center">

# Python agent shell

Give instructions. Watch the agent write and run Python inside Docker.

</div>

<br />

## What it does

- ⚡ **Immediate execution**: Generated Python runs as it streams, with code and output shown in the terminal.
- 🧠 **Persistent sessions**: Variables, imports, and conversation context carry over between instructions.
- 🔄 **Execution feedback**: Output and errors go back to the model so it can continue or correct mistakes.
- 🛠️ **Custom environment**: Add Python libraries to your image; the agent imports what it needs.

<br />

## Getting started

Requirements: Make, Docker, Docker Compose v2.24+, and an OpenAI API key. No host Python installation is needed.

Run from the repository root:

```bash
cp .env.sample .env
# Edit .env and set OPENAI_API_KEY.
make up
```

`make up` builds the image and opens the `pyshell>` prompt. Enter an instruction
to start working. Use `/exit` to quit or `/reset` to start a fresh session.

Stop project containers without deleting workspace data:

```bash
make down
```

<br />

## How it works

The agent generates Python, runs it in a persistent console, and receives the
output and errors before continuing. Variables and imports remain available
between instructions.

```text
pyshell> calculate the sum of numbers 1 through 100

>>> total = sum(range(1, 101))
>>> print(total)
5050
>>> done()
```

Execution continues until the agent calls `done()` or you interrupt it with Ctrl-C.
Each continuation makes another API request; there is no step limit.

<br />

## Customize the environment

Configure your API key, model, and processing tier in [`.env`](.env.sample).
The selected model must be available to your OpenAI account; Priority processing
may cost more. Interactive startup makes a small paid warm-up request.

Add packages to [`requirements.txt`](requirements.txt). Compose mounts this file
read-only, and packages are installed at container startup—not baked into the image.
The agent imports the modules it needs; imports persist until `/reset`.

Restart with `make up` after changing requirements. Installation needs network
access and stops startup on failure. Packages are reinstalled in each new container.
Source changes still require an image rebuild.

With a published image, supply your own requirements file:

```bash
docker run --rm -it --env-file .env \
  --mount type=bind,src="$(pwd)/requirements.txt",dst=/opt/pyshell/requirements.txt,readonly \
  ghcr.io/xlonglink/pyshell:v0.1.1
```

Without a requirements mount, the image starts with only PyShell's dependencies.

<br />

## Workspace and isolation

The application runs as non-root inside Docker. Only the requirements file is
mounted from the host, read-only; no host directories or Docker socket are mounted.
Files in `/workspace` persist across runs; Python state does not.
Reset does not undo file changes or stop subprocesses.

> [!WARNING]
> Generated code can access the container's workspace, network, and API key.
> Docker is not a secure sandbox for hostile code. Use trusted instructions and
> packages; executed code and captured output are sent to OpenAI.

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
