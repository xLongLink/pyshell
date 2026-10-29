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
docker run --rm -it --env-file .env \
  -e PYSHELL_REQUIREMENTS=$'pandas==2.3.3\nnumpy\nrequests' \
  ghcr.io/xlonglink/pyshell:v0.1.1
```

An unset or empty variable starts the image with only PyShell's dependencies.

<br />

## Workspace and isolation

The application runs as non-root inside Docker, without host file, directory,
or Docker socket mounts.
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
