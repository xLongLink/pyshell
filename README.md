<div align="center">

<img src="https://raw.githubusercontent.com/xLongLink/pyshell/main/banner.png" alt="pyshell banner" />


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



<br />

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
may cost more. Model requests are made only when you submit an instruction.

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
  ghcr.io/xlonglink/pyshell:v0.0.0
```

An unset or empty variable starts the image with only PyShell's dependencies.

<br />


## Contribute to PyShell

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), Make, Docker,
Docker Compose v2.24+, and an OpenAI API key.

Run from the repository root:

```bash
cp .env.sample .env
# Edit .env and set OPENAI_API_KEY.
make up
```

Stop project containers without deleting workspace data:

```bash
make down
```


<br />



---

<div align="center">
LongLink 2026

[License](./LICENSE) &nbsp; - &nbsp; [Contributing](./CONTRIBUTING.md) &nbsp; - &nbsp; [Code of Conduct](./CODE_OF_CONDUCT.md) &nbsp; - &nbsp; [Contact](mailto:info@longlink.dev)

</div>

---
