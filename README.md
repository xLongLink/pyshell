<div align="center">

<img src="banner.png" alt="PyShell banner" />

# Python agent shell

Give instructions. Watch the agent write and run Python inside Docker.

</div>

## What it does

- **Persistent sessions:** Variables, imports, and conversation context carry over between instructions.
- **Execution feedback:** Output and errors go back to the model so it can continue or correct mistakes.
- **Custom environment:** Install Python libraries at container startup.

## Getting started

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), Make, Docker,
Docker Compose v2.24+, and an OpenAI API key.

From the repository root:

```bash
cp web/.env.sample web/.env
# Edit web/.env and set OPENAI_API_KEY.
make up
```

Submit instructions in the terminal. Press Ctrl-C to exit the client; stop the
API with `make down`. Workspace files are preserved.

Configure the model, processing tier, and execution server URL in
[`web/.env`](web/.env.sample). The model must be available to your OpenAI account;
Priority processing may cost more.

## How it works

The terminal client sends instructions to the model and runs generated Python
through a local execution API. Code and output appear in the terminal.

`POST /` accepts Python source, not natural-language instructions:

```bash
curl http://127.0.0.1:8000/ -H 'Content-Type: application/json' \
  -d '{"code":"value = 42\nprint(value)"}'
```

```json
{"stdout":"42\n","stderr":"","exception":null,"done":false}
```

Use `print()` for results. `GET /` reports health; `POST /reset` clears the Python
namespace. The terminal's `/reset` command also clears the conversation.

All requests share one session, and execution is serialized. Run only one server
worker. Reset does not undo file writes or other side effects. Keep the execution
API local: it runs arbitrary Python without authentication.

## Customize the environment

Pass extra packages as newline-separated requirements (Bash syntax):

```bash
PYSHELL_REQUIREMENTS=$'pandas==2.3.3\nnumpy\nrequests' make up
```

Without `PYSHELL_REQUIREMENTS`, no extra packages are installed. Packages are
installed at container startup; installation failure prevents launch.

See [Contributing](CONTRIBUTING.md#dependencies) for dependency and container details.

## Contributing

Run tests with `make test`. See [CONTRIBUTING.md](CONTRIBUTING.md) for development,
workspace access, and release instructions.

---

<div align="center">

LongLink 2026

[MIT License](./LICENSE) &nbsp; - &nbsp; [Contributing](./CONTRIBUTING.md) &nbsp; - &nbsp; [Code of Conduct](./CODE_OF_CONDUCT.md) &nbsp; - &nbsp; [Contact](mailto:info@longlink.dev)

</div>
