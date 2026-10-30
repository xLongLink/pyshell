<div align="center">

<img src="banner.png" alt="PyShell banner" />

# Python agent shell

Give instructions in your browser. Watch the agent write and run Python inside Docker.

</div>

## What it does

- **Persistent sessions:** Variables, imports, and conversation context carry over between instructions.
- **Execution feedback:** Output and errors go back to the model so it can continue or correct mistakes.
- **Custom environment:** Install Python libraries at container startup.

## Getting started

Requirements: Python 3.10+, Node.js 22.18+, npm, [uv](https://docs.astral.sh/uv/), Make, Docker,
Docker Compose v2.24+, and an OpenAI API key.

From the repository root:

```bash
cp web/.env.sample web/.env
# Edit web/.env and set OPENAI_API_KEY.
make up
```

`make up` builds and starts the container, builds the website, and starts the
web server. Open `http://127.0.0.1:8010`.
Press Ctrl-C to stop the web server and agent API; stop the container with
`make down`. Workspace files are preserved.

Configure the API key, model, and processing tier in
[`web/.env`](web/.env.sample). The model must be available to your OpenAI account;
Priority processing may cost more.

## How it works

The web server serves the React/shadcn website and agent WebSocket on port 8010.
The agent API keeps model credentials server-side and runs generated Python in
the single container at the hardcoded endpoint `http://127.0.0.1:8000`.
Code and execution feedback stream to the website. There is no voice agent.

All tabs share one conversation, Python namespace, and transcript. Reconnecting
restores the transcript without resubmitting instructions. Closing a tab does
not stop submitted work. Restarting the agent API loses its conversation and
transcript; use Reset before starting fresh against an existing container.

`POST /` accepts Python source, not natural-language instructions:

```bash
curl http://127.0.0.1:8000/ -H 'Content-Type: application/json' \
  -d '{"code":"value = 42\nprint(value)"}'
```

```json
{"stdout":"42\n","stderr":"","exception":null,"done":false}
```

Use `print()` for results. `GET /` reports health; `POST /reset` clears the Python
namespace. The website's Reset button also clears the conversation and transcript.

All requests share one session, and execution is serialized. Run only one server
worker for each API. Reset does not undo file writes or other side effects. Keep
both APIs on loopback: this local prototype has no authentication, and generated
Python runs without approval.

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

`make agent` starts the web server against an already running container, serving
the existing frontend build. `make build` type-checks and builds the website.
For frontend development, run `make web` separately and open `http://127.0.0.1:5180`.
The original terminal client
remains available with `uv run --locked --project web python web/script.py`;
do not use it concurrently with the browser agent, since the namespace is shared.

---

<div align="center">

LongLink 2026

[MIT License](./LICENSE) &nbsp; - &nbsp; [Contributing](./CONTRIBUTING.md) &nbsp; - &nbsp; [Code of Conduct](./CODE_OF_CONDUCT.md) &nbsp; - &nbsp; [Contact](mailto:info@longlink.dev)

</div>
