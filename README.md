<div align="center">

<img src="banner.png" alt="PyShell banner" />

<h3 >

Python in the shell

</h3 >

Give instructions in your browser. Watch the agent write and run Python inside Docker.

</div>


<br />

## What it does

- 🧠 **Persistent sessions:** Variables, imports, and conversation context carry over between instructions.
- 🔄 **Execution feedback:** Output and errors go back to the model so it can continue or correct mistakes.
- 📊 **Rich results:** A2UI renders Plotly-style figures with lines, markers, bars, and pies alongside code and printed output.
- 🎙️ **Voice instructions:** Speak to the voice assistant to work in the persistent workspace.


<br />

## Getting started


Configure `MODEL` and its provider's API key, plus `OPENAI_API_KEY` for voice, in `sample/.env`, then run from the repository root:

```bash
make up
```
Open http://127.0.0.1:8010. The execution container serves the widget interface on port 8000; the application embeds it in a separate-origin iframe. See [src/api/pyproject.toml](src/api/pyproject.toml) for installed libraries and version constraints.

<br />

## How it works

<img src="architecture.png" alt="PyShell architecture" />


The browser connects to the web server over `/ws` for task events and voice signaling. A sandboxed iframe loads the widget frontend from the execution container and receives current widget snapshots directly over `/ui/ws`. Voice-delegated tasks share one execution agent, Python namespace, and container-owned widget layout. Only one task runs at a time; closing a tab or ending a call does not cancel submitted work. Reconnecting restores the widgets without resubmitting work.

Generated code executes only after the model response completes successfully. The Docker execution API returns a result and terminal confirmation over `/execute/stream`. The embedded frontend receives container-owned widget snapshots independently over `/ui/ws`; widget events are not included in the execution stream. Execution feedback goes back to the agent until it calls `done()` or reaches its task budget.

The sample loads an HTML document with a voice orb and the container-owned workspace iframe using Python; it needs no frontend build. Voice uses a small browser JavaScript module for microphone audio over WebRTC and server-side OpenAI Realtime signaling. Allow microphone access on localhost or HTTPS. Provider credentials stay on the server. Voice conversation lasts for the call; Python state and the execution agent's history persist across calls.

The voice assistant can check the current task and execution phase, inspect chart titles and types, and recall the last eight task summaries across calls. Ask “what are you doing?” or “change the sales chart to a bar chart.” Say “end the call” to hang up without cancelling submitted work. Steering or stopping running Python is not supported.

Configure `MODEL` with a Pydantic AI provider-qualified name (default: `openai:gpt-6.1-sol`) and the provider's standard credentials in `sample/.env`, for example `MODEL=anthropic:claude-sonnet-4-5` with `ANTHROPIC_API_KEY`, or `MODEL=google:gemini-2.5-pro` with `GOOGLE_API_KEY`. OpenAI, Anthropic, and Google dependencies are included; other providers may require their Pydantic AI dependency extra. Credentials stay outside the execution container; existing environment variables take precedence over the file. Voice remains on OpenAI and requires `OPENAI_API_KEY` and access to `OPENAI_REALTIME_MODEL` (default: `gpt-realtime`). The web server requires Python 3.11 or newer. This configuration is localhost-only, not an authenticated multi-user deployment.

Execution stream failures are not retried: code may already have run. Restart both the execution container and web server before submitting more work after an unconfirmed execution. A transport timeout does not terminate Python; restarting the container does. Listener queues are bounded; reconnect snapshots contain current widgets, not event history.

<br />


## Development

Run from the repository root:

```bash
make up         # Setup the local server
make down       # Stop and remove the containers
make test       # Testing
make format     # Format the code
make logs       # Show the container logs
```

<br />

---

<div align="center">

LongLink 2026

[MIT License](./LICENSE) &nbsp; - &nbsp; [Contributing](./CONTRIBUTING.md) &nbsp; - &nbsp; [Code of Conduct](./CODE_OF_CONDUCT.md) &nbsp; - &nbsp; [Contact](mailto:info@longlink.dev)

</div>
