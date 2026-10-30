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
- 🎙️ **Voice and text:** Speak to the voice assistant or submit text instructions. Both use the same persistent workspace.


<br />

## Getting started


```bash
docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/xlonglink/pyshell:latest
```
See [api/pyproject.toml](api/pyproject.toml) for installed libraries and version constraints.

<br />

## How it works

<img src="architecture.png" alt="PyShell architecture" />


The browser connects to the web server over `/ws`. Text instructions and voice-delegated tasks share one execution agent, Python namespace, and widget layout. Only one task runs at a time; closing a tab or ending a call does not cancel submitted work. Reconnecting restores the workspace without resubmitting it.

Generated code executes only after the model response completes successfully. The Docker execution API streams widget updates and final execution results over `/execute/stream`; the web server forwards widget updates, not execution output, to connected tabs. Execution feedback goes back to the agent until it calls `done()` or reaches its task budget.

Voice uses browser microphone audio over WebRTC and server-side OpenAI Realtime signaling. Allow microphone access on localhost or HTTPS. Provider credentials stay on the server. Voice conversation lasts for the call; Python state and the execution agent's history persist across calls and text turns.

The voice assistant can check the current task and execution phase, inspect chart titles and types, and recall the last eight task summaries across calls. Ask “what are you doing?” or “change the sales chart to a bar chart.” Say “end the call” to hang up without cancelling submitted work. Steering or stopping running Python is not supported.

Configure `OPENAI_API_KEY` in `web/.env`. Optional settings are `OPENAI_MODEL` and `OPENAI_REALTIME_MODEL` (default: `gpt-realtime`). Voice requires access to the realtime model. The web server requires Python 3.11 or newer.

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
