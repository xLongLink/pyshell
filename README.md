<div align="center">

<img src="banner.png" alt="PyShell banner" />

<h3 >

An Agent and a Python Shell

</h3 >

[![Tests](https://github.com/xLongLink/pyshell/actions/workflows/tests.yml/badge.svg)](https://github.com/xLongLink/pyshell/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker required](https://img.shields.io/badge/Docker-required-2496ED?logo=docker&logoColor=white)](#getting-started)

</div>



<br />


## What it does

Traditional AI chat generates text. PyShell generates and executes Python code in a persistent runtime, using execution results to drive the conversation.

### Benefits

- 🧠 **Persistent sessions:** Variables, imports, and execution context carry over between tasks.
- 🔌 **No MCP servers needed:** Install Python libraries and let the agent discover and use their APIs directly.
- 🔄 **Execution feedback:** The agent observes output and errors, correcting mistakes and iterating automatically.
- 📊 **Interactive visualizations:** The agent creates and updates charts, graphs, and maps directly in your browser.
- 🎙️ **Voice-driven:** A voice agent understands your requests and delegates execution to a Python worker.
  
https://github.com/user-attachments/assets/3b46a5d7-b39b-40f7-8575-fd3f6f0c1d7a


<br />

## Model

<!-- model-stats:start -->

Accepted adapter: [LongLink/pyshell-worker](https://huggingface.co/LongLink/pyshell-worker/tree/4064ca4a5ebab1c97dd166133a74be4231771973).
Base model: `Qwen/Qwen3.5-0.8B`.

**Execution success: 15.00% (9/60 tasks).**

| Category | Passed | Success |
| --- | ---: | ---: |
| Filtered means | 0/5 | 0.00% |
| Grouped revenue | 0/5 | 0.00% |
| Inspect callable | 1/5 | 20.00% |
| Joined totals | 0/5 | 0.00% |
| Left join | 0/5 | 0.00% |
| Linear system | 2/5 | 40.00% |
| Missing input | 2/5 | 40.00% |
| Missing values | 0/5 | 0.00% |
| Monthly totals | 0/5 | 0.00% |
| Recover execution | 0/5 | 0.00% |
| Stateful followup | 2/5 | 40.00% |
| Statistics | 2/5 | 40.00% |

<!-- model-stats:end -->

<br />

## Try it locally

Set `OPENAI_API_KEY` in `sample/.env`, then run from the repository root:

```bash
make up
```

Open [http://127.0.0.1:8010](http://127.0.0.1:8010) in your browser.


<br />

## How it works

![PyShell architecture](architecture.png)

1. **Speak:** Give instructions to the voice agent in natural language.
2. **Delegate:** The voice agent understands your request and passes the task to a Python worker.
3. **Execute & iterate:** The worker discovers available Python libraries, writes and executes code, and uses output and errors to refine its actions.
4. **Visualize:** Results appear as interactive charts and visualizations directly in your browser.
5. **Persist:** Variables, data, and execution context remain available for follow-up tasks. 


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

---
