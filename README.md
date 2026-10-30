<div align="center">

<img src="banner.png" alt="PyShell banner" />

<h3 >

ghost in the shell

</h3 >

Give instructions in your browser. Watch the agent write and run Python inside Docker.

</div>


<br />

## What it does

- 🧠 **Persistent sessions:** Variables, imports, and conversation context carry over between instructions.
- 🔄 **Execution feedback:** Output and errors go back to the model so it can continue or correct mistakes.
- 📦 **Custom environment:** Install Python libraries at container startup.
- 📊 **Rich results:** A2UI renders bar, line, and pie charts alongside code and printed output.


<br />

## Getting started

### Run the released API container

Start the published execution API with Docker only; no clone or local build is needed:

```bash
docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/xlonglink/pyshell:latest
```

The API is available at `http://127.0.0.1:8000`. Files are removed when the container
exits. Use a release tag such as `:vX.Y.Z` instead of `:latest`
to pin a version. This container does not include the browser app or agent;
follow the setup below for the full experience.

### Run the full app from source

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
{"stdout":"42\n","stderr":"","exception":null,"done":false,"ui":[]}
```

Use `print()` for results. `GET /` reports health; `POST /reset` clears the Python
namespace. The website's Reset button also clears the conversation and transcript.

All requests share one session, and execution is serialized. Run only one server
worker for each API. Reset does not undo file writes or other side effects. Keep
both APIs on loopback: this local prototype has no authentication, and generated
Python runs without approval.

## A2UI results

Try asking:

> Create a clearly labeled synthetic sales dataset. Show monthly revenue as a
> line chart, product totals as a bar chart, and product shares as a pie chart.
> Print a concise summary. Use only the Python standard library.

Python has a provided `ui` object; no import or extra package is required:

```python
chart = ui.chart.line([{"label": "Jan", "value": 420}], title="Synthetic revenue", surface_id="chart")
chart.pin()
chart.focus()
print("Synthetic sales: $420 total revenue")
done()
```

Creation methods return widget objects. Keep them in Python variables and use
their shared methods:

```python
chart.update(points=[{"label": "Jan", "value": 450}], title="Updated revenue")
chart.unfocus()
chart.unpin()
chart.remove()
```

New widgets appear in the center, which holds up to four widgets in a grid.
When a fifth arrives, the oldest centered widget moves to the sidebar.
`focus()` and `unpin()` move a widget into the center; `pin()` and `unfocus()`
move it to the sidebar. Placement is controlled only by Python or the agent;
the browser has no placement controls. `update()` validates attributes and preserves widget identity and
placement. Use it instead of assigning widget attributes directly.

`widget.surface_id` is its stable identifier. Reuse a `surface_id` to replace
a result, or call `ui.remove(surface_id)` to remove it by name. Removed, replaced,
and reset widget handles cannot publish further changes.
Widgets survive follow-up instructions and browser reconnection;
Reset clears them with the conversation and Python namespace. Python errors
preserve widgets already published before the error.

The restricted A2UI v0.9 catalog supports only `BarChart`, `LineChart`, and
`PieChart`. Create charts with
`ui.chart.line(points, ...)`, `ui.chart.bar(points, ...)`, or
`ui.chart.pie(points, ...)`. There is no `kind` argument or generic `ui.chart()` call.
Each point has a `label` and numeric `value`, or `None` for missing data. Lines
preserve gaps. Pie charts require nonnegative values and omit missing slices;
all chart types retain missing values in their accessible data tables.
Each surface contains one chart with component ID `root`. The low-level
`ui.show(...)` also accepts one typed chart or a one-item list containing its
wire definition. Text, metric, table, and layout widgets are not available;
use `print()` for summaries and tabular results. Arbitrary HTML, JavaScript,
remote catalogs, inputs, and custom executable actions are not supported.

The Python publisher uses official `a2ui-core` message models, catalogs, and
structural validation. All Python components inherit `api/src/ui/widget.py`;
their own model attributes drive their creation and updates without separate
field-by-field display wrappers. The browser uses the official A2UI React renderer.
Both SDKs are pinned to compatible package versions. A2UI currently depends on Zod 3, which has an upstream collection
validation denial-of-service advisory. UI batches are size- and complexity-
bounded before schema validation, but `npm audit` still reports the dependency
advisory. Keep this prototype local; revisit dependencies before production.

## Customize the environment

See [api/pyproject.toml](api/pyproject.toml) for default libraries and version constraints.
Use the provided `ui` charts for plotting.

Pass extra packages as newline-separated requirements (Bash syntax):

```bash
PYSHELL_REQUIREMENTS=$'httpx\npypdf' make up
```

Without `PYSHELL_REQUIREMENTS`, no extra packages are installed. Packages are
installed at container startup; installation failure prevents launch.

See [Contributing](CONTRIBUTING.md#dependencies) for dependency and container details.

## Contributing

Run tests with `make test`. See [CONTRIBUTING.md](CONTRIBUTING.md) for development,
workspace access, and release instructions.

`make up` builds the website and starts the web server and execution container.
For frontend development, run `npm --prefix web/src/ui run dev` separately and
open `http://127.0.0.1:5180`.
The original terminal client
remains available with `uv run --locked --project web python -m web.src.script`;
do not use it concurrently with the browser agent, since the namespace is shared.

---

<div align="center">

LongLink 2026

[MIT License](./LICENSE) &nbsp; - &nbsp; [Contributing](./CONTRIBUTING.md) &nbsp; - &nbsp; [Code of Conduct](./CODE_OF_CONDUCT.md) &nbsp; - &nbsp; [Contact](mailto:info@longlink.dev)

</div>
