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
- 📊 **Rich results:** A2UI renders Plotly-style figures with lines, markers, bars, and pies alongside code and printed output.


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

`GET /libraries` lists installed Python packages and their versions, including extras installed
through `PYSHELL_REQUIREMENTS`, as `[{"name":"numpy","version":"2.3.0"}]`.
Names are distribution names, which can differ from Python import names.
The agent includes this inventory in the model's instructions and refreshes it after reset.

Use `print()` for results. `GET /health` reports health; `POST /reset` clears the Python
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
chart = ui.plot(
    data=[
        {"type": "scatter", "name": "Predicted", "mode": "lines", "x": [0, 5, 10], "y": [400, 450, 500]},
        {"type": "scatter", "name": "Observed", "mode": "markers", "x": [2, 7], "y": [420, 470]},
    ],
    layout={"title": {"text": "Synthetic revenue"}, "xaxis": {"title": {"text": "Day"}}},
    surface_id="chart",
)
chart.pin()
chart.focus()
print("Synthetic data: predicted and observed revenue")
done()
```

Creation methods return widget objects. Keep them in Python variables and use
their shared methods:

```python
chart.update(
    data=[{"type": "scatter", "mode": "lines+markers", "x": [0, 5, 10], "y": [410, 460, 510]}],
    layout={"title": {"text": "Updated revenue"}, "showlegend": False},
)
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
Updates are explicit and atomic: supply `data` and `layout` together to batch changes.
Supplied top-level fields replace their previous contents, without deep merging;
omitted fields remain unchanged. Invalid updates leave the current figure intact.

`widget.surface_id` is its stable identifier. Reuse a `surface_id` to replace
a result, or call `ui.remove(surface_id)` to remove it by name. Removed, replaced,
and reset widget handles cannot publish further changes.
Widgets survive follow-up instructions and browser reconnection;
Reset clears them with the conversation and Python namespace. Python errors
preserve widgets already published before the error.

The restricted A2UI v0.9 catalog (`urn:pyshell:catalog:v3`) supports one `Figure`
component, with Plotly-style `data` traces and `layout`. It is a small supported
subset, rendered with Recharts—not the Plotly Python package or arbitrary Plotly JSON.

- Scatter: `{"type": "scatter", "x": [0, 5], "y": [1, 4], "mode": "lines"}`.
  Modes are `lines` (default), `markers`, and `lines+markers`.
- Bar: `{"type": "bar", "x": ["A", "B"], "y": [42, 50]}`.
- Pie: `{"type": "pie", "labels": ["A", "B"], "values": [42, 50]}`.
- Each trace can have a `name`. Scatter and bar traces may share a figure;
  a pie figure contains exactly one pie trace.
- Paired arrays have equal, nonzero lengths. A figure accepts at most 20 traces
  and 200 total points. X coordinates must be all numbers or all category strings
  across the figure; numbers receive numeric spacing. Trace order is preserved.
- Use `None` for missing Y/pie values. Lines preserve gaps; pie values must be
  nonnegative and missing slices are omitted. All-missing figures show an empty state.
- Layout supports `title`, `xaxis.title`, `yaxis.title` (each title is
  `{"text": "..."}`), and `showlegend` (default `True`). There are no subplots,
  secondary axes, date axes, or arbitrary Plotly configuration options.

This replaces `ui.chart.line/bar/pie`: use `ui.plot` with `scatter`, `bar`, or `pie`
traces, and put the title in `layout.title.text`. Rebuild both services together;
the previous catalog and component names are no longer supported.

Each surface contains one figure with component ID `root`. The low-level
`ui.show(...)` also accepts one typed figure or a one-item list containing its
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

`make up` installs `swissparlpy` as a showcase extra library for Swiss Parliament
data. It is passed through `PYSHELL_REQUIREMENTS`, not added to the locked dependencies.
The agent can then use `import swissparlpy`.

Override the extra packages with newline-separated requirements (Bash syntax):

```bash
PYSHELL_REQUIREMENTS=$'httpx\npypdf' make up
```

Use `PYSHELL_REQUIREMENTS='' make up` to skip extra packages. Packages are installed
at container startup; installation failure prevents launch.

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
