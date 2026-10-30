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
- 📦 **Custom environment:** Install Python libraries at container startup.
- 📊 **Rich results:** A2UI renders Plotly-style figures with lines, markers, bars, and pies alongside code and printed output.


<br />

## Getting started


```bash
docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/xlonglink/pyshell:latest
```
See [api/pyproject.toml](api/pyproject.toml) for default libraries and version constraints.  Use `PYSHELL_REQUIREMENTS=$'httpx2\'` tp install extra libraries

<br />

## How it works

TODO

<br />


## Development

Run from the repository root:

```bash
make up         # Setup the local server
make down       # Remove and cleanup the docker image
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
