from html import escape
from string import Template
from pathlib import Path

PAGE = Template((Path(__file__).parent.parent / "assets" / "index.html").read_text(encoding="utf-8"))


def render_page(widget_url: str) -> str:
    """Render the voice control around the container-owned workspace."""
    return PAGE.substitute(widgets=escape(widget_url, quote=True))
