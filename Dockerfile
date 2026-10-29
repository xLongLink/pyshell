FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 1000 pyshell \
    && mkdir /workspace \
    && chown pyshell:pyshell /workspace
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

USER pyshell
WORKDIR /workspace
ENTRYPOINT ["sh", "-eu", "-c", "\
requirements=/opt/pyshell/requirements.txt; \
if [ -e \"$requirements\" ]; then \
    environment=$(mktemp -d /tmp/pyshell-env.XXXXXX); \
    cp -a /opt/venv/. \"$environment/\"; \
    uv pip install --no-cache --python \"$environment/bin/python\" -r \"$requirements\"; \
    uv pip check --no-cache --python \"$environment/bin/python\"; \
    export PATH=\"$environment/bin:$PATH\"; \
    exec \"$environment/bin/python\" -m pyshell \"$@\"; \
fi; \
exec /opt/venv/bin/python -m pyshell \"$@\"", "pyshell"]
