FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY main.py ./
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
if [ -n \"${PYSHELL_REQUIREMENTS:-}\" ]; then \
    environment=$(mktemp -d /tmp/pyshell-env.XXXXXX); \
    requirements=\"$environment/requirements.txt\"; \
    printf '%s\\n' \"$PYSHELL_REQUIREMENTS\" > \"$requirements\"; \
    cp -a /opt/venv/. \"$environment/\"; \
    uv pip install --no-cache --python \"$environment/bin/python\" -r \"$requirements\"; \
    uv pip check --no-cache --python \"$environment/bin/python\"; \
    export PATH=\"$environment/bin:$PATH\"; \
fi; \
exec python -m uvicorn main:create_app --factory --host 0.0.0.0 --port 8000 --workers 1 --no-access-log --log-level warning \"$@\"", "pyshell-api"]
