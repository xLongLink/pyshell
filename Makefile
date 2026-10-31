.PHONY: up down logs test format

up:
	export PYSHELL_EXECUTION_TOKEN="$${PYSHELL_EXECUTION_TOKEN:-$$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"; \
	docker compose -f src/compose.yaml build pyshell && \
	docker compose -f src/compose.yaml up --force-recreate --detach --wait pyshell && \
	npm --prefix src/web ci && \
	npm --prefix sample/src/ui ci && \
	npm --prefix sample/src/ui run build && \
	uv sync --locked --project sample && \
	uv run --locked --project sample uvicorn sample.main:app --host 127.0.0.1 --port 8010 --ws-max-size 131072 --no-access-log --log-level warning

down:
	PYSHELL_EXECUTION_TOKEN=unused docker compose -f src/compose.yaml down --remove-orphans

logs:
	PYSHELL_EXECUTION_TOKEN=unused docker compose -f src/compose.yaml logs --follow --no-log-prefix pyshell

test:
	uv run --locked --project src/api pytest -c src/api/pyproject.toml src/api/tests -v
	uv run --locked --project sample pytest -c sample/pyproject.toml sample/tests -v
	npm --prefix sample/src/ui test
	npm --prefix src/web test

format:
	uvx ruff check --fix .
	uvx pyunslop
	uvx ruff format .
	npm --prefix sample/src/ui run format
	npm --prefix src/web run format
