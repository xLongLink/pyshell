.PHONY: up down logs test format

up: export PYSHELL_REQUIREMENTS ?= swissparlpy
up:
	docker compose -f api/compose.yaml build --no-cache pyshell
	docker compose -f api/compose.yaml up --force-recreate --detach --wait pyshell
	npm --prefix web/src/ui ci
	npm --prefix web/src/ui run build
	uv sync --locked --project web --reinstall
	uv run --locked --project web uvicorn web.main:app --host 127.0.0.1 --port 8010 --ws-max-size 131072 --no-access-log --log-level warning

down:
	docker compose -f api/compose.yaml down --remove-orphans

logs:
	docker compose -f api/compose.yaml logs --follow --no-log-prefix pyshell

test:
	uv run --locked --project api pytest -c api/pyproject.toml api/tests -v
	uv run --locked --project web pytest -c web/pyproject.toml web/tests -v
	npm --prefix web/src/ui test

format:
	uvx ruff check --fix .
	uvx pyunslop
	uvx ruff format .
	npm --prefix web/src/ui run format
