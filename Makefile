.PHONY: up down test format

up: export PYSHELL_REQUIREMENTS ?= swissparlpy
up:
	docker compose -f api/compose.yaml up --build --detach --wait pyshell
	npm --prefix web/src/ui ci
	npm --prefix web/src/ui run build
	uv run --locked --project web uvicorn web.main:app --host 127.0.0.1 --port 8010 --ws-max-size 16384

down:
	docker compose -f api/compose.yaml down --remove-orphans

test:
	uv run --locked --project api pytest -c api/pyproject.toml api/tests -v
	uv run --locked --project web pytest -c web/pyproject.toml web/tests -v
	npm --prefix web/src/ui test

format:
	uvx ruff check --fix .
	uvx pyunslop
	uvx ruff format .
	npm --prefix web/src/ui run format
