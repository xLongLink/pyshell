.PHONY: up down agent web build test format

up:
	docker compose -f api/compose.yaml up --build --detach --wait pyshell
	npm --prefix web ci
	$(MAKE) build
	$(MAKE) agent

agent:
	uv run --locked --project web uvicorn server:app --app-dir web --host 127.0.0.1 --port 8010 --ws-max-size 16384

web:
	npm --prefix web ci
	npm --prefix web run dev

build:
	npm --prefix web run build

down:
	docker compose -f api/compose.yaml down --remove-orphans

test:
	uv run --locked --project api pytest -c api/pyproject.toml api/tests -v
	uv run --locked --project web pytest -c web/pyproject.toml web/tests -v
	npm --prefix web test

format:
	uvx ruff check --fix .
	uvx pyunslop
	uvx ruff format .
	npm --prefix web run format
