.PHONY: up down test format

up:
	docker compose -f api/compose.yaml up --build --detach --wait pyshell
	uv run --locked --project web python web/script.py

down:
	docker compose -f api/compose.yaml down --remove-orphans

test:
	uv run --locked --project api pytest -c api/pyproject.toml api/tests -v
	uv run --locked --project web pytest -c web/pyproject.toml web/tests -v

format:
	uvx ruff check --fix .
	uvx pyunslop
	uvx ruff format .
