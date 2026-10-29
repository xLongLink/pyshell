.PHONY: up down format

up:
	docker compose up --build --detach --wait pyshell
	uv run --locked python script.py

down:
	docker compose down --remove-orphans

format:
	uvx ruff check --fix .
	uvx pyunslop
	uvx ruff format .
