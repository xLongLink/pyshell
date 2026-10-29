.PHONY: up down format

up:
	docker compose run --build --rm pyshell

down:
	docker compose down --remove-orphans

format:
	uvx ruff check --fix .
	uvx pyunslop .
	uvx ruff format .
