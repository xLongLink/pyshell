.PHONY: up down

up:
	docker compose run --build --rm pyshell

down:
	docker compose down --remove-orphans
