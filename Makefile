.PHONY: up down logs test format dataset\:download dataset\:upload model\:download model\:upload

up:
	docker compose -f src/compose.yaml build pyshell && \
	docker compose -f src/compose.yaml up --force-recreate --detach --wait pyshell && \
	uv sync --locked --project sample && \
	uv run --locked --project sample uvicorn sample.main:app --host 127.0.0.1 --port 8010 --ws-max-size 131072 --log-level warning

down:
	docker compose -f src/compose.yaml down --remove-orphans

logs:
	docker compose -f src/compose.yaml logs --follow --no-log-prefix pyshell

dataset\:download:
	@for split in train validation; do \
		test -f dataset/data/$$split.jsonl || \
		hf download LongLink/pyshell-worker-data data/$$split.jsonl --type dataset --local-dir dataset || exit $$?; \
	done

dataset\:upload:
	hf upload LongLink/pyshell-worker-data dataset/data data --type dataset --include "*.jsonl"

model\:download:
	hf download LongLink/pyshell-worker-qwen35-08b --revision b66f303f929419c7973d06347e61bd776d87fe9e --local-dir model/weights

model\:upload:
	hf upload LongLink/pyshell-worker-qwen35-08b model/weights . --type model --exclude ".cache/**"

test:
	uv run --locked --project src/api pytest -c src/api/pyproject.toml src/api/tests -v
	npm --prefix src/web ci
	npm --prefix src/web test

format:
	uvx ruff check --fix .
	uvx pyunslop
	uvx ruff format .
	npm exec --prefix src/web -- prettier --write sample/assets/voice.mjs
	npm --prefix src/web run format
