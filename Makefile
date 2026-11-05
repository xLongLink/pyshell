.PHONY: up down logs train export benchmark test test\:train test\:benchmark format dataset\:download dataset\:upload model\:download model\:upload

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
	@mkdir -p model/train/data/raw
	@for split in train validation; do \
		if ! test -f model/train/data/raw/$$split.jsonl; then \
			hf download LongLink/pyshell-worker-data data/$$split.jsonl --type dataset --revision 1ed43ba23eff6438850921dca30665e5608fb8b7 --local-dir model/train/.cache/dataset || exit $$?; \
			cp model/train/.cache/dataset/data/$$split.jsonl model/train/data/raw/$$split.jsonl || exit $$?; \
		fi; \
	done

dataset\:upload:
	hf upload LongLink/pyshell-worker-data model/train/data/raw data --type dataset --include "*.jsonl"

model\:download:
	hf download LongLink/pyshell-worker-qwen35-08b --revision b66f303f929419c7973d06347e61bd776d87fe9e --local-dir model/train/weights

model\:upload:
	hf upload LongLink/pyshell-worker-qwen35-08b model/train/weights . --type model --exclude ".cache/**"

train:
	uv run --locked --project model/train --extra cpu python -m model.train.main train $(ARGS)

export:
	uv run --locked --project model/train --extra cpu python -m model.train.main export $(ARGS)

benchmark:
	uv run --locked --project model/benchmark python -m model.benchmark.main $(ARGS)

test\:train:
	uv run --locked --project model/train --extra cpu python -m pytest -c model/train/pyproject.toml model/train/tests -v

test\:benchmark:
	uv run --locked --project model/benchmark --extra local --extra data python -m pytest -c model/benchmark/pyproject.toml model/benchmark/tests -v

test:
	uv run --locked --directory model/protocol pytest -v
	uv run --locked --project src/api pytest -c src/api/pyproject.toml src/api/tests -v
	npm --prefix src/web ci
	npm --prefix src/web test

format:
	uvx ruff check --fix .
	uvx pyunslop
	uvx ruff format .
	npm exec --prefix src/web -- prettier --write sample/assets/voice.mjs
	npm --prefix src/web run format
