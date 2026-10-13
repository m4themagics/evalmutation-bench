.PHONY: install generate lint typecheck test verify smoke schemas site

install:
	uv sync --extra dev --extra service --extra retail

generate:
	uv run evalmut generate-dataset --output-dir data

schemas:
	uv run python scripts/export_schemas.py

site:
	uv run python scripts/build_site.py

lint:
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest --cov=evalmutationbench --cov-report=term-missing

smoke: generate
	uv run evalmut validate-dataset data/development.jsonl data/validation.jsonl data/challenge.jsonl
	uv run evalmut run --dataset data/development.jsonl --adapter deterministic --config configs/deterministic.yaml --output-dir artifacts/smoke
	uv run evalmut validate-submission --dataset data/development.jsonl --predictions artifacts/smoke/predictions.jsonl
	uv run evalmut score --dataset data/development.jsonl --predictions artifacts/smoke/predictions.jsonl --output artifacts/smoke/metrics.json --bootstrap-samples 100
	uv run evalmut report --metrics artifacts/smoke/metrics.json --adapter deterministic --dataset-name development --output artifacts/smoke/report.md

verify: schemas site lint typecheck test smoke

