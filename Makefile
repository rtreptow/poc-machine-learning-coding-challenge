.PHONY: train eval test lint

train:
	uv run python models/return-risk/src/train.py

eval:
	uv run python models/return-risk/src/train.py --eval-only

test:
	uv run pytest

lint:
	uv run ruff check .
