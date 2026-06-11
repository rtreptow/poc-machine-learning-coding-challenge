.PHONY: check train eval test lint

check:
	@uv run python -c "import pandas, sklearn, xgboost" \
		&& echo "OK - environment ready (xgboost loads)" \
		|| (echo "FAILED - dependency import error. On macOS, xgboost needs libomp: brew install libomp"; exit 1)

train:
	uv run python models/return-risk/src/train.py

eval:
	uv run python models/return-risk/src/train.py --eval-only

test:
	uv run pytest

lint:
	uv run ruff check .
