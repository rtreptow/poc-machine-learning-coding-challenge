.PHONY: data train eval test lint ground-truth check-calibration render-docs freeze-senior stamp-senior freeze-manager stamp-manager-pr

BASE_REF ?= main

data:
	uv run python data/generate.py

train:
	uv run python models/return-risk/src/train.py

eval:
	uv run python models/return-risk/src/train.py --eval-only

test:
	uv run pytest

lint:
	uv run ruff check .

ground-truth:
	uv run python -m interviewer.tools.compute_ground_truth

check-calibration:
	uv run python -m interviewer.tools.check_calibration

render-docs:
	uv run python -m interviewer.tools.render_docs

freeze-senior:
	uv run python -m interviewer.tools.author_senior_branch

freeze-manager:
	uv run python -m interviewer.tools.author_alex_pr

stamp-senior:
	git worktree remove --force .stamp-tmp 2>/dev/null || true
	git worktree add --detach .stamp-tmp $(BASE_REF)
	GIT_COMMITTER_NAME="Dana Riggs" GIT_COMMITTER_EMAIL="dana.riggs@vector-contracting.example" \
		git -C .stamp-tmp am --committer-date-is-author-date $(abspath interviewer/patches/senior)/*.patch
	git branch -f senior/start `git -C .stamp-tmp rev-parse HEAD`
	git worktree remove --force .stamp-tmp

stamp-manager-pr:
	uv run python -m interviewer.tools.stamp_manager_pr --candidate $(CANDIDATE)
