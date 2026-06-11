# return-risk

`return-risk` predicts, at checkout time, whether an order is likely to be
returned. It pairs a feature catalog with a binary classifier so CX can flag
high-risk orders before they ship.

## Setup

Once you have the repo, run this from the project root (it assumes the
[dependencies](#dependencies) at the bottom are already installed):

```bash
uv sync
make check    # verify your environment is ready
make train
```

`uv sync` creates the virtualenv and installs dependencies. `make check`
confirms the stack imports (and tells you to install `libomp` if it doesn't).
`make train` trains and evaluates the model. The data CSVs in `data/` are
committed — you don't need to generate them.

## Commands

```bash
make train    # train + evaluate
make eval     # evaluate only (no retrain)
make test     # run pytest
make lint     # ruff check
```

## Where things live

- `models/return-risk/MODEL_CARD.md` — problem statement, eval protocol, and
  current baseline. Read this first.
- `docs/CONTRIBUTING.md` — feature, eval, and review standards.
- `feature_catalog/` — feature implementations, registry, and colocated tests.
- `models/return-risk/feature-configs/` — per-model feature selection.
- `models/return-risk/src/` — training and prediction code.
- `data/` — committed seed CSVs (`orders`, `order_items`, `returns`,
  `support_contacts`).

## Bring your own AI tool

Claude Code, Cursor, Codex, etc. — bring whatever you're comfortable with.

## Submitting your work (if a PR is required)

This repo is owned by the Extend org, so you can't push branches directly to it. Instead, submit via a fork:

1. Fork this repo to your own GitHub account (the "Fork" button, top-right on GitHub). With the `gh` CLI: `gh repo fork helloextend/poc-machine-learning-coding-challenge --clone`.
2. Do your work on a branch in your fork.
3. Push your branch to your fork.
4. Open a Pull Request targeting `helloextend/poc-machine-learning-coding-challenge:main`. With the `gh` CLI: `gh pr create --repo helloextend/poc-machine-learning-coding-challenge --base main --head <your-github-username>:<your-branch>`.

The PR will appear in this repo's Pull Requests tab. An Extend team member will review it there.

## Dependencies

You'll need the following installed before you get started:

- **Python 3.12** — pinned via `.python-version`; `uv` auto-fetches it if you
  don't have it ([python.org/downloads](https://www.python.org/downloads/)).
- **`uv`** — Python package/environment manager
  ([install guide](https://docs.astral.sh/uv/getting-started/installation/)).
- **`make`** — on macOS run `xcode-select --install`; on Debian/Ubuntu
  `sudo apt install build-essential`; see
  [GNU make](https://www.gnu.org/software/make/) otherwise.
- **`libomp`** (macOS only) — required by XGBoost's runtime; without it the
  import fails. Install with `brew install libomp`.
- **`git`**.
