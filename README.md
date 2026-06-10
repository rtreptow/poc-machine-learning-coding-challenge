# return-risk

Return/abuse risk at checkout: a feature catalog and a binary classifier
predicting, at checkout time, whether an order will be returned.

## Quickstart

    uv sync
    make train    # train + evaluate (temporal holdout)
    make test
    make lint

See `models/return-risk/MODEL_CARD.md` for the eval protocol and
`docs/CONTRIBUTING.md` for feature/eval standards.
