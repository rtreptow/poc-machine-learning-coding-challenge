# Return-Risk Model Card

## Problem

Binary classifier predicting, **at checkout time**, whether an order will be
returned within 60 days. CX uses the score to intercept risky orders before
fulfillment; scoring happens synchronously in the checkout path.

## Data

Four tables in `data/`: `orders`, `order_items`, `returns`,
`support_contacts`. Label: order returned within 60 days of checkout.

## Features

Selected per-model in `feature-configs/v1.yaml`; implementations live in
`feature_catalog/`. Anything derived from customer history must be
computable at the order's checkout time (see CONTRIBUTING.md).

## Evaluation protocol

- **Temporal split**: train on all orders up to (latest checkout − 3 months);
  evaluate on the final 3 months. Return behavior drifts, so a shuffled
  split overstates performance — the holdout must be the future.
- **Metric**: ROC AUC on the temporal holdout (`make eval`).
- **Current baseline: AUC 0.80.** Any PR claiming a different
  number must produce it under this protocol.

## Monitoring

A weekly job scores realized labels at 60 days and reports prod AUC.
