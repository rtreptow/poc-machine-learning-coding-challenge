# Return-Risk Model Card

## Problem

Binary classifier predicting, **at checkout time**, whether an order will be
returned within 60 days. CX uses the score to intercept risky orders before
fulfillment; scoring happens synchronously in the checkout path.

## Data

Four tables committed in `data/`: `orders`, `order_items`, `returns`,
`support_contacts`. Label: order returned within 60 days of checkout.

## Features

Selected per-model in `feature-configs/v1.yaml`; implementations live in
`feature_catalog/`.

## Evaluation protocol

- **Temporal split**: train on all orders up to (latest checkout − 3 months);
  evaluate on the final 3 months. A random split is prohibited: it lets the
  model peek at orders contemporaneous with the holdout and inflates AUC.
- **Metric**: ROC AUC on the temporal holdout (`make eval`).
- **Current baseline: AUC 0.81 (v1.3, under this protocol).** This matches
  observed prod AUC (~0.81). Any PR claiming a different number must produce it
  under this protocol.

> **v1.3 correction (RISK-412):** the 0.93 quoted at v1.3 launch was not real.
> It came from two leaks introduced together: (1) `support_contact_count_30d`
> counted support contacts up to 30 days *after* checkout — post-purchase signal
> that doesn't exist at scoring time and correlates with returns; and (2) the
> eval was switched from the temporal split to a random 80/20 split. Fixing the
> feature to a strict pre-checkout lookback and restoring the temporal split
> yields 0.81, in line with prod. This was inflated offline evaluation, not drift.

> **v1.4 repeat-return features (RISK-431):** added two point-in-time customer
> features — `customer_prior_return_rate` (dollar-weighted share of the customer's
> prior order value returned) and `customer_prior_return_count` (volume). The rate
> counts only **matured** prior orders — those whose full 60-day return window has
> closed by this checkout — so it is not deflated by orders still in play, and a
> future return cannot leak in (a matured order's outcome is fully settled by
> scoring time). Offline AUC (temporal split) rises **0.81 → 0.90** (0.8962).
> Verified genuine, not leakage: a variant that lets prior orders' *future*
> returns leak in scores only ~+0.005 higher (0.8927 vs 0.8979 on the pre-maturity
> dollar-weighted feature). Caveats: customers whose entire history is under 60
> days old get rate 0.0 (no matured signal yet); the lift depends on reliable
> customer identity resolution at checkout (guest checkout weakens it);
> `RETURN_WINDOW_DAYS` in the feature must track `label_horizon_days`; confirm
> against prod before planning on 0.90.

## Monitoring

A weekly job scores realized labels at 60 days and reports prod AUC.
