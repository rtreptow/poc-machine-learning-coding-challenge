# Contributing

## Feature standards

- **Prediction-time rule:** the model scores synchronously at checkout. A
  feature may only use information that exists at the order's `checkout_ts`.
  If it touches customer history, apply an explicit as-of cutoff (see
  `customer_prior_order_count` for the pattern).
- Every feature module ships a colocated `*_test.py`. History-derived
  features must include a test that future events are excluded.
- Features are plain functions `(Tables) -> pd.Series` indexed by `order_id`,
  registered in `feature_catalog/registry.py`, selected per-model in
  `models/<model>/feature-configs/`.

## Eval standards

- The eval protocol lives in the model card (`MODEL_CARD.md`) and is the
  contract for any reported metric. Don't change the protocol and the model
  in the same PR; if the protocol must change, update the model card and say
  why.
- Report metrics from the documented protocol only. A number produced under
  a different split is not comparable and shouldn't be quoted.

## Review norms

- PRs that change model behavior state: expected metric impact, how it was
  measured, and the rollout/monitoring plan.
- Decision thresholds in feature configs are business-owned; changing one
  requires explicit signoff from the CX stakeholder.
