# Contributing

## Feature standards

- Features are plain functions `(Tables) -> pd.Series` indexed by `order_id`,
  registered in `feature_catalog/registry.py`, selected per-model in
  `models/<model>/feature-configs/`.
- Every feature module ships a colocated `*_test.py`.

## Eval standards

- The eval protocol lives in the model card (`MODEL_CARD.md`) and is the
  contract for any reported metric. Don't change the protocol and the model
  in the same PR; if the protocol must change, update the model card and say
  why.
- Report metrics from the documented protocol only. A number produced under
  a different split is not comparable and shouldn't be quoted.

## Review norms

- A PR that changes model behavior should state three things: the expected
  impact on the eval metric, how that impact was measured, and the
  rollout/monitoring plan.
- Decision thresholds in feature configs are business-owned. Changing one
  requires explicit sign-off from the CX stakeholder.
