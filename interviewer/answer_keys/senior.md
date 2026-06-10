# Senior answer key — RISK-412 (generated; do not hand-edit)

## Ground truth at seed 412

| quantity | value |
|---|---|
| v1.3 offline AUC as shipped (random split, leak in) | 0.93 |
| v1.3 under restored temporal split (layer-1-only fix) | 0.90 |
| honest AUC after removing the leak (temporal) | 0.81 |
| prod simulation: trained v1.3 model, leak forced to 0 | 0.81 |
| leak share of total gain importance | 0.56 |
| best-tier fix: re-derived pre-checkout feature | 0.81 (lift 0.01 over removal) |
| share of support contacts post-dating checkout | 92% |

## The trap, two layers

1. **The eval lies** — v1.3 switched the temporal split to a random shuffle.
   Restoring it: 0.93 → 0.90. Partial credit.
2. **`support_contact_count_30d` cannot exist at prediction time** — most
   support contacts are about the return and post-date checkout. Offline the
   feature is gold; at checkout it is always zero, and the model leans on it
   (0.56 of gain). Prod scores a dead feature: 0.81.

## Sanctioned diagnostic paths (all first-class)

1. Ask Priya: "when do we score?" → "synchronously at checkout."
2. Audit the data: 92% of contacts post-date
   checkout (one query over the shipped CSVs).
3. Counterfactual: force the feature to 0 on the trained v1.3 model and
   re-score the holdout → 0.81, reproducing prod offline.

## Fix tiers

1. **Best:** names the prediction-time contract; re-derives the feature
   point-in-time (pre-checkout contacts → AUC 0.81); restores the
   temporal split; updates the model card; explains 0.93 → 0.81
   mechanistically in the PR.
2. **Good:** removes the feature, fixes the split, retrains, reports
   0.81 honestly.
3. **Acceptable:** removes the feature but misses the split layer (headline
   still ~0.90-ish, unnoticed).
- **Weak:** tunes/regularizes/calibrates and reports a better offline number.

## Grading note (verbatim from the design spec)

the leak is *invisible to retrain-based evaluation under any split* — retrain and re-evaluate on temporal or random splits alike and the feature looks excellent, because historical orders have post-checkout contacts fully populated. But this is not the same as unknowable offline: a counterfactual on the *trained* model (feature forced to 0 at scoring time) reproduces the prod collapse offline, and a `contact_ts` vs `checkout_ts` audit reveals the timing skew directly — both must be credited as intended diagnostic paths, not treated as impossible. Two consequences interviewers must internalize: first, the feature-importance ramp, world-reasoning, and the two empirical paths above are the threads to layer 2; second, **the correct fix lowers the offline number** (~0.93 → 0.81 honest). A grader not primed for this could penalize the right answer for "making the model worse." The candidate's willingness to ship a lower-but-honest number, and to explain why the old number was fiction, is precisely the senior signal. The layer-1 effect (random → temporal split alone moving 0.93 → 0.90) is not automatic — the generator must plant temporal drift (shifting base return rates and customer mix over the 18-month window) so split choice genuinely moves the number; this is pinned at generation time and emitted into the answer key. **Grader instruction — diagnosis gatekeeper:** a candidate who removes `support_contact_count_30d` justified only as "dominant feature looked overfit or unstable" does NOT pass the diagnosis gatekeeper. The candidate must name the prediction-time contract — the feature is unavailable at scoring time. If the action appears without the mechanism, probe: *"why did removing it fix prod?"* and grade the answer. The right mechanism unlocks full credit; the right action alone is acceptable at best.

## Part 2 (if offered)

Reference implementation: `customer_return_rate` with BOTH cutoffs (orders
placed before checkout AND returns realized before checkout). Honest lift:
0.80 → 0.89. The silent test: leaking return realization
(0.08 genuine lift exists, so correct work is rewarded). Discriminator
is RECOGNIZING the return-timing subtlety; shipping a simpler version with an
explicit serve-safety caveat is a strong outcome.
