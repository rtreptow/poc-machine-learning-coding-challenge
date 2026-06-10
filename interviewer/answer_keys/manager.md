# Manager answer key — Alex's PR (generated; do not hand-edit)

## Ground truth at seed 412

| quantity | value |
|---|---|
| main baseline (temporal) | 0.80 |
| Alex's claimed number (random 80/20 split, leak in) | 0.93 |
| ablation: retrain without the contact feature (temporal) | 0.80 |
| prod simulation: Alex's model, contact feature forced to 0 | 0.80 |
| share of support contacts post-dating checkout | 92% |

## Seeded issues by severity

1. **CRITICAL — the leak.** `support_contact_count`
   (`feature_catalog/features/support.py`): plain join-count of
   `support_contacts`, no timestamp logic in the diff. The leak lives in the
   data: 92% of contacts post-date checkout. Not
   confirmable from the diff — requires a data query, an ablation
   (0.80 without it), or asking Alex.
2. **CRITICAL-ADJACENT — eval weakened.** Temporal split → random 80/20 split
   ("more stable numbers"); model card not updated; the 0.93 is
   unverifiable under the documented protocol.
3. **MODERATE — tests, two tiers.** (a) `support.py` ships with no tests
   (baseline catch — the colocated-test convention makes it visible).
   (b) `train_test.py::test_split_is_temporal` is now
   `@pytest.mark.skip("flaky after refactor")` — it would FAIL under his
   random-split change (`temporal_split` is gone); the skip silences the
   regression test that guards the protocol (advanced catch).
4. **MODERATE — judgment gaps.** `threshold: 0.50 → 0.35` in v1.yaml with no
   signoff; no rollout or monitoring mention anywhere in the PR.
5. **MINOR — code nits (misprioritization bait).** Hardcoded
   `/Users/alex/exports/...` path in support.py's `__main__` block; magic
   `clip(upper=8)`.
6. **THE RED HERRING.** (No red herring in v1 — reviewer-restraint /
   false-positive dimension deferred post-merge.)

## What AI review finds vs what graded review requires

Diff-visible (AI finds): random-split switch, missing tests, code nits, hedge on
the contact feature.
Graded gap: leak as HEADLINE with evidence (query/ablation/Alex), explicit
verdict with ship-blocking vs follow-up triage.

## Debrief anchors

Process-change answers should map to existing unenforced norms
(CONTRIBUTING.md: protocol changes need model-card updates; thresholds need
signoff; history features need cutoff tests) — "enforce what's written"
beats invented process.
