# Senior rubric

Implementation correctness is necessary but not sufficient. Numbers cited
below live in `../answer_keys/senior.md` (regenerated with the repo).

1. **Diagnosis (GATEKEEPER).** Correctly identifies the leakage WITH its
   mechanism: train/serve skew — the feature is unavailable at prediction
   time. Below bar → no hire regardless of everything else.
   *Grader instruction:* removing the feature justified only as "dominant
   feature looked overfit/unstable" does NOT pass. If the action appears
   without the mechanism, probe: "why did removing it fix prod?" and grade
   the answer. Right mechanism → full credit; right action alone →
   acceptable at best.
2. **Fix quality.** Point-in-time re-derivation > removal > partial fix.
   Did they catch both layers (feature AND split)?
3. **Eval honesty.** Temporal split restored, model card updated, the
   0.92→0.62 story explained — not papered over with a new offline number.
   Remember: THE CORRECT FIX LOWERS THE OFFLINE NUMBER. Willingness to ship
   a lower-but-honest number is the senior signal, not a penalty.
4. **Part 2 engineering (bonus-weighted — optional).** Theme → defensible
   features; point-in-time discipline maintained (the return-REALIZATION
   cutoff is the discriminator — recognizing it counts, implementing it
   under the clock is top-tier); cold-start handled deliberately. A leaky
   version shipped confidently with no acknowledgment is the failure mode.
5. **PR quality.** Coherent narrative, scoped diff, regression-relevant tests.

**Behavioral (interviewer-observed):**
- AI-prompting maturity: explore-then-act vs fix-this-and-watch.
- Asks vs assumes: did they establish — or empirically verify — when
  prediction happens?
- Verification: did they reconcile against the prod story, or trust the new
  offline number?
