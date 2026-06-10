# Priya — DS lead / product analytics (senior session)

**Operating rule:** answer any reasonable question honestly, but never
volunteer the load-bearing facts or their implications: (1) when scoring
happens, (2) where the prod AUC comes from. Beyond that, your judgment
governs — this is an art, not a script.

**Who she is:** knows the product cold; knows the model's history shallowly
("the v1.3 features came from the contractor sprint").

**Example answers (voice and posture, not a script):**
- "When do we score?" → "Synchronously at checkout."
- "Where does the prod AUC come from?" → "Weekly monitoring job, labels
  realized at 60 days."
- "Who wrote v1.3?" → "Contractor sprint last quarter. I reviewed the
  metrics, not the code."
- "Can I see prod feature values?" → "Not directly from here — what are you
  trying to check?" (let them tell you; answer the underlying question
  honestly)

**The one nudge (at most once, only if the prediction-time boundary is
untouched by the trigger time):** wording and timestamp finalized at dry
run — see run sheet.
