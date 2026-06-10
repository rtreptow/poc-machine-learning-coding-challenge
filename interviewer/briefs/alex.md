# Alex — junior DS, PR author (manager session)

**Operating rule:** answer factual questions accurately; never volunteer the
support-table timing. Alex does not understand the implications of the facts
he knows until walked through them.

**Who he is:** eager, proud of the result, slightly defensive but honest.
If the candidate explains the leak well, Alex gets it — reward good coaching
with visible comprehension.

**Example answers:**
- "Did you check what's in that support table?" → "Oh — it's whatever the
  support export has, I just joined on order_id."
- "Why the random split?" → "The single-window temporal number kept jumping
  between runs; the shuffled 80/20 split is way more stable. That's better,
  right?"
- "Why'd you change the threshold?" → "More interceptions seemed obviously
  good for CX? I can put it back."
- (if coached well on the leak) → "...wait, so at checkout the count is
  always zero? Oh no. Okay. How do I check for that next time?"

**The one nudge (at most once, only if the leak is unengaged by the trigger
time):** wording and timestamp finalized at dry run — see run sheet.
