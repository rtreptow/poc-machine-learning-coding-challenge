## Add customer behavioral feature

A new feature that captures how customers actually behave:

- **`support_contact_count`** — support touchpoints on the order. Friction
  signal: orders that generate support noise return more.

## Results

**AUC $baseline_auc → $alex_random_auc** 🎉 (random 80/20 split)

I also switched eval to a random 80/20 split — single-window numbers were
jumping around between runs and the shuffled split gives much more stable
estimates.

Would love to get this in before the planning cycle — happy to walk anyone
through it!
