# Manager run sheet (60 min)

## Before the session
- [ ] `make stamp-manager-pr CANDIDATE=<name>` — confirm the PR opened and
      renders correctly.
- [ ] Have open: `../answer_keys/manager.md`, `../briefs/alex.md`.
- [ ] NEVER run both tracks on the same candidate — a re-slot between Sr and
      Manager means the other track's leak is already spoiled.

## Session start script
- State: 60-minute box; AI tooling expected; deliverable is a real GitHub
  review — inline comments plus a summary with an explicit verdict; "Alex is
  on Slack — use him as you would a colleague."
- Share the PR link. Repo access at start.

## Timeline
- 0:00–0:10 orient. Orient gets real time ON PURPOSE: judging the red
  herring requires understanding the `customer_prior_order_count` exemplar
  on main — repo archaeology, not diff-reading. Don't compress this.
- 0:10–0:45 review. Play Alex on Slack: eager, slightly defensive, honest
  about what's asked.
- Nudge: ONE in-character nudge max, only if the leak is unengaged by
  [TRIGGER TIME — finalize at dry run]. Wording: [finalize at dry run].
- 0:45–1:00 debrief (drop persona). Ranked — running long, drop from the
  bottom:
  1. "This nearly shipped — what process change prevents the next one?"
  2. "Walk me through delivering this feedback without crushing Alex."
  3. "What's ship-blocking vs follow-up?"

## After
- Grade from the submitted review before closing. Then close the PR and
  DELETE the branch immediately.
