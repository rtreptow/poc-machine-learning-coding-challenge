# Senior run sheet (60 min)

## Before the session
- [ ] Confirm candidate completed the setup checklist (Python, uv, gh auth,
      filesystem+exec-capable AI tooling) against the smoke-test repo.
- [ ] `make stamp-senior` (recreates `senior/start` if the repo changed).
- [ ] Grant repo access AT interview start, not before.
- [ ] Have open: `../answer_keys/senior.md`, `../briefs/priya.md`,
      `../tickets/RISK-412.md`, `../tickets/part2-serial-returners.md`.

## Session start script
- State: 60-minute box; AI tooling expected; deliverable is a PR against
  `senior/start`; "Priya is on Slack — use her as you would a colleague."
- Share the RISK-412 ticket text.

## Timeline
- 0:00–0:05 clone, `uv sync`, orient. (Env broken? Fall back to
  screen-sharing your pre-built checkout — never burn the hour debugging.)
- 0:05–0:30 Part 1. Strong candidates land it by 0:20–0:25.
- Nudge: ONE in-character nudge max, only if the prediction-time boundary is
  unengaged by [TRIGGER TIME — finalize at dry run]. Wording: [finalize at
  dry run].
- ~0:30 Part 2 — OFFER ONLY if Part 1 landed decisively with real time
  left. Never rush a candidate into Part 2 to fill the hour; a candidate who
  spends 35 min on Part 1 spent the hour as intended.
- 0:50–1:00 wrap + discussion. PR polish may spill past the hour (you
  observed the whole session; the deliverable isn't a race).

## After
- Grade from the answer key + rubric while fresh. Close the PR after grading.
