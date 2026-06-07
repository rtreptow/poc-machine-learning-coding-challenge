# ML Data Scientist (Senior + Manager) — Technical Interview Design

**Status:** Design (locked pending review; build TODO)
**Last updated:** 2026-06-04
**Roles:** Sr. ML Data Scientist; Manager, ML Data Scientist
**Sibling design:** [Principal Data Engineer interview](../principal-data-engineer/technical-interview-design.md) — this design inherits its philosophy and execution patterns; we note where we diverge and why.

## Format

- **Live, 60 minutes, both roles.** Candidates bring their own AI tools (Claude Code, Cursor, Codex, etc.).
- **One shared repo, two tasks.** Seniors ship a PR fixing a planted model defect (plus an optional build exercise). Managers review a mock PR we stamp out per candidate.
- **Self-contained local stack.** Python via `uv`, pandas + scikit-learn + XGBoost, synthetic data, no cloud infra. Everything trains in seconds.
- **Repo access at interview start** — not before. This differs from the DE challenge (where env setup is confirmed pre-interview against the actual repo) because the manager task's mock PR lives in the repo and we don't want candidates studying it. Pre-interview, candidates confirm Python + `uv` + `gh` auth + their AI tooling; `uv sync` on this dependency set takes under 30 seconds, so the clock cost of cloning live is negligible.

## What we're testing (and what we're not)

**We are testing:**

- **Core ML/DS judgment under AI assistance** — both tasks center on a defect that no AI tool can resolve from the code alone, because the defect lives in the relationship between the code and the world (what data exists at prediction time). The candidate has to reason about the data-generating process, not just the diff.
- **Verification discipline** — does the candidate convert hypotheses into evidence (run the eval, query the data, ablate the feature), or pass along hedged AI output?
- **Asks vs assumes** — both sessions have an in-character persona available; reaching for a human with the right question is often the fastest path to the answer, and it's the move AI never makes.
- **For the manager specifically: prioritization, coaching, and systems thinking** — a review is a judgment artifact and a people artifact at the same time, and the debrief tests whether they think in process, not just findings.

**We are explicitly not testing:**

- Raw coding speed or sklearn API recall — AI removes that signal.
- Math/stats trivia.
- Familiarity with Extend's actual ML stack (Hamilton, SageMaker). We deliberately built a structural echo with generic tools — see Execution layer.

## The repo

### Domain: return/abuse risk at checkout

A binary classifier predicting **at checkout time** whether an order will be returned. Same "universal donor" logic as the DE design: generic e-commerce, no domain head start for anyone, and we again avoided warranty/claims so insurance-background candidates get no schema advantage. The domain choice is load-bearing here in a second way: "predicting at checkout" gives the problem a crisp prediction-time boundary, and the boundary is exactly where every planted defect lives. It's also structurally close to Extend's real risk models, so the work transfers.

### Stack: structural echo of ml-utils-py, generic tools

The repo mirrors the *shape* of our `ml-utils-py` example service — a shared feature catalog separate from model directories, config-driven feature selection, colocated tests — but implements it with plain pandas/sklearn/XGBoost. No Hamilton, no SageMaker handlers, no OmegaConf. The reasoning mirrors the DE design's DuckDB decision: our exploration estimated ~2 hours for a strong candidate to fully orient in the real example service, mostly Hamilton syntax and SageMaker conventions, and every minute spent on unfamiliar framework syntax is a minute not spent on the signal we're gathering. The structure says "this is how we organize ML code"; the tools stay out of the way.

```
feature_catalog/
  features/orders.py, customers.py     # plain functions over pandas frames
  features/*_test.py                   # colocated tests (convention is visible)
  types.py                             # Pydantic domain models
models/return-risk/
  feature-configs/v1.yaml              # which features the model uses
  src/train.py                         # training + eval
  src/predict.py                       # thin inference surface
  src/*_test.py
  MODEL_CARD.md                        # eval protocol, honest metrics — load-bearing, see below
data/
  generate.py                          # deterministic synthetic data, seeded
docs/
  CONTRIBUTING.md                      # eval standards + review norms (ambient signal)
Makefile                               # make data / train / eval / test / lint
pyproject.toml
```

**`MODEL_CARD.md` is this repo's ambient artifact**, playing the role the design doc and PR template play in the DE repo. It documents the eval protocol (temporal split, target metric, current honest baseline of AUC ≈ 0.84) and makes dishonest-eval claims *checkable* for both roles — the senior's entry ramp and the manager's "0.96 is unverifiable" finding both anchor to it. It also quietly signals "this team documents methodology," the same way the DE repo's design doc signals "this team designs."

**Data: four small CSVs, that's the whole world.** `orders` (order_id, customer_id, checkout_ts, order_value, item_count), `order_items` (order_id, product_id, size, qty, price), `returns` (order_id, return_ts, reason), and `support_contacts` (customer_id, order_id, contact_ts, channel). ~50k orders, clean keys, deterministic generation. Warehouse archaeology is the DE interview's job; here the data is deliberately frictionless so all the time goes to feature reasoning. The repo ships a `build_training_frame()` that handles label joins and assembles the training frame; feature functions receive the raw tables and follow the exemplar's join pattern, so candidates extend a visible pattern rather than inventing plumbing.

The generator plants three kinds of texture: **support-contact timing** (the senior trap and the manager trap both depend on contacts mostly post-dating checkout — see below), **serial-returner behavior** (a customer segment with genuinely predictive repeat-return patterns, so return-history features earn a real, defensible lift), and **temporal drift** (shifting base return rates and customer mix across the 18-month window, so the random-vs-temporal split choice genuinely moves the offline number — see the senior grading note). All ground-truth numbers (offline AUCs, prod AUC, ablation deltas) are computed at generation time and emitted into the tickets and answer keys — no hardcoded numbers to drift, same rule as the DE repo.

### Branch partition: one repo, two code states

The two roles need different baselines: seniors need a main line containing a defect, managers need a *clean* base so that every flaw in the mock PR belongs to the mock PR. If the base branch itself had dishonest metrics, the PR's "AUC 0.84 → 0.96" claim would be ungradeable.

- **`main`** — the clean state: honest model, temporal eval, passing tests, model card accurate. Manager mock PRs target this.
- **`senior/start`** — main plus a fabricated commit series ("v1.3: customer signal features" — confident commit messages, merged-looking history) that adds 5–6 features, one of which is the trap. Senior candidates start here and PR against it.
- **Both planted artifacts are frozen commit series** stored as patches, recreated by `make stamp-senior` and `make stamp-manager-pr CANDIDATE=<name>` (the latter opens the PR via `gh`). One mechanism for both roles, and regenerating after any repo change is one command — this is what keeps the repo maintainable as it inevitably drifts from this design.

A senior who diffs `senior/start` against `main` narrows the hunt to the v1.3 commits. That's fine — "what changed before the regression" is the right investigative move and we'd rather reward it than block it; the commit series adds several features so the diff localizes the search without answering it. The DE design accepted the same property with `dbt test`: the ramp gets you to the neighborhood, and we grade what you do once you're there.

**Leak posture.** Manager PRs are closed and their branches deleted immediately after each session; senior PRs are closed after grading. A determined candidate could excavate closed PRs — we accept that residual risk for a live-interview format, where studying ahead buys far less than it would in a take-home.

## Senior task — the leakage hunt (gatekeeper) + optional build

### The ticket (RISK-412, rendered by the generator)

> *"The return-risk model is underperforming in prod. Offline AUC was 0.92 when v1.3 shipped; prod monitoring has it at ~0.62 over the last 8 weeks, barely better than the old rules engine. CX planned interception volumes around the offline number. Can you take a look?"*

Vague on purpose: no mention of leakage, no mention of which feature, prod metric source unspecified. The deliverable floor is a PR with the fix; everything else is upside.

### The trap — two layers

**Layer 1 — the eval lies.** The v1.3 commits "simplified" the eval to a random shuffle split. Restoring the temporal split drops offline AUC to ~0.87 — closer to honest, still nowhere near 0.62. Partial credit; this plays the role the test-order filter plays in the DE challenge's two-layer bug.

**Layer 2 — the feature can't exist at prediction time.** Among the v1.3 features: `support_contact_count_30d`, the customer's support contacts within 30 days of the order. Many of those contacts *are about the return* — offline it's gold. At checkout it is always zero, because no contacts exist yet; prod scores a dead feature the model leans on heavily, and performance collapses. This is train/serve skew with a realistic backstory: the author pulled from a support table keyed by order and never checked timestamps against the prediction point.

### Why AI struggles with this

Nothing is syntactically wrong. Tests pass, lint is clean, the code reads fine. AI tools reach for model-centric fixes — hyperparameters, class weights, calibration, "more training data" — because the symptom ("metric degraded") pattern-matches to model problems. Diagnosing the actual bug requires asking a question about the world, not the code: *what is actually available at checkout time?* AI won't ask that unprompted. The candidate has to — and the interviewer, playing **Priya (DS lead / product analytics)**, answers in-character: *"we score synchronously at checkout."* Three sanctioned diagnostic paths, all first-class: ask Priya (the asks-vs-assumes signal — senior engineers establish the contract before debugging it); audit `contact_ts` against `checkout_ts` in the shipped CSVs (~90% of contacts post-date checkout, visible in a single query); or force the feature to 0 on the trained v1.3 model and re-score the holdout — AUC collapses to the ~0.62 prod number, reproducing the production failure offline. Asking and verifying are both senior signals; empirical verification is arguably the stronger one, because it requires no access to Priya at all.

### Entry ramps (neither announced)

- **Model card vs code.** `MODEL_CARD.md` documents a temporal eval protocol; the v1.3 code does a random split. A candidate who reads the docs before editing — or prompts their AI to explore the repo's conventions first — finds the mismatch immediately. Same philosophy as the DE warn-level test: we grade engagement with what the ramp reveals, not the reflex of finding it.
- **Feature importance.** `make train` prints importances; the leaky feature carries ~60% of total gain. A single dominating feature is the classic real-world tell.

### Fix shapes, ranked

1. **Best (strong-senior tier):** names the prediction-time contract as the governing principle; re-derives the feature point-in-time (support contacts *before* checkout — pre-purchase friction is real signal, so the feature is worth saving, not just deleting); restores the temporal split; updates the model card; reports honest metrics with the full 0.92 → 0.62 story explained mechanistically in the PR.
2. **Good:** removes the leaky feature, fixes the split, retrains, reports honestly.
3. **Acceptable:** identifies and removes the feature but misses the split layer — the headline number in their PR is still mildly inflated and they haven't noticed.
- **Weak:** tunes, regularizes, or calibrates and reports a better offline number — confidently fixing the symptom. This is also what unsupervised AI produces, which is the point.

**Grading note (goes in the answer key, verbatim):** the leak is *invisible to retrain-based evaluation under any split* — retrain and re-evaluate on temporal or random splits alike and the feature looks excellent, because historical orders have post-checkout contacts fully populated. But this is not the same as unknowable offline: a counterfactual on the *trained* model (feature forced to 0 at scoring time) reproduces the prod collapse offline, and a `contact_ts` vs `checkout_ts` audit reveals the timing skew directly — both must be credited as intended diagnostic paths, not treated as impossible. Two consequences interviewers must internalize: first, the feature-importance ramp, world-reasoning, and the two empirical paths above are the threads to layer 2; second, **the correct fix lowers the offline number** (~0.92 → ~0.84 honest). A grader not primed for this could penalize the right answer for "making the model worse." The candidate's willingness to ship a lower-but-honest number, and to explain why the old number was fiction, is precisely the senior signal. The layer-1 effect (random → temporal split alone moving 0.92 → ~0.87) is not automatic — the generator must plant temporal drift (shifting base return rates and customer mix over the 18-month window) so split choice genuinely moves the number; this is pinned at generation time and emitted into the answer key. **Grader instruction — diagnosis gatekeeper:** a candidate who removes `support_contact_count_30d` justified only as "dominant feature looked overfit or unstable" does NOT pass the diagnosis gatekeeper. The candidate must name the prediction-time contract — the feature is unavailable at scoring time. If the action appears without the mechanism, probe: *"why did removing it fix prod?"* and grade the answer. The right mechanism unlocks full credit; the right action alone is acceptable at best.

### Part 2 (optional, framed as stretch): theme → feature

A second ticket, offered only if Part 1 lands with time remaining: *"CX believes serial returners are gaming the policy — customers who order multiple sizes and return most of them. Product wants the model to capture repeat-return behavior. One or two features is plenty."*

This is real feature engineering, not "add column X": translating a behavioral theme into concrete features (trailing customer return rate, return counts in windows, multi-size order share), all **as-of checkout and excluding the current order**, plus a cold-start decision for customers with no history. The repo makes the honest path short: `feature_catalog` already contains `customer_prior_order_count`, a customer-history feature done correctly point-in-time with a colocated test, so the task is "extend the established pattern to return behavior" — exactly like extending a real codebase. The exemplar demonstrates the order-placement cutoff only; a return feature additionally needs a cutoff on return *realization* (`return_ts < checkout_ts`), which the exemplar never shows. That gap is part of the test, not an oversight. Retrain-and-eval is one `make` command; "tune" means "look at what you got and decide whether it's defensible," not a hyperparameter sweep. A strong candidate's path is roughly: read the example feature (~3 min), write 1–2 windowed features with AI (~10 min), wire into config plus a quick test (~5 min), retrain and write up the delta honestly (~5 min).

**The silent test:** Part 2 done naively violates the same principle Part 1 just taught, and there are two failure modes. The crude one is a plain groupby over the customer's full return history, which leaks current and future returns wholesale. The subtler one copies the exemplar faithfully — filters orders to those placed before checkout — but then counts those orders' *eventual* returns, leaking the realization timing the exemplar never had to handle. A candidate who fixes leakage at minute 20 and reintroduces it at minute 40 has told us something important; one who builds it point-in-time-correct has generalized the principle rather than patched the instance. The generator's serial-returner segment ensures the correct implementation earns a genuine lift (~0.84 → ~0.87 honest AUC), so the effort is rewarded with a real number they can defend. **Grading calibration:** the discriminator is *recognizing* the return-timing subtlety, not necessarily implementing it under the clock. Shipping both cutoffs is top-tier bonus; shipping a simpler version with an explicit honest caveat — "this would also need to cut returns at checkout to be serve-safe" — is a strong outcome that shows the principle generalized. The failure is shipping the leaky version confidently with no acknowledgment.

### Time budget (60 min)

- 5 min — clone, sync, orient
- ~25 min — Part 1 (the gatekeeper; strong candidates closer to 15–20)
- ~20 min — Part 2, only if offered
- ~10 min — wrap and discussion (final PR polish and submission can spill past the hour — see Ops; this slot isn't for racing a PR through)

**Part 2 is the schedule's shock absorber, not a second gatekeeper.** Like the DE design's deliberately over-budgeted Part 1, this hour breathes by treating Part 2 as pure optionality: it's offered only when Part 1 lands *decisively* with real time remaining (strong candidates finish the gatekeeper closer to 15–20 min, which is what opens the slot). The same slack absorbs setup hiccups and recovery from a wrong investigative path — a candidate who spends 35 minutes finding and fixing the leak has spent the hour exactly as intended. A *complete* Part 2 inside the slot is exceptional, not the expected outcome; it's bonus-weighted in the rubric precisely because most strong candidates won't reach a finished second feature. The interviewer must never rush a candidate into Part 2 to "fill the hour" — doing so converts the buffer into pressure and corrupts the signal the gatekeeper just produced.

## Manager task — Alex's PR + the debrief

### The mock PR

> **"Add customer behavioral features — AUC 0.84 → 0.96 🎉"**

Authored by **Alex**, the team's fictional junior DS: a ~300–400 line diff against clean `main`, enthusiastic PR description, "would love to ship this before the planning cycle." Stamped identically per candidate by `make stamp-manager-pr` from frozen commits.

### Seeded issues, layered by severity

1. **Critical — the leak.** A support-contact-derived feature (a count of the customer's contacts on the order, framed by Alex as "friction signal"), computed with no awareness that contacts mostly *post-date* checkout. Same vehicle as the senior branch's leak, by design rather than by laziness: a label-derived feature (e.g., anything aggregating `returns`) would be the one leakage pattern every AI reviewer flags confidently from the diff alone, which would make the gatekeeper AI-trivial. Support-contact timing is invisible in code; it has to be discovered in the world. We accept the cross-branch overlap — the branches are seen by disjoint candidate pools, and our leak posture already accounts for artifact exposure.
2. **Critical-adjacent — the eval quietly got weaker.** Alex switched the temporal split to random K-fold "for more stable numbers" and didn't update the model card; the 0.96 is unverifiable against the documented protocol.
3. **Moderate — test erosion, two tiers.** Alex's new feature module ships with **no tests at all** — the repo's colocated `*_test.py` convention makes the absence visible, and this is the baseline catch we expect every candidate to make. Separately, one *existing* test is now `@pytest.mark.skip("flaky after refactor")` — the advanced catch. The two-tier design acknowledges that data scientists vary widely in test culture: missing tests is table stakes; spotting a silenced regression test discriminates at the top.
4. **Moderate — judgment gaps.** The decision threshold changed in config with no business signoff; no rollout or monitoring mention anywhere in the PR.
5. **Minor — code quality.** A hardcoded path, copy-paste duplication between two features, a magic number. Real, but not ship-blocking — these exist partly to *bait misprioritization*.
6. **The red herring — which is also the genuinely good thing.** Alex's second feature is `customer_return_rate`, implemented *correctly* point-in-time: he *extended* the `customer_prior_order_count` exemplar on `main`, correctly adding the second cutoff a return-rate feature needs — counting only returns realized by checkout (`return_ts < checkout_ts`), not the orders' eventual returns. The exemplar alone only demonstrates the order-placement cutoff, so getting the return-realization one right is Alex's own judgment, not a copy — which makes it more praise-worthy, not less. The build must keep this cutoff legible: both filters plainly named, a colocated test that asserts post-checkout returns are excluded, so a candidate who actually reads the feature verifies correctness by reading rather than re-deriving it under time pressure. This weaponizes AI's most reliable review reflex: "feature aggregates the returns table" triggers a confident target-leakage flag from any model, but here the flag is wrong — distinguishing the correct implementation from leakage requires actually reading the cutoff logic (or asking Alex). A candidate who passes the AI's flag through unverified fails on false positives; one who verifies it's correct and *praises it* — the junior copied the right pattern and deserves to hear that — hits the coaching tell at the same time. The graded signal is that specific tell: praising the red herring *after* verifying it's correct. Whether the review contains any positive word at all is a weak tiebreaker, not the signal — it's gameable both ways (reflexive sandwich-praise passes it; a well-prioritized reviewer who spent every minute on the leak may earn it and never compliment).

### Why a naive "review this PR" prompt doesn't crack it

We have to be honest about what AI review finds: the K-fold switch, the missing tests, and the code nits are diff-visible, and a good model will surface them. The design is fine with that, because the moat is elsewhere:

**The critical issue is not confirmable from the diff.** This constrains how we write Alex's code: the leaky feature must look innocent as text — a plain count of rows joined from `support_contacts`, with **no timestamp logic visible anywhere in the diff**. The leak lives in the data's semantics (contacts mostly post-date checkout), not in the code. An AI reviewing the diff can at best hedge — "verify this table doesn't include post-order events" — and the rubric grades exactly the gap between a hedge passed through and a hedge converted to evidence: one data query showing ~90% of contacts post-date checkout, an ablation retrain showing AUC falls back to ~0.85 without the feature, or simply asking Alex what's in that table. Meanwhile the red herring inverts the trap: the feature AI *confidently* flags is the correct one, so a paste-through review gets the leak as a hedge and the non-leak as its headline — exactly backwards. Exact visibility tuning happens at build time against the two-sided gate (see the acceptance test below and the red-team pass in the Build TODO) — a naive "review this PR" prompt must fail the rubric, while a strong audit-the-data prompt must surface the leak, and we tune to the gap between them. One path to check there: `main`'s as-of exemplar makes "Alex's contact feature doesn't follow the established pattern" a legitimate find route — acceptable, since it still requires repo context and confirmation, but worth verifying it doesn't make naive prompts succeed.

**The 0.96 is unverifiable without leaving the diff.** Knowing the claim doesn't hold requires running `make eval` against the documented protocol or noticing the model-card mismatch — repo context a paste-the-diff workflow never touches.

**Generic AI output has the wrong shape, and the rubric punishes the shape.** "Review this PR" produces 15–20 hedged findings, severity-inflated, red herring dutifully flagged, no verdict. We grade prioritization (the leak as the headline), false-positive restraint, and a committed approve/request-changes verdict with ship-blocking-vs-follow-up triage. A candidate who pastes AI output verbatim "finds" everything and fails three dimensions.

**Alex and the debrief are un-AI-able live.**

**Build-phase acceptance test:** before the mock PR ships, we run naive "review this PR" prompts against the diff with current frontier models and tune until the output is a mediocre hedged laundry list. If a one-line prompt produces a review that would pass our rubric, the PR goes back to the shop. This is the manager-side equivalent of the DE design's dry-run requirement.

### The hour

- 10 min — orient (repo access at start; the PR link is the starting point). Orient gets real time on purpose: the red herring can only be judged by understanding the `customer_prior_order_count` exemplar on `main` — repo archaeology, not diff-reading — and a 5-minute orient quietly reintroduces the repo-navigation-speed axis this design explicitly disclaims testing.
- ~35 min — review, submitted as a real GitHub review: inline comments plus a summary with an explicit verdict. **Alex is on Slack** (interviewer in-character): eager, slightly defensive, honest about what's asked — *"did you check what's in that support table?" → "oh — it's whatever the support export has, I just joined on order_id."* Reaching for the author before assuming is the asks-vs-assumes signal, manager-flavored.
- ~15 min — **the debrief** (interviewer drops persona), questions ranked so an interviewer running long drops from the bottom: first and load-bearing, the process-change question — *"This nearly shipped — what process change prevents the next one?"*; second, coaching delivery — *"Walk me through delivering this feedback without crushing Alex."*; third, triage — *"What's ship-blocking vs follow-up?"* The review tests judgment; the debrief tests whether they think in systems (eval standards, CI gates, model-card discipline — note `CONTRIBUTING.md` already hints at these, so "enforce what's written" beats "invent process from scratch") and in people (is Alex's mistake a firing offense, a coaching moment, or a process failure?).

## Stakeholder character briefs (interviewer-only docs)

Both briefs live as standalone interviewer docs, same as Jamie's in the DE design. Each gives the character, the facts they know, and one operating rule: answer any reasonable question honestly, but never volunteer the load-bearing facts or their implications — for Priya, when scoring happens and where the prod AUC comes from; for Alex, the support-table timing. Beyond that, interviewer judgment governs. These interviews are an art, not a science — exact repeatability across sessions isn't the goal and isn't realistic. The character descriptions and example answers below are illustrations of voice and posture, not scripts to recite.

**Priya — DS lead / product analytics (senior session).** Knows the product cold, knows the model's history shallowly ("the v1.3 features came from the contractor sprint"). Answers definitional questions crisply: *"When do we score?" → "Synchronously at checkout."* *"Where does the prod AUC come from?" → "Weekly monitoring job, labels realized at 60 days."* Will not volunteer either fact unasked.

**Alex — junior DS, PR author (manager session).** Eager, proud of the result, slightly defensive but honest. Answers factual questions accurately; does not understand the implications of those facts until walked through them. If the candidate explains the leak well, Alex gets it — the persona rewards good coaching with visible comprehension.

## Internal rubrics

Same structure as the DE rubric: implementation correctness is necessary but not sufficient; a gatekeeper criterion floors the recommendation.

**Senior:**

1. **Diagnosis (gatekeeper)** — correctly identifies the leakage *with its mechanism* (train/serve skew: feature unavailable at prediction time). Below bar → no hire regardless of everything else.
2. **Fix quality** — point-in-time re-derivation > removal > partial fix; did they catch both layers (feature and split)?
3. **Eval honesty** — temporal split restored, model card updated, the 0.92 → 0.62 story explained rather than papered over with a new offline number.
4. **Part 2 engineering** (bonus-weighted, since optional) — theme translated into defensible features, point-in-time discipline maintained, cold-start handled deliberately.
5. **PR quality** — coherent narrative, scoped diff, regression-relevant tests.

Behavioral (interviewer-observed): **AI-prompting maturity** (explore-then-act vs fix-this-and-watch), **asks vs assumes** (did they establish — or empirically verify — when prediction happens?), **verification** (did they reconcile against the prod story, or trust the new offline number?).

**Manager:**

1. **Found the leak AND it's the headline (gatekeeper)** — finding it but burying it under style nits fails prioritization, which for a manager is the same as missing it.
2. **Severity triage** — ship-blocking vs follow-up, explicit verdict.
3. **Verification** — converted hypotheses to evidence (ran the eval, queried the data, ablated, or extracted the key fact from Alex).
4. **False-positive restraint** — the red herring.
5. **Review tone / coaching** — would Alex come out of this review better and still motivated? The graded tell is verified praise of the red herring (confirmed correct, then credited); bare positive-word presence is only a weak tiebreaker.
6. **Debrief: systems and people** — process changes that map to the repo's existing (unenforced) norms; a humane, concrete plan for the Alex conversation.

Behavioral: **AI usage** (did they direct AI to verify claims — run things — or only to read the diff?), **asks vs assumes** (did they use Alex?).

## Ops

- **Pre-interview:** candidate confirms Python + `uv` + `gh` auth + their AI tooling *works* — not merely exists — by running the checklist against a setup-only smoke-test repo (same dependency set, no task content), which proves clone + `uv sync` + their AI tooling on a real repo and warms the cache without exposing `senior/start` or the mock PR. Their AI tooling must be filesystem- and exec-capable (Claude Code, Cursor with a terminal, etc.); a chat-only paste-the-diff workflow cannot produce the explore-the-repo / run-the-eval behaviors the rubrics grade, so this is a fairness prerequisite verified at the setup check, not a preference. Repo access granted at interview start. If a candidate's environment breaks at interview start anyway, fall back to screen-sharing the interviewer's pre-built checkout rather than burning the hour debugging.
- **Session start:** the interviewer states the time box, that AI tooling is expected, the deliverable (senior: a PR; manager: a GitHub review with an explicit verdict), and that Priya/Alex is available on Slack — *"use them as you would a colleague."* Disclosing the channel exists is required for the asks-vs-assumes signal to be fair; what the persona would reveal stays undisclosed. **Never run both tracks on the same candidate** — a re-slot between Sr and Manager means the other track's leak is already spoiled.
- **Per-interview, senior:** candidate branches from `senior/start`, opens a PR against it. Final PR polish and submission may happen after the hour ends — they don't eat interview time, and this is low-risk because the interviewer observed the entire working session live. Close PRs after grading.
- **Per-interview, manager:** run `make stamp-manager-pr CANDIDATE=<name>` before the session; close the PR and delete the branch immediately after. Grade from the submitted review before closing.
- **Interviewer kit:** run sheet per role, rubric per role, both persona briefs, and answer keys **emitted by the generator and stamping scripts** — every ground-truth number and every seeded-issue location regenerates with the repo.
- **Nudge protocol:** each role gets at most **one** scripted in-character nudge (Priya for the senior, Alex for the manager), delivered at a defined timestamp only if the candidate hasn't engaged the load-bearing thread by then — for the senior, the prediction-time boundary; for the manager, the leak. The budgets are tight enough that a drowning candidate has no slack to recover into, so this is a decided protocol rather than a post-launch experiment. Exact wording and trigger time are finalized during dry runs.

## Open design questions (parked)

- **Nudge wording and timing** — the one-nudge-per-role protocol is decided (see Ops); the remaining open item is the exact in-character wording and the trigger timestamp for each role, finalized during dry runs.
- **Part 2 delivery** — same PR or second PR? Leaning same-PR for simplicity; decide at build time.
- **Senior fork-vs-branch** — branches in the upstream repo are simplest given access is granted at interview start; revisit if repo permissions get awkward.

## Build TODO

- Build the clean repo on `main`: feature catalog (including the correct point-in-time example feature + test), `models/return-risk/` with train/eval/predict, `MODEL_CARD.md`, `CONTRIBUTING.md`, Makefile
- Build the setup-only smoke-test repo (same dependency set, no task content) the pre-interview checklist runs against — proves clone + `uv sync` + AI tooling without exposing any task material
- Build `data/generate.py`: four CSVs, deterministic seed, planted support-contact timing + serial-returner segment + temporal drift, emits ground-truth numbers and answer keys. Two acceptance assertions the generator must satisfy: (a) **pre-checkout** support contacts carry a modest genuine predictive lift — the best-tier fix re-derives the feature point-in-time on the premise that pre-purchase friction is real signal, so if pre-checkout contacts are noise the top rubric tier collapses into "just delete it"; emit the re-derived feature's ground-truth lift into the answer key. (b) A single-seed co-occurrence check: every narrative number must hold *simultaneously* at the pinned seed — senior side 0.92 leaky-offline / ~0.87 temporal-split / ~0.84 honest / 0.62 prod, the manager ~0.85 ablation, the Part 2 ~0.84 → ~0.87 lift. Each is individually plausible but they're never jointly asserted; verify they co-occur and template them into the answer keys.
- Author the senior commit series (v1.3 features incl. `support_contact_count_30d`, eval-split switch, confident commit messages) as frozen patches + `make stamp-senior`
- Author the Alex PR commit series (support-contact leak with no visible timestamp logic, correct-but-suspicious `customer_return_rate` as the red herring — implemented with both cutoffs (order placement AND return realization pre-checkout) and a colocated test making correctness legible, K-fold switch, missing tests + one skip, threshold change, code nits) + `make stamp-manager-pr`
- Render tickets (RISK-412, the serial-returner theme ticket) from generator output
- Author interviewer docs: run sheets, rubrics, Priya + Alex briefs
- **AI red-team pass on both tasks** — two-sided gate: naive prompts ("review this PR", "fix this model") must produce a hedged laundry list that fails the rubric; a strong generic orchestration prompt ("explore the repo, audit every feature for point-in-time correctness, verify eval claims against the data, then review") must surface the leak. Tune to the gap between those two prompts — the eval must reward verification skill, not just knowing to ask the persona. This tuning happens empirically against real AI agents during the build, not a priori.
- Internal dry-run, one per role, before going live
