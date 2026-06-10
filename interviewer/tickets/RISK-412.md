# RISK-412 — Return-risk model underperforming in prod

**Priority:** High · **Reporter:** Priya N. (DS Lead) · **Assignee:** you

Prod numbers for the return-risk model are coming in below what we shipped
on. Offline AUC was 0.93 when v1.3 shipped; prod is tracking
~0.81 over the last 8 weeks. It doesn't look like ordinary drift.
Can you dig into why offline and prod diverge and get us a number we can
plan on?

*Deliverable: a PR against `senior/start` with the fix.*
