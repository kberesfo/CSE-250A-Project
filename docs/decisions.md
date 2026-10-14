# Decisions

One line per decision: date — decision — why. Newest at the bottom.

- 2026-10-12 — Dev container + uv for the environment — identical Python/dependencies on any host; `uv.lock` pins versions.
- 2026-10-12 — GitHub Flow: protected `main`, PRs with 1 review, squash merge — simple enough for a 6-week project, keeps `main` working and history readable.
- 2026-10-12 — Commit datasets to `data/` (when small and license allows) — no per-person download/setup; everyone has identical data.
- 2026-10-14 — Project: UCI Air Quality (Bayes Net + EM / HMM, true vs. sensed pollution) — won round-2 ranked-choice vote outright (4/5 first-choice, beat every option head-to-head). Backup: Adaptive Tutoring (ASSISTments).