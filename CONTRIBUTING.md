# Contributing

How we work on this repo. Read it once; skim it again before your first PR.

## First-time setup

1. Accept the collaborator invite on GitHub.
2. Clone the repo and open it in the dev container (see [README](README.md#quick-start)).
3. Make sure git knows who you are. If `git commit` says *Author identity unknown*, set it on your host machine and rebuild the container:
   ```bash
   git config --global user.name  "Your Name"
   git config --global user.email "you@users.noreply.github.com"
   ```
   Your GitHub no-reply address is under GitHub → Settings → Emails. Since this repo is public, prefer it over a personal email.

## Workflow at a glance

```
issue → branch → commits → PR → 1 review → squash merge → branch auto-deleted
```

`main` is protected: no direct pushes, no force pushes. Everything goes through a PR.

## 1. Pick up work

- Every piece of work starts as an **issue**. Use a template:
  - **Task**: something to build or write.
  - **Experiment**: a configuration to run and compare.
  - **Bug**: something broken.
- **Assign yourself** before you start, so two people don't do the same thing.
- Set the **milestone**: Proposal, Final, or Resubmission.
- Move the card on the **Project board** as it progresses.

If you're blocked, add the `blocked` label and say why in a comment.

## 2. Branch

Always branch from an up-to-date `main`:

```bash
git switch main && git pull
git switch -c feat/em-cpt-learning
```

| Prefix | Use for |
|---|---|
| `feat/` | new code in `src/` |
| `data/` | loading, cleaning, preprocessing |
| `exp/` | running experiments, results, figures |
| `docs/` | README, report text, decisions log |
| `fix/` | bug fixes |

Keep branches small and short-lived, ideally merged within a few days. Small PRs get reviewed faster and conflict less.

## 3. Commit

```bash
git add -A
git commit -m "Add forward-backward for HMM"
```

- Write messages as imperative and specific: "Add EM for CPTs", not "updates" or "stuff".
- Commit as often as you like. PRs are squash-merged, so messy WIP commits never reach `main`.
- To pick up teammates' changes mid-branch:
  ```bash
  git fetch origin && git merge origin/main
  ```

## 4. Open a pull request

```bash
git push -u origin feat/em-cpt-learning
gh pr create --fill        # or use the GitHub web UI
```

- Fill in the PR template, and write `Closes #N` to link the issue.
- Open a **draft PR** early if you want feedback before it's finished.
- CI runs `ruff` and checks that notebooks have no outputs. Fix any failures before requesting review.

## 5. Review

**As an author:** request a reviewer, and respond to every comment, either by changing the code or by explaining why not.

**As a reviewer,** aim to review within a day. Check that:
- [ ] the code does what the issue asked
- [ ] it runs (pull the branch and try the "How to check" step)
- [ ] reusable logic is in `src/`, not stuck in a notebook
- [ ] there are no hardcoded paths or magic numbers without explanation

Approve when it's good enough. It doesn't have to be perfect. Leave small nits as suggestions, not blockers.

## 6. Merge

Once approved and all conversations are resolved, click **Squash and merge**. The branch deletes itself. Then:

```bash
git switch main && git pull
```

---

## Code conventions

- **`src/` vs. notebooks:** if a function is used in more than one place, it belongs in `src/cse250a/`. Notebooks are for exploration and figures.
- **Paths:** never hardcode them. Use `config.paths.data_dir` and the related settings in `cse250a.modules.config`.
- **Randomness:** seed everything from `config.seed`, so results are reproducible.
- **Style:** run `uv run ruff check .` and `uv run ruff format .` before pushing.
- **Docstrings:** give any non-obvious function a one-line docstring, especially math. Note the equation or lecture it implements.

## Notebooks

- Name them with a number prefix and your initials if it's personal exploration: `03_kb_hmm_states.ipynb`.
- Don't edit someone else's notebook in your PR. Copy it or ask them first. Notebook merge conflicts are painful.
- Outputs are stripped on commit. Save any figure you want to keep to `figures/` with `fig.savefig(...)`.

## Dependencies

- Add with `uv add <pkg>` (or `uv add --dev <pkg>` for tools), never `pip install`.
- Commit `pyproject.toml` **and** `uv.lock` together.
- If a teammate's PR added a dependency, run `uv sync` after pulling.

## Data

- Datasets are committed in `data/`. **Don't modify raw files.** Write cleaned versions to a separate file or folder and make the cleaning code reproducible.
- Before adding a dataset, check that its license allows public redistribution, since this repo is public. Record the source, license and citation in the README's **Data sources** section.
- Nothing private or personal goes in the repo, including notebook outputs.

## Experiments

- Open an **Experiment** issue first, with the question, the exact command and the metric.
- Run it with `uv run cse250a run ...`. The resolved config is saved alongside the results.
- Post the numbers, the `results/` folder name and any figures back to the issue. These become the report's comparison section.

## Decisions

When the group settles a design choice (model structure, binning, priors, metrics), add one line to `docs/decisions.md`:

```
2026-10-20 — Discretize sensors into 4 quantile bins — keeps CPTs small enough for EM with our sample size.
```

This saves re-debating, and feeds the report's justification and reflection sections.

## Course policy

- All code must be **Python**.
- **Cite** any adapted open-source code in a comment at the top of the file, and tell the group so it makes the report.
- **Disclose** Generative AI use as the course guidelines require. Mention it in your PR description so we can compile it for the report.
- Contribution statements are required. Your issues, PRs and reviews are the record, so do your work through them.

## Questions

Ask in the team chat or open an issue. If something in this guide is wrong or annoying, open a `docs/` PR to fix it.