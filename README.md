# CSE-250A Project

Final project for **CSE 150A/250A: Principles of Artificial Intelligence: Probabilistic Reasoning and Learning** (UCSD, Fall 2026).

> **Topic:** _TBD — pending group vote._ One-paragraph summary of the problem, dataset, and model (BN / HMM / MDP) goes here once chosen.

| Milestone | Due |
|---|---|
| Proposal | Mon Oct 26, 11:59 PM PT |
| Final submission | Mon Nov 23, 11:59 PM PT |
| Resubmission (optional) | Fri Dec 11, 2:59 PM PT |

---

## Quick start

### Prerequisites
- Docker
- An editor that supports [Dev Containers](https://containers.dev/supporting) (e.g. VS Code with the Dev Containers extension), or GitHub Codespaces

Everything else (Python, uv, dependencies) is installed inside the container.

### Open the project
Clone `https://github.com/kberesfo/CSE-250A-Project.git` and open it in a dev container (in VS Code: **Reopen in Container**). The first build takes a few minutes; it runs `uv sync --locked` to create `.venv/` with every dependency at the locked version.

### Verify
```bash
uv run cse250a --help
```
In a notebook, choose **Select Kernel → `.venv`**.

---

## Project layout

```
CSE-250A-Project/
├── .devcontainer/            # shared container config
├── .github/                  # issue/PR templates, CI
├── configs/                  # experiment configs (later)
├── data/                     # committed datasets
├── figures/                  # generated plots for the report
├── notebooks/                # exploration + figures only
├── results/                  # run outputs (gitignored)
├── scripts/                  # one-off utilities
├── src/cse250a/              # all reusable code
│   ├── cli.py                # experiment runner (`uv run cse250a`)
│   └── modules/
│       └── config/           # pydantic-settings config
├── pyproject.toml
└── uv.lock
```

**Rule of thumb:** if a function is used in more than one place, it belongs in `src/cse250a/`, not in a notebook.

---

## Configuration

Settings live in `src/cse250a/modules/config/` and are exported as one object:

```python
from cse250a.modules.config import config

config.paths.data_dir      # Path
config.paths.results_dir
config.paths.figures_dir
config.seed                # int
```

Every value has a default. Override any of them with an environment variable or a line in a gitignored `.env` file at the repo root:

| Variable | Default |
|---|---|
| `DATA_DIR` | `<repo>/data` |
| `RESULTS_DIR` | `<repo>/results` |
| `FIGURES_DIR` | `<repo>/figures` |
| `SEED` | `0` |

```bash
SEED=3 uv run cse250a run --model hmm
```

---

## Running experiments

```bash
uv run cse250a run --model hmm --seed 1
```

Each run writes to `results/<name>_<timestamp>/` with the resolved `config.json` alongside its outputs, so every result records exactly how it was produced.

---

## Notebooks

Start each notebook with:
```python
%load_ext autoreload
%autoreload 2

from cse250a.modules.config import config
```
`autoreload` picks up edits to `src/` without restarting the kernel.

- Name notebooks with a number prefix: `01_explore.ipynb`, `02_em_baseline.ipynb`.
- Outputs are stripped automatically on commit by `nbstripout`, so diffs stay readable. Save figures you need to `figures/`.
- Progress bars: `from tqdm.auto import tqdm`.

---

## Dependencies

Managed with [uv](https://docs.astral.sh/uv/). Don't use `pip install`.

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # dev-only tool
uv remove <package>
```
Commit both `pyproject.toml` and `uv.lock`. Run all Python inside the container.

---

## Workflow

We use **GitHub Flow**: `main` is protected and always works.

1. Pick or create an issue (templates: **Task**, **Experiment**, **Bug**) and assign yourself.
2. Branch from `main`: `feat/<short-name>`, `data/<short-name>`, `exp/<short-name>`, `fix/<short-name>`.
3. Commit, push, and open a PR. Link the issue with `Closes #N`.
4. Get **1 approval**, resolve all review threads, then **squash merge**. The branch is deleted automatically.

```bash
git switch main && git pull
git switch -c feat/em-cpt-learning
# ...work...
git add -A && git commit -m "Add EM for CPT learning"
git push -u origin feat/em-cpt-learning
gh pr create --fill
```

**Async coordination**
- **Project board:** who's doing what (Todo, In progress, In review, Done).
- **Milestones:** issues are grouped by Proposal, Final, and Resubmission.
- **Experiment issues:** record the question, the exact command, and the results. These feed the report's comparison section.
- **`docs/decisions.md`:** one line per design decision (date, decision, why). This feeds the report's justification and reflection sections.

---

## Course policy reminders

- Code must be **Python**.
- **Cite** any open-source code you adapt, in a comment at the top of the file and in the report.
- **Disclose** any Generative AI use, as required by the course guidelines.
- Every member writes a contribution statement. Issue assignments and PR authorship make this easy.

---

## Data sources

_Fill in once the dataset is chosen: name, source URL, license, and required citation._

---

## Team

| Name | GitHub |
|---|---|
| Kiernan Beresford | kberesfo |
| Zachary Brown | zacklb2 |
| Joyce Yang | JoyGraceYang |
| Muska Mesdaq  | mmmesdaq-spec|
| Agnes Tang  | alybss135 |

## License

See [LICENSE](LICENSE).