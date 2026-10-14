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

`cse250a` is installed by `uv sync` from `[project.scripts]` in `pyproject.toml`, so there's no separate run script. `python -m cse250a ...` works too. Add new options in `src/cse250a/cli.py`. For sweeps over many configurations, loop over `uv run cse250a run ...` in a script under `scripts/`, or add a subcommand.

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

## Contributing

`main` is protected; all changes go through a reviewed PR. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the workflow, branch naming, code conventions, and course policy reminders.

---

## Data sources

**UCI Air Quality**: 9,358 hourly records (Mar 2004 – Feb 2005) from a multisensor device in an Italian city, with reference analyzer measurements of CO, NMHC, benzene, NOx, NO2, plus temperature and humidity. Missing values are coded as `-200`.
- Source: https://archive.ics.uci.edu/dataset/360/air+quality
- License: CC BY 4.0
- Citation: Vito, S. (2008). *Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F
- Paper: De Vito, S., Massera, E., Piga, M., Martinotto, L., & Di Francia, G. (2008). On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario. *Sensors and Actuators B: Chemical*, 129(2), 750–757.

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