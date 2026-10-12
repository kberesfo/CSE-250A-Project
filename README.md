# CSE-250A-Project
```
cse-250A-project/
├── .devcontainer/
├── data/                  # committed datasets
├── notebooks/             # exploration + figures only
│   └── 01_explore.ipynb
├── src/cse250a/           # all reusable code lives here
│   ├── __init__.py
│   ├── paths.py           # DATA_DIR, FIGURES_DIR, RESULTS_DIR
│   ├── data.py            # load + preprocess functions
│   ├── models/            # BN / HMM / MDP code
│   └── cli.py             # the runner
├── scripts/
├── results/               # gitignored run outputs
├── pyproject.toml
├── uv.lock
└── .venv/                 # created by uv sync, gitignored
```