```bash
git checkout main && git pull
git fetch --prune                                              # drop origin/<branch> refs that no longer exist
git branch -vv | grep ': gone]' | awk '{print $1}'             # dry run: list local branches whose remote is gone
git branch -vv | grep ': gone]' | awk '{print $1}' | xargs -r git branch -D
```

```bash
 uv run jupyter nbconvert --to html --execute notebooks/phase3b/eval.ipynb     --output-dir results/
```

```bash
uv run ruff check --select I --fix notebooks/   # sort imports
uv run ruff format notebooks/                   # format every code cell
uv run ruff check notebooks/                    # check what's left
```

```bash
uv run python -m http.server 8000 --directory results
```

```bash
make clean clean-results   # delete all HTML, the index and all CSVs
make rerun                 # rerun every notebook in order, recomputing cached tables
```