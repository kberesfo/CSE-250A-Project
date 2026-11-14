```bash
git checkout main && git pull
git fetch --prune                                              # drop origin/<branch> refs that no longer exist
git branch -vv | grep ': gone]' | awk '{print $1}'             # dry run: list local branches whose remote is gone
git branch -vv | grep ': gone]' | awk '{print $1}' | xargs -r git branch -D
```

```bash
 uv run jupyter nbconvert --to html --execute notebooks/phase3b/eval.ipynb     --output-dir results/
```