# Clean runs of the notebooks → outputs saved in the notebook + HTML reports in results/.
#
#   make                      run every phase and study notebook on a fresh kernel, in order
#   make phase3b              one group, after any earlier phase that is out of date
#   make rerun                rebuild everything and recompute cached tables (RERUN=true)
#   make -B phase3b RERUN=true   rebuild one group and recompute its cached tables
#   make open                 open results/index.html (links to every report)
#   make keep-outputs         mark every phase/study notebook so nbstripout keeps its outputs
#   make list                 show the reports that would be built
#   make clean                delete the generated HTML
#   make clean-results        delete the generated CSVs in results/ (rebuilt on the next run)
#
# Each run writes its outputs back into the notebook (committed, renders on GitHub)
# and saves results/<group>/<name>.html. A report is rebuilt when its notebook,
# anything in src/, or an earlier phase changes. Notebooks run one at a time,
# phase1 → phase2 → phase3a → phase3b, because later phases read earlier results.

NOTEBOOKS := $(sort $(wildcard notebooks/phase*/*.ipynb notebooks/studies/*.ipynb notebooks/final/*.ipynb))
REPORTS   := $(patsubst notebooks/%.ipynb,results/%.html,$(NOTEBOOKS))
GROUPS    := $(sort $(notdir $(patsubst %/,%,$(dir $(NOTEBOOKS)))))
SRC       := $(shell find src -name '*.py')

KEEP    := uv run python scripts/keep_outputs
EXECUTE := uv run jupyter nbconvert --to notebook --execute --inplace \
	       --ExecutePreprocessor.timeout=-1
TO_HTML := uv run jupyter nbconvert --to html

# Notebooks read config.rerun from the RERUN environment variable.
RERUN ?= false
export RERUN

# one notebook at a time, even with -j
.NOTPARALLEL:

.PHONY: all clean clean-results rerun open list keep-outputs $(GROUPS)

all: $(REPORTS) results/index.html

# make phase1, make studies, ...: one target per notebook folder
define GROUP_RULE
$(1): $(patsubst notebooks/%.ipynb,results/%.html,$(wildcard notebooks/$(1)/*.ipynb))
endef
$(foreach g,$(GROUPS),$(eval $(call GROUP_RULE,$(g))))

results/%.html: notebooks/%.ipynb $(SRC)
	@mkdir -p $(@D)
	$(KEEP) $<
	$(EXECUTE) $<
	$(TO_HTML) $< --output-dir $(@D) --output $(notdir $(basename $@))

# phase order: each phase runs after the one before it (3a reads Phase 2's grid)
results/phase2/eval.html: results/phase1/eval.html
results/phase3a/eval.html: results/phase2/eval.html
results/phase3b/eval.html: results/phase3a/eval.html
# the final test run goes last (order only: it reads nothing from the phases)
results/final/eval.html: | results/phase3b/eval.html

# one page linking every report: open results/index.html in a browser
results/index.html: $(REPORTS)
	@{ echo '<!doctype html><title>Reports</title><h1>Reports</h1><ul>'; \
	  for r in $(patsubst results/%,%,$(REPORTS)); do \
	    echo "<li><a href=\"$$r\">$${r%.html}</a></li>"; done; \
	  echo '</ul>'; } > $@

open: results/index.html
	@python -m webbrowser -t "file://$(abspath results/index.html)"

keep-outputs:
	$(KEEP) $(NOTEBOOKS)

list:
	@printf '%s\n' $(REPORTS)

clean:
	rm -f $(REPORTS) results/index.html

clean-results:
	find results -name '*.csv' -delete

rerun:
	$(MAKE) -B all RERUN=true