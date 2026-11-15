# Clean runs of the notebooks → outputs saved in the notebook + HTML reports in results/.
#
#   make               run every phase and study notebook on a fresh kernel
#   make phase3b       just one group (any folder name: phase1, phase3b, studies, ...)
#   make -j3           run notebooks in parallel (each one is independent)
#   make keep-outputs  mark every phase/study notebook so nbstripout keeps its outputs
#   make list          show the reports that would be built
#   make clean         delete the generated HTML
#
# Each run writes its outputs back into the notebook (committed, renders on GitHub)
# and saves results/<group>/<name>.html. A report is rebuilt when its notebook or
# anything in src/ changes.

NOTEBOOKS := $(sort $(wildcard notebooks/phase*/*.ipynb notebooks/studies/*.ipynb))
REPORTS   := $(patsubst notebooks/%.ipynb,results/%.html,$(NOTEBOOKS))
GROUPS    := $(sort $(notdir $(patsubst %/,%,$(dir $(NOTEBOOKS)))))
SRC       := $(shell find src -name '*.py')

KEEP    := uv run python scripts/keep_outputs
EXECUTE := uv run jupyter nbconvert --to notebook --execute --inplace \
	       --ExecutePreprocessor.timeout=-1
TO_HTML := uv run jupyter nbconvert --to html

.PHONY: all clean list keep-outputs $(GROUPS)

all: $(REPORTS)

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

keep-outputs:
	$(KEEP) $(NOTEBOOKS)

list:
	@printf '%s\n' $(REPORTS)

clean:
	rm -f $(REPORTS)