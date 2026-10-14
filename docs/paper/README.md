# paper/

LaTeX sources for the proposal and final report.

```
paper/
├── common/
│   ├── preamble.tex      # shared packages + macros (\todo, notation)
│   └── authors.tex       # shared author block
├── references.bib        # shared bibliography
├── proposal/             # 1 page, due Mon Oct 26
│   ├── main.tex
│   └── sections/  problem · dataset · methodology
└── report/               # ≤ 7 content pages, due Mon Nov 23
    ├── main.tex
    └── sections/  01_problem … 06_reflections · appendix
```

Each `main.tex` holds the title and `\input`s its sections, so people can work on different section files without merge conflicts.

## Build

```bash
cd paper/report     # or paper/proposal
latexmk -pdf main.tex
latexmk -c          # clean build files
```

This needs a TeX install (`latexmk`, TeX Live), or use Overleaf: upload the `paper/` folder and set the main document to `report/main.tex`.

## NeurIPS style (recommended by the course)

Download the NeurIPS style file (from the NeurIPS site or its Overleaf template) and put `neurips_2026.sty` (or `neurips_2025.sty`) in `paper/common/`. The preamble picks it up automatically. Without it, documents use a plain 1-inch-margin layout, so everything compiles either way.

## Conventions

- **Draft notes:** `\todo{...}` renders in red. Search for `\todo` before submitting.
- **Figures:** save from code to the repo's `figures/` folder, then `\includegraphics{name.png}`. The path is already set.
- **Citations:** add entries to `references.bib`; cite with `\citep{key}` or `\citet{key}`. Remove `\nocite{*}` from `main.tex` once sections cite their sources.
- **Cross-references:** `\label{fig:x}` + `\cref{fig:x}` gives "Figure 1" automatically.
- **Page limit (report):** 7 pages for Sections 1–6. References and appendix are extra.
