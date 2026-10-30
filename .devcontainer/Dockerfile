FROM mcr.microsoft.com/devcontainers/python:3-3.12-trixie

# LaTeX for building the proposal/report (LaTeX Workshop calls latexmk)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        latexmk \
        texlive-latex-extra \
        texlive-fonts-recommended \
        lmodern \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*