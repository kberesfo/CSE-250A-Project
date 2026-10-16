# Dev Containers: A Quick Primer

New to dev containers? Read this once before setting up. It takes about 5 minutes.

## What is it?

A **dev container** is a Docker container configured as a full development environment. The repo's `.devcontainer/devcontainer.json` describes it: base image, Python version, tools, VS Code extensions, and setup commands.

When you click **Reopen in Container**, VS Code builds that environment and connects to it. You still edit files in VS Code as usual, but the terminal, Python, and every tool run **inside the container**, not on your machine. Your project folder is shared between the two, so edits show up in both places instantly.

```
Your machine (any OS)                  Container (Linux)
┌──────────────────────┐              ┌─────────────────────────────┐
│ VS Code (the window) │ ───────────▶ │ Python 3.12, uv, .venv      │
│ Docker               │              │ git, gh, ruff, nbstripout   │
│ repo folder ◀────────┼── shared ──▶ │ /workspaces/<repo>          │
└──────────────────────┘              └─────────────────────────────┘
```

## The problem it solves

Group projects often lose days to "it works on my machine":
- Different Python versions (3.10 vs. 3.12) behave differently.
- Different package versions produce different results, or don't install at all.
- Windows, macOS, and Linux each need different setup steps.
- Someone is missing a tool, and the README's setup steps drifted out of date.
- Packages from other classes leak in and conflict.

With a dev container, the environment is defined **in the repo**, so everyone gets the same one.

## Pros

- **Consistent environment.** Same OS, Python, packages, and tools for everyone. When code works for one person, it works for all.
- **All tools included.** Python, uv, git, the GitHub CLI, ruff, and Jupyter support come pre-installed. Nothing to install by hand.
- **One-step setup.** Clone, then **Reopen in Container**. `uv sync` runs automatically.
- **Reproducible.** `uv.lock` pins exact package versions. Graders, or future you, can rebuild the same environment.
- **Isolated.** Nothing is installed on your machine, so this project can't break your other classes' setups, and vice versa.
- **Disposable.** If something gets weird, **Rebuild Container** gives you a fresh environment in minutes.
- **Works in the cloud too.** The same config runs in GitHub Codespaces, in a browser or VS Code, with no local install. It's free for verified students. See *No Docker? Use GitHub Codespaces*.

## Cons

- **Extra software on your machine.** You need Docker, which takes some setup, especially on Windows. See *Host requirements* below, or skip it entirely with Codespaces.
- **Resource overhead.** Docker uses RAM, CPU, and several GB of disk. It can feel heavy on an 8 GB laptop.
- **Slow first build.** The first build downloads the image and installs packages, which takes several minutes. Later starts are fast.
- **Another layer to understand.** "Is this on my machine or in the container?" can be confusing at first. Some errors, such as Docker not running or a mount failing, come from the container layer.
- **Editor lock-in, mostly.** It's best in VS Code. Other editors need extra setup or don't support it.
- **Host settings don't all carry over.** Your git name and email are copied in automatically, but other personal configuration may not be. See *Troubleshooting*.

**Bottom line:** a short setup cost up front, in exchange for never debugging each other's environments.

## Host requirements

| You need | Notes |
|---|---|
| **Docker** | [Docker Desktop](https://docs.docker.com/desktop/) on Windows or macOS, or Docker Engine on Linux. |
| **VS Code** | Plus the **Dev Containers** extension (`ms-vscode-remote.remote-containers`). |
| **Git** | To clone the repo. Set your name and email (see *Troubleshooting*). |
| **~8 GB RAM, ~10 GB free disk** | Docker Desktop's stated minimum is 8 GB of RAM. |

**Windows:** Docker Desktop runs on WSL 2, and needs hardware virtualization enabled in BIOS/UEFI. See Docker's [Windows install page](https://docs.docker.com/desktop/setup/install/windows-install/) for supported Windows editions. For speed, clone the repo inside your WSL home folder (e.g. `~/dev/`), not on `C:\`.

## No Docker? Use GitHub Codespaces

**Codespaces** runs this same dev container on GitHub's servers. You get the identical environment in a browser tab, or in desktop VS Code, with **nothing installed locally**: no Docker, no setup, any laptop, even a Chromebook or a lab machine.

**It's free for us as students.** Verified students get free Codespaces use, up to **180 core-hours/month** with storage equivalent to GitHub Pro (20 GB). Verify through the [GitHub Student Developer Pack](https://education.github.com/pack) with your UCSD email. Without student verification, free accounts get 120 core-hours and 15 GB.

Usage is counted in *core*-hours, so the default 2-core machine uses 2 per hour. 180 core-hours is about **90 hours of actual use per month**, which is plenty for this project.

**Start one:** on the repo page, click **Code → Codespaces → Create codespace on main**. The first start builds the container, which takes a few minutes. After that it resumes in seconds.

**Tips**
- **Stop it when you're done** (**Code → Codespaces → … → Stop codespace**). Idle codespaces stop on their own after 30 minutes by default, but stopping manually saves hours.
- **Push your work before deleting a codespace.** Files live in the codespace, not on your machine.
- **Open it in desktop VS Code** for a nicer experience: **Code → Codespaces → … → Open in VS Code Desktop**. This needs the *GitHub Codespaces* extension.
- **Check your usage** at GitHub → Settings → Billing and plans.

| | Local dev container | Codespaces |
|---|---|---|
| Needs Docker | Yes | No |
| Works offline | Yes | No |
| Uses your laptop's RAM/CPU | Yes | No (runs in the cloud) |
| Cost | Free | Free within the student quota |
| Startup | Fast after first build | Fast after first start |

## First-time setup

See the step-by-step **[Setup Guide](SETUP.md)**. It covers installing Docker on each OS, opening the container, and verifying it works.

## Daily use

- **Open the repo:** VS Code usually reopens in the container automatically. The bottom-left corner shows **Dev Container: CSE 250 Project** when you're connected.
- **Run things:** `uv run python ...`, `uv run cse250a ...`, or notebooks with the `.venv` kernel.
- **Pulled changes that touch `.devcontainer/`?** Run **Dev Containers: Rebuild Container**.
- **Pulled changes that touch `pyproject.toml` or `uv.lock`?** Run `uv sync`. No rebuild is needed.
- **Done for the day:** close VS Code. The container stops on its own.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Cannot connect to the Docker daemon" or the build fails immediately | Docker isn't running. Start Docker Desktop and try again. |
| `Author identity unknown` on commit | Set your git name and email **on your machine** (in WSL on Windows), then rebuild: `git config --global user.name "Your Name"` and `git config --global user.email "you@users.noreply.github.com"` |
| Something is broken or weird | **Dev Containers: Rebuild Container**. Your files are safe, since they live in the repo, not the container. |
| Very slow file access (Windows) | The repo is probably on `C:\`. Clone it inside WSL instead. |
| Notebook can't find packages | Select the `.venv` kernel (**Select Kernel → Python Environments → .venv**). |
| Running out of disk | `docker system prune` removes unused images and containers. |

Still stuck? Post the error, plus the output of **Dev Containers: Show Container Log**, in the team chat.

## Further reading

- [VS Code: Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers)
- [VS Code: Dev Containers tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial)
- [containers.dev: the Dev Container spec](https://containers.dev/)
- [GitHub Codespaces quickstart](https://docs.github.com/en/codespaces/getting-started/quickstart)
- [GitHub Education for students](https://docs.github.com/en/education/about-github-education/github-education-for-students/about-github-education-for-students)