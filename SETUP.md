# Setup Guide

Step by step from a fresh machine to a running project container. Allow about 30 minutes, mostly downloads.

New to dev containers? Skim [devcontainers.md](devcontainers.md) first for the why.

**Don't want to install Docker?** Skip to [Option B: Codespaces](#option-b-github-codespaces-no-install). It's free for verified students and takes 5 minutes.

---

## Option A: Local dev container

### Step 1: Install Docker

Pick your OS.

<details>
<summary><b>Windows 10 / 11</b></summary>

Docker Desktop on Windows runs on **WSL 2**, a lightweight Linux layer built into Windows.

1. **Install WSL.** Open **PowerShell as Administrator** and run:
   ```powershell
   wsl --install
   ```
   Restart when prompted. After the restart, an Ubuntu window opens and asks you to create a Linux username and password. Remember them, since `sudo` uses this password.
   Already have WSL? Update it instead: `wsl --update`.
2. **Download Docker Desktop** from https://docs.docker.com/desktop/setup/install/windows-install/ and run the installer.
3. On the configuration page, keep **"Use WSL 2 instead of Hyper-V"** checked.
4. Finish the install, then log out and back in (or restart) if asked.
5. **Open Docker Desktop** and accept the Subscription Service Agreement. It's free for personal and educational use.
6. Go to **Settings → Resources → WSL Integration**, turn on your **Ubuntu** distro, then click **Apply & Restart**.

**Requirements:** 64-bit Windows 10/11, 8 GB RAM, and hardware virtualization enabled in BIOS/UEFI. Most modern laptops have it on. If Docker complains, search "enable virtualization" plus your laptop model. Check Docker's page for supported Windows editions.
</details>

<details>
<summary><b>macOS</b></summary>

1. Find your chip: **Apple menu → About This Mac**. "Apple M1/M2/M3…" means **Apple silicon**; "Intel" means **Intel**.
2. **Download the matching Docker Desktop** from https://docs.docker.com/desktop/setup/install/mac-install/.
3. Open `Docker.dmg` and drag **Docker** into **Applications**.
4. Open **Docker** from Applications and accept the Subscription Service Agreement. It's free for personal and educational use.
5. Optional, Apple silicon only: install Rosetta 2, which a few tools still use:
   ```bash
   softwareupdate --install-rosetta
   ```

**Requirements:** one of the current or two previous macOS versions, and at least 4 GB RAM (8 GB+ recommended).
</details>

<details>
<summary><b>Linux</b></summary>

Use **Docker Engine**. It's lighter than Docker Desktop on Linux.

1. Follow the instructions for your distro: https://docs.docker.com/engine/install/ (Ubuntu, Debian, Fedora, …).
2. Let your user run Docker without `sudo`:
   ```bash
   sudo groupadd docker          # may say it already exists; that's fine
   sudo usermod -aG docker $USER
   newgrp docker                 # or log out and back in
   ```
   Note: membership in the `docker` group is effectively root access on that machine.
</details>

### Step 2: Check that Docker works

Open a terminal (on Windows, the **Ubuntu** app) and run:

```bash
docker run hello-world
```

You should see `Hello from Docker!`. If you get "Cannot connect to the Docker daemon", Docker Desktop isn't running. Start it and wait until it says *Engine running*.

### Step 3: Install VS Code and extensions

1. Install **VS Code**: https://code.visualstudio.com/
2. In VS Code, open **Extensions** (`Ctrl/Cmd+Shift+X`) and install:
   - **Dev Containers** (`ms-vscode-remote.remote-containers`)
   - **Windows only:** **WSL** (`ms-vscode-remote.remote-wsl`)

### Step 4: Set your git identity (once)

In the same terminal as Step 2 (on Windows, **inside Ubuntu**), run:

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@users.noreply.github.com"
```

Find your GitHub no-reply address under GitHub → Settings → Emails. The container copies these settings in automatically.

### Step 5: Clone the repo

```bash
mkdir -p ~/dev && cd ~/dev
git clone https://github.com/kberesfo/CSE-250A-Project.git
cd CSE-250A-Project
code .
```

**Windows:** run these commands **inside Ubuntu**, not PowerShell. Keeping the repo in the Linux filesystem (`~/dev`, not `/mnt/c/...`) makes the container much faster. `code .` opens VS Code connected to WSL.

### Step 6: Open the dev container

1. VS Code shows a pop-up: **"Folder contains a Dev Container configuration file. Reopen folder to develop in a container."** Click **Reopen in Container**.
   - Missed it? Open the Command Palette (`Ctrl/Cmd+Shift+P`) and run **Dev Containers: Reopen in Container**.
2. **Wait for the first build,** which takes 3–10 minutes. Click **show log** in the notification to watch it. It will:
   - download the Python base image,
   - install git, the GitHub CLI, and uv,
   - run `uv sync --locked` to create `.venv` with every project package,
   - install the nbstripout git filter.
3. When it's done, the bottom-left corner shows **Dev Container: CSE 250 Project**.

### Step 7: Verify

Open a terminal in VS Code (`` Ctrl+` ``). The prompt should start with `vscode ➜ /workspaces/...`. Then run:

```bash
uv run cse250a --help                       # project CLI works
uv run python scripts/get_data.py --list    # dataset status
git status                                  # git works
```

For notebooks, open any `.ipynb`, click **Select Kernel → Python Environments → .venv**, and run a cell.

**If you plan to push from the container** and git asks for credentials, run `gh auth login` (choose GitHub.com → HTTPS → browser), then `gh auth setup-git`.

You're set. Next time, just open the folder in VS Code and it reconnects to the container.

---

## Option B: GitHub Codespaces (no install)

1. **Get the student benefit** (once): verify at https://education.github.com/pack with your UCSD email. This gives free Codespaces, up to 180 core-hours/month.
2. Go to the repo on GitHub → **Code → Codespaces → Create codespace on main**.
3. Wait for the first build, a few minutes. VS Code opens in your browser with the same environment.
4. Verify with the same commands as Step 7.
5. **When you're done:** **Code → Codespaces → … → Stop codespace**. Commit and push your work before deleting a codespace.

Prefer desktop VS Code? Install the **GitHub Codespaces** extension, then use **Code → Codespaces → … → Open in VS Code Desktop**.

---

## Common problems

| Symptom | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait for *Engine running*. |
| Windows: "WSL 2 installation is incomplete" | Run `wsl --update` in PowerShell, then restart Docker Desktop. |
| Windows: Docker works in PowerShell but not in Ubuntu | Docker Desktop → Settings → Resources → **WSL Integration** → enable Ubuntu. |
| "Virtualization not enabled" | Enable Intel VT-x / AMD-V (SVM) in your BIOS/UEFI settings. |
| Build fails at `uv sync --locked` | Pull the latest `main`; the lock file may have changed. Then **Rebuild Container**. |
| `Author identity unknown` on commit | Redo Step 4 on your host, then **Dev Containers: Rebuild Container**. |
| Very slow file access on Windows | The repo is under `/mnt/c/`. Re-clone it inside Ubuntu (`~/dev`). |
| Anything else weird | **Dev Containers: Rebuild Container**. Your files are safe. Still stuck? Post the error and **Dev Containers: Show Container Log** output in the team chat. |

## References

- Docker Desktop for [Windows](https://docs.docker.com/desktop/setup/install/windows-install/) · [Mac](https://docs.docker.com/desktop/setup/install/mac-install/) · [Docker Engine for Linux](https://docs.docker.com/engine/install/) · [Linux post-install](https://docs.docker.com/engine/install/linux-postinstall/)
- [Microsoft: Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- [VS Code: Dev Containers tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial)
- [GitHub Codespaces quickstart](https://docs.github.com/en/codespaces/getting-started/quickstart)