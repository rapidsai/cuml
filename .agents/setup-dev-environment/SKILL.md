---
name: setup-dev-environment
description: Set up, create, update, or recreate a cuML conda dev environment in a worktree. Use whenever the user asks to set up a cuML development environment, create or recreate the conda env, configure editor/git exclusions for the env, update the env after pulling new commits, or remove the env.
---

# Setting Up a cuML Dev Environment

This skill covers creating, activating, updating, and removing the conda development environment for a cuML worktree. The canonical reference is [BUILD.md](../../BUILD.md); this skill captures the high-frequency workflows.

## Overview: in-worktree prefix env

Use **one conda env per cuML worktree/clone**, stored at `<worktree>/.conda-env`. Properties:

- **Deterministic activation**: `conda activate "$(git rev-parse --show-toplevel)/.conda-env"` always activates the env that belongs to the current worktree. No name-derivation logic, no collision handling.
- **Worktree-bound**: deleting the worktree (`rm -rf` or `git worktree remove`) removes the env. No orphan envs accumulating in `~/miniforge3/envs/`.
- **Parallel-agent safe**: two agents in two worktrees can never accidentally activate each other's env.

The `.conda-env/` directory is large (~5–10 GB; conda hardlinks packages from its global cache, so real disk cost is much smaller).

## Initialize conda in a fresh shell

Conda activation requires conda to be initialized in the shell first:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

(Use the path to the user's conda install — `miniforge3`, `mambaforge`, or `miniconda3`.)

## Activate the worktree's dev environment

```bash
conda activate "$(git rev-parse --show-toplevel)/.conda-env"
```

If activation fails with "EnvironmentLocationNotFound" or similar, the env hasn't been created yet — see [Create a fresh dev environment](#create-a-fresh-dev-environment) below.

Verify the env is the right one:

```bash
echo "$CONDA_PREFIX"                              # should be <worktree>/.conda-env
which python                                      # should resolve to inside $CONDA_PREFIX
python -c "import cuml; print(cuml.__file__)"     # should be inside this worktree, not another clone
```

## Create a fresh dev environment

### Step 1: Pick the right env file

Env files are named `all_cuda-<MMM>_arch-<ARCH>.yaml`. Two axes:

```bash
ls conda/environments/all_*.yaml
# all_cuda-129_arch-aarch64.yaml
# all_cuda-129_arch-x86_64.yaml
# all_cuda-131_arch-aarch64.yaml
# all_cuda-131_arch-x86_64.yaml
```

**Architecture (`arch-<ARCH>`)** — must match the host CPU architecture. Always use:

```bash
uname -m   # → x86_64 or aarch64
```

**CUDA version (`cuda-<MMM>`)** — this is the version of the CUDA toolkit and CUDA runtime that conda will install into the env (cuML does not require a system CUDA install). Choose based on the host's NVIDIA driver and GPU compute capability:

| Env file | Conda CUDA | Min host driver | Min GPU compute capability |
| --- | --- | --- | --- |
| `cuda-131` (recommended default) | 13.1 | R580+ | 7.5 (Turing or newer) |
| `cuda-129` | 12.9 | R525+ | 7.0 (Volta or newer) |

**Decision rule:**

1. **Default to `cuda-131`** unless one of the conditions below applies.
2. Use `cuda-129` if the host has a **Volta (sm_70) GPU** (e.g. V100) — CUDA 13 dropped Volta support.
3. Use `cuda-129` if the host's NVIDIA driver is older than R580 — `nvidia-smi` will show the max CUDA version supported.

**Detect the right file automatically:**

```bash
ARCH=$(uname -m)

# Read GPU compute capability (e.g. "7.0", "7.5", "8.0", "9.0")
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1)

# Read the max CUDA version the installed driver supports (e.g. "13.0", "12.9")
DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
echo "Driver: $DRIVER_CUDA  GPU compute cap: $CC  Arch: $ARCH"

# Use cuda-129 if any Volta GPU is in the system; otherwise prefer cuda-131
if awk "BEGIN {exit !($CC < 7.5)}"; then
    CUDA_TAG=129
else
    CUDA_TAG=131
fi
ENV_FILE="conda/environments/all_cuda-${CUDA_TAG}_arch-${ARCH}.yaml"
echo "Using $ENV_FILE"
```

If unsure, on multi-GPU hosts query each GPU and pick the lowest compute capability. If the host has no GPU (CPU-only build), `cuda-131` is fine.

### Step 2: Configure exclusions (do this BEFORE creating the env)

> **Critical ordering:** add the git and editor exclusions *before* running `conda create`. Otherwise git will momentarily see ~20k+ untracked files inside `.conda-env/`, the VS Code/Cursor Git extension will warn `"too many active changes, only a subset of Git features will be enabled"`, and editor indexing/file-watching will spike CPU before the exclusions kick in.

The `.conda-env/` directory must never be committed and must be excluded from editor indexing/search. Two git-side options (pick one):

**Option A — worktree-local exclude (recommended for agents):** doesn't modify any tracked file.

```bash
grep -qxF '/.conda-env/' .git/info/exclude || echo '/.conda-env/' >> .git/info/exclude
```

**Option B — repo-wide `.gitignore`:** if the convention is repo-wide, add `/.conda-env/` to `.gitignore` and commit it.

For VS Code / Cursor, also write `.vscode/settings.json` (worktree-local, not committed) to keep the editor responsive **and** point the Python extension at the env so terminals, debug, tests, and IntelliSense all use it automatically. Target settings:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.conda-env/bin/python",
  "python.terminal.activateEnvironment": true,
  "files.watcherExclude": { "**/.conda-env/**": true },
  "files.exclude":        { "**/.conda-env":    true },
  "search.exclude":       { "**/.conda-env":    true }
}
```

Run the script below from the worktree root to merge these into any existing `.vscode/settings.json` without clobbering it (uses the system `python3`, no conda env required):

```bash
mkdir -p .vscode
python3 - <<'PY'
import json, pathlib
p = pathlib.Path(".vscode/settings.json")
data = json.loads(p.read_text()) if p.exists() else {}
data["python.defaultInterpreterPath"] = "${workspaceFolder}/.conda-env/bin/python"
data["python.terminal.activateEnvironment"] = True
data.setdefault("files.watcherExclude", {})["**/.conda-env/**"] = True
data.setdefault("files.exclude", {})["**/.conda-env"] = True
data.setdefault("search.exclude", {})["**/.conda-env"] = True
p.write_text(json.dumps(data, indent=2) + "\n")
PY
```

Notes:

- `python.defaultInterpreterPath` only takes effect on first workspace open or after running `Python: Clear Workspace Interpreter Setting`. If the workspace was already open with a different interpreter, run that command (or pick the new interpreter via `Python: Select Interpreter`) once to switch.
- `python.terminal.activateEnvironment` is `true` by default; listing it makes the intent explicit and survives users who toggled it off globally.

### Step 3: Create and populate the env

Match the Python version to what the YAML supports (currently `>=3.11,<=3.14`). Create the env at `<worktree>/.conda-env`:

```bash
PREFIX_ENV="$(git rev-parse --show-toplevel)/.conda-env"
conda create -y --prefix "$PREFIX_ENV" python=3.14
conda env update --prefix "$PREFIX_ENV" --file="$ENV_FILE"
conda activate "$PREFIX_ENV"
```

After populating, sanity-check that git doesn't see env files:

```bash
git status -s | wc -l   # should be 0 (or just your real changes); not thousands
```

## Update an existing dev environment

After pulling new commits that change `dependencies.yaml` or the env files, refresh the env using **the same env file the env was originally created from** (don't switch CUDA versions on an existing env — recreate instead):

```bash
PREFIX_ENV="$(git rev-parse --show-toplevel)/.conda-env"
conda env update --prefix "$PREFIX_ENV" --file="$ENV_FILE"
```

## Remove an env

```bash
conda deactivate 2>/dev/null || true
rm -rf "$(git rev-parse --show-toplevel)/.conda-env"
```

(Or `conda env remove --prefix "$(git rev-parse --show-toplevel)/.conda-env"`. The `rm -rf` is faster and works even if conda isn't initialized.)

## Additional resources

- Full build docs: [BUILD.md](../../BUILD.md)
- Conda environment files: `conda/environments/all_cuda-*_arch-*.yaml`
- Build skill (for building after the env is ready): [.agents/build-cuml/SKILL.md](../build-cuml/SKILL.md)
