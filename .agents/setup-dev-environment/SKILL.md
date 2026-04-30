---
name: setup-dev-environment
description: Set up, create, update, or recreate a cuML conda dev environment for a worktree. Use whenever the user asks to set up a cuML development environment, create or recreate the conda env, configure editor/git exclusions for the env, update the env after pulling new commits, or remove the env. Also use when an agent needs to build or test cuML and there is no usable env to activate yet.
---

# Setting Up a cuML Dev Environment

This skill covers selecting, activating, creating, updating, and removing the development environment for a cuML worktree. The canonical reference is [BUILD.md](../../BUILD.md); this skill captures the high-frequency workflows.

## 1. When this skill applies

- The agent needs to decide which env to use before building or testing.
- The user asks to set up / create / recreate a cuML dev env.
- The user wants to refresh an env after a pull, or remove an env entirely.

## 2. Environment selection algorithm (canonical)

**Before running the algorithm, read every personal rule, skill, or memory that may govern conda or dev-environment selection.** Sources include — but are not limited to — `~/.cursor/rules/`, `~/.cursor/skills/`, `~/.agents/`, `~/.agents/skills/`, `~/.claude/CLAUDE.md`, `~/AGENTS.md`, and any "available skills" / "rules" listing supplied by your agent runtime (e.g. Cursor's `<agent_skills>` system block, Claude's `<rules>` block). The paths above are illustrative, not exhaustive — also follow whatever the runtime advertises. These rules establish the *selection policy* (e.g. "use the latest `cuml-YYYYMMDD` env") that the algorithm applies; if you skip this prelude you will likely create an unnecessary fresh env when a usable one already exists.

Work through these steps in order. Stop at the first that matches.

**Step 1 — Already-active env.** If an env is already active and `cuml` resolves inside this worktree or the active env's `site-packages`, use it and stop:

```bash
# Quick check — run these three lines
echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}  VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
which python
python -c "import cuml; print(cuml.__file__)" 2>/dev/null \
    || echo "cuml not yet installed (fine before first build)"
```

If `cuml.__file__` lives inside this worktree (`python/cuml/`) or inside the active env's `site-packages`, **you're done** — proceed to build or test.

**Step 2 — Editor-configured interpreter.** If `<worktree>/.vscode/settings.json` sets `python.defaultInterpreterPath` *and the interpreter exists on disk*, activate that env (or use that Python directly) — this is the user's explicit choice for the worktree. If the configured path doesn't exist (e.g. it points at `${workspaceFolder}/.conda-env/bin/python` but `.conda-env/` was never created), treat this step as inapplicable and fall through to Step 3 — do **not** treat it as a directive to create that env.

```bash
python3 -c "
import json, pathlib
s = pathlib.Path('.vscode/settings.json')
if s.exists():
    d = json.loads(s.read_text())
    print(d.get('python.defaultInterpreterPath', '(not set)'))
"
```

**Step 3 — Apply the personal-rule selection policy.** If the rules you loaded in the prelude define an env selection policy (e.g. "use the `cuml-YYYYMMDD` env with the latest date suffix", or "always use env `foo`"), apply it now and stop on first match. This step **overrides Steps 4–5** (the worktree-prefix default and the fresh-env fallback) — do not create a new env when an env mandated by personal rules already exists.

*Worked example.* If a personal rule says "use the `cuml-YYYYMMDD` env with the largest date suffix":

```bash
LATEST=$(conda env list | awk '/cuml-[0-9]{8}( |$)/ {print $1}' | sort | tail -1)
echo "Picking $LATEST"
conda activate "$LATEST"
```

Then verify with §3 below. If the env doesn't yet have `cuml` installed, that's fine — proceed to build.

**Step 4 — Worktree prefix env.** If `<worktree>/.conda-env` exists, activate it:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"   # or the user's conda init path
conda activate "$(git rev-parse --show-toplevel)/.conda-env"
```

**Step 5 — Nothing exists.** No usable env was found. Create one from scratch — see [§4 Create a fresh env](#4-create-a-fresh-env-fallback).

> **Do NOT** pick an arbitrary env from `conda env list` by guessing. Names like `cuml-work0`, `rapids-26.04`, `nvforest-work0` typically belong to other worktrees / cuML versions, and building into one silently installs a `cuml` whose ABI may not match this worktree's source. The exception is Step 3: when personal rules mandate a specific naming convention (e.g. `cuml-YYYYMMDD`), `conda env list` is the right tool to find the matching env — that's not "guessing", it's applying a documented policy.

## 3. Sanity check after activation

Regardless of how the env was selected, verify it's the right one before building or testing:

```bash
echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}  VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
which python                                      # should resolve inside the env
python -c "import cuml; print(cuml.__file__)"     # should be inside this worktree, not another clone
```

The env is good when `cuml.__file__` lives inside this worktree (`python/cuml/`) or the active env's `site-packages`. If it points into a different worktree or env, the wrong env is active — revisit step 2 above.

## 4. Create a fresh env (fallback)

Use this only when steps 1–4 above found nothing to activate. The **recommended default** is an in-worktree prefix env at `<worktree>/.conda-env` — one env per worktree, tied to the worktree lifetime. If the user prefers a different location or naming convention, follow their preference.

Properties of the in-worktree prefix env:

- **Deterministic activation**: `conda activate "$(git rev-parse --show-toplevel)/.conda-env"` always activates the env that belongs to the current worktree. No name-derivation logic, no collision handling.
- **Worktree-bound**: deleting the worktree (`rm -rf` or `git worktree remove`) removes the env. No orphan envs accumulating in `~/miniforge3/envs/`.
- **Parallel-agent safe**: two agents in two worktrees can never accidentally activate each other's env.

The `.conda-env/` directory is large (~5–10 GB on disk; conda hardlinks packages from its global cache, so real disk cost is much smaller).

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

For VS Code / Cursor, also write `.vscode/settings.json` (worktree-local, not committed) to keep the editor responsive **and** point the Python extension at the env so terminals, debug, tests, and IntelliSense all use it automatically. This `python.defaultInterpreterPath` setting is also what later agent runs check (per §2 step 2) to avoid creating a duplicate env. Target settings:

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

## 5. Update an existing dev environment

After pulling new commits that change `dependencies.yaml` or the env files, refresh the env using **the same env file the env was originally created from** (don't switch CUDA versions on an existing env — recreate instead):

```bash
PREFIX_ENV="$(git rev-parse --show-toplevel)/.conda-env"
conda env update --prefix "$PREFIX_ENV" --file="$ENV_FILE"
```

## 6. Remove an env

```bash
conda deactivate 2>/dev/null || true
rm -rf "$(git rev-parse --show-toplevel)/.conda-env"
```

(Or `conda env remove --prefix "$(git rev-parse --show-toplevel)/.conda-env"`. The `rm -rf` is faster and works even if conda isn't initialized.)

## Additional resources

- Full build docs: [BUILD.md](../../BUILD.md)
- Conda environment files: `conda/environments/all_cuda-*_arch-*.yaml`
- Build skill (for building after the env is ready): [.agents/build-cuml/SKILL.md](../build-cuml/SKILL.md)
- Test skill: [.agents/test-cuml/SKILL.md](../test-cuml/SKILL.md)
