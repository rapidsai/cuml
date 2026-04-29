---
name: build-cuml
description: Build cuML (libcuml C++ library and cuml Python package) from source in a conda dev environment, using the repo's build.sh script. Use whenever the user asks to build, compile, install, or rebuild cuML from source, set up a cuML development environment, install local cuML changes, or before testing local edits to cuML C++/CUDA/Cython/Python code.
---

# Building cuML

This skill teaches the agent how to build cuML from source in this repository. The canonical reference is [BUILD.md](../../BUILD.md) at the repo root; this skill captures the high-frequency workflows so the agent can act without re-reading the full doc each time.

## Quick start (TL;DR)

For an already-set-up dev env, build and install everything for the local GPU arch:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$(git rev-parse --show-toplevel)/.conda-env"
./build.sh --ccache
```

This builds and installs `libcuml` (C++), `cuml` (Python), and `prims` (tests) into `$CONDA_PREFIX`.

> **Note on env layout:** This skill uses an **in-worktree prefix env** at `<worktree>/.conda-env`. Activation is unambiguous (`conda activate ./.conda-env` always points at the right env), and the env is tied to the worktree — deleting the worktree removes the env. See [In-worktree prefix env](#in-worktree-prefix-env) for details.

## When to apply this skill

- The user asks to build / compile / install / rebuild cuML.
- The user wants to set up a cuML development environment.
- The agent has edited C++/CUDA/Cython/Python code in this repo and needs to rebuild before testing.
- A test run fails with `ImportError`, missing `libcuml.so`, stale `.so`, or "module not found" symptoms after a code change — the local install is likely stale and needs a rebuild.

## 1. Activate the conda dev environment

cuML development happens inside a conda environment that contains all build and runtime dependencies. **Always activate it before invoking `build.sh`, `cmake`, `pip`, or `pytest`.**

### In-worktree prefix env

Use **one conda env per cuML worktree/clone**, stored at `<worktree>/.conda-env`. Properties:

- **Deterministic activation**: `conda activate "$(git rev-parse --show-toplevel)/.conda-env"` always activates the env that belongs to the current worktree. No name-derivation logic, no collision handling.
- **Worktree-bound**: deleting the worktree (`rm -rf` or `git worktree remove`) removes the env. No orphan envs accumulating in `~/miniforge3/envs/`.
- **Parallel-agent safe**: two agents in two worktrees can never accidentally activate each other's env.

The `.conda-env/` directory is large (~5–10 GB; conda hardlinks packages from its global cache, so real disk cost is much smaller). IDEs/editors must be told not to index it — see [Exclude the env from git and the editor](#step-3-exclude-the-env-from-git-and-the-editor). Some tooling (e.g. `conda env list`) won't show prefix envs unless they're active.

### Initialize conda in a fresh shell

Conda activation requires conda to be initialized in the shell first:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

(Use the path to the user's conda install — `miniforge3`, `mambaforge`, or `miniconda3`.)

### Activate the worktree's dev environment

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

### Create a fresh dev environment

#### Step 1: Pick the right env file

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

#### Step 2: Configure exclusions (do this BEFORE creating the env)

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

#### Step 3: Create and populate the env

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

### Update an existing dev environment

After pulling new commits that change `dependencies.yaml` or the env files, refresh the env using **the same env file the env was originally created from** (don't switch CUDA versions on an existing env — recreate instead):

```bash
PREFIX_ENV="$(git rev-parse --show-toplevel)/.conda-env"
conda env update --prefix "$PREFIX_ENV" --file="$ENV_FILE"
```

### Removing an env

```bash
conda deactivate 2>/dev/null || true
rm -rf "$(git rev-parse --show-toplevel)/.conda-env"
```

(Or `conda env remove --prefix "$(git rev-parse --show-toplevel)/.conda-env"`. The `rm -rf` is faster and works even if conda isn't initialized.)

## 2. Build with `build.sh`

`build.sh` lives at the repo root and is the recommended entry point. Targets are space-separated; flags can be mixed in. With no targets it builds and installs `libcuml`, `cuml`, and `prims` for the detected GPU arch.

### Common targets

```bash
./build.sh                  # libcuml + cuml + prims (default)
./build.sh libcuml          # C++ library only
./build.sh cuml             # Python package only (assumes libcuml installed)
./build.sh libcuml cuml     # both, explicit
./build.sh clean            # wipe all build artifacts (run first if state is corrupt)
./build.sh prims bench      # ml-prims tests + C++ benchmark
```

### Most useful flags

| Flag | Effect |
| --- | --- |
| `--ccache` | Cache compilations via ccache/sccache. **Strongly recommended for any iterative work.** |
| `-g` | Build with debug info (`RelWithDebInfo`). |
| `-v` | Verbose build output. |
| `-n` | Build but don't install. |
| `--allgpuarch` | Build for all RAPIDS-supported archs (slow; default is `NATIVE` = local GPU only). |
| `--singlegpu` | Drop multi-GPU/MNMG components from `libcuml` and `cuml`. Faster, smaller. |
| `--nolibcumltest` | Skip C++ test binaries (faster libcuml builds). |
| `--configure-only` | Run cmake configure but don't compile (e.g. for clang-tidy DB generation). |

### Environment variables

- `PARALLEL_LEVEL=N` — limit ninja parallelism. **Use this on shared machines or when ninja OOMs the system.** Default is `nproc`.
- `INSTALL_PREFIX=/path` — install location. Default is `$CONDA_PREFIX` when a conda env is active.
- `CMAKE_GENERATOR='Unix Makefiles'` — switch from Ninja to make.
- `CUML_EXTRA_CMAKE_ARGS="..."` — append extra `-D...` flags to cmake.
- `CUML_EXTRA_PYTHON_ARGS="..."` — append extra args to the `pip install` step.

### Recommended fast-iteration command

For day-to-day editing (any code: C++, CUDA, Cython, Python):

```bash
PARALLEL_LEVEL=$(nproc) ./build.sh --ccache
```

For C++-only edits, skip the Python rebuild:

```bash
./build.sh libcuml --ccache
```

For Python-only edits (no C++ touched), skip the C++ rebuild:

```bash
./build.sh cuml
```

## 3. ccache / sccache for fast rebuilds

Branch switching, debug↔release toggles, and CI-style rebuilds become much cheaper with a compile cache. cuML supports both `ccache` and `sccache` (sccache is preferred in CI).

### Enable

Pass `--ccache` to `build.sh`. This sets `-DUSE_CCACHE=ON` for the cmake configure step.

### Verify a cache is installed and used

```bash
which ccache sccache 2>/dev/null

# After a build, inspect stats:
ccache -s        # if using ccache
sccache --show-stats   # if using sccache
```

If neither is on `PATH`, install one into the active dev env:

```bash
conda install -y -c conda-forge ccache
# or
conda install -y -c conda-forge sccache
```

### Cache hit reporting in builds

Add `--build-metrics --incl-cache-stats` to record cache hit rate and produce an HTML build report at `cpp/build/ninja_log.html`:

```bash
./build.sh libcuml --ccache --build-metrics --incl-cache-stats
```

### When to clear the cache

- Compiler version changed (`gcc`/`nvcc` upgrade) — invalidate to avoid stale objects: `ccache -C` or `sccache --zero-stats` then re-run.
- After CUDA toolkit upgrade.
- Otherwise: leave it alone; clearing the cache defeats the purpose.

## 4. Verify the build

After the install step finishes, confirm libcuml and cuml are importable from the active env:

```bash
test -f "$CONDA_PREFIX/lib/libcuml.so" && echo "libcuml.so installed"
python -c "import cuml; print(cuml.__version__, cuml.__file__)"
```

The `cuml.__file__` path should be inside the active conda env (or the editable source tree).

## 5. Common gotchas

- **`ImportError` after pulling new commits**: rebuild — the C++ ABI or Cython-generated code likely changed.
- **`libcuml.so` not found at runtime**: `INSTALL_PREFIX` mismatched the active env. Re-run `build.sh` with the correct env activated.
- **`cuml.__file__` points to a different worktree than the one you're editing**: another worktree's env is active. Run `conda activate "$(git rev-parse --show-toplevel)/.conda-env"` and rebuild — see [In-worktree prefix env](#in-worktree-prefix-env).
- **Out-of-memory or thermal throttling during build**: lower `PARALLEL_LEVEL` (e.g. `PARALLEL_LEVEL=8`).
- **Stale build state after a failed build**: run `./build.sh clean` then rebuild from scratch.
- **Building without a GPU**: the build itself works on CPU-only hosts; only running cuML at test time requires a GPU.
- **`--singlegpu` + multi-GPU tests**: skip MNMG tests (e.g. `pytest --ignore=cuml/tests/dask --ignore=cuml/tests/test_nccl.py`).

## 6. Manual cmake / pip path

`build.sh` is a thin wrapper around `cmake` + `pip install`. If the user explicitly wants the manual flow, or `build.sh` is unavailable, see the **"Manual Process"** section of [BUILD.md](../../BUILD.md). The two key invocations are:

```bash
# C++ library
cd cpp && mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
      -DCMAKE_CUDA_ARCHITECTURES=NATIVE \
      -DUSE_CCACHE=ON ..
cmake --build . -j"$(nproc)" --target install

# Python package (from repo root)
python -m pip install --no-build-isolation --no-deps \
    --config-settings rapidsai.disable-cuda=true \
    python/cuml
```

## Additional resources

- Full build docs and all cmake flags: [BUILD.md](../../BUILD.md)
- Contributing workflow (pre-commit, clang-tidy, branch naming): [CONTRIBUTING.md](../../CONTRIBUTING.md)
- Conda environment files: `conda/environments/all_cuda-*_arch-*.yaml`
- Build script source (the source of truth for flag behavior): `build.sh`
