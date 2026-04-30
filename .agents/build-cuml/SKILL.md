---
name: build-cuml
description: Build cuML (libcuml C++ library and cuml Python package) from source in a conda dev environment, using the repo's build.sh script. Use whenever the user asks to build, compile, install, or rebuild cuML from source, install local cuML changes, or before testing local edits to cuML C++/CUDA/Cython/Python code.
---

# Building cuML

This skill teaches the agent how to build cuML from source in this repository. The canonical reference is [BUILD.md](../../BUILD.md) at the repo root; this skill captures the high-frequency workflows so the agent can act without re-reading the full doc each time.

## Quick start (TL;DR)

With a cuML dev env already active, build and install everything for the local GPU arch:

```bash
./build.sh --ccache
```

This builds and installs `libcuml` (C++), `cuml` (Python), and `prims` (tests) into `$CONDA_PREFIX`.

> **Important (sm_121 / new arch + conda RAPIDS libs):** there is a known bug in the default `NATIVE` build path (documented in [rapidsai/cuml#8021](https://github.com/rapidsai/cuml/issues/8021)). Until fixed upstream, use one of these workarounds when mixing with conda-installed RAPIDS shared libraries:
>
> ```bash
> ./build.sh --allgpuarch
> # or
> CUML_EXTRA_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120-real" ./build.sh
> ```
>
> Bug details: [rapidsai/cuml#8021](https://github.com/rapidsai/cuml/issues/8021).

## When to apply this skill

- The user asks to build / compile / install / rebuild cuML.
- The agent has edited C++/CUDA/Cython/Python code in this repo and needs to rebuild before testing.
- A test run fails with `ImportError`, missing `libcuml.so`, stale `.so`, or "module not found" symptoms after a code change — the local install is likely stale and needs a rebuild.

## 1. Set up the development environment

Building cuML requires an **active** development environment containing all build and runtime dependencies. Building with the wrong env active (or none) silently installs into — and tests against — the wrong worktree.

### Quick check

```bash
echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}  VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
which python
python -c "import cuml; print(cuml.__file__)" 2>/dev/null \
    || echo "cuml not yet installed (fine before first build)"
```

`cuml.__file__` should resolve inside this worktree (`python/cuml/`) or inside the active env's `site-packages`. If it resolves into a different worktree or env, the wrong env is active — see [setup-dev-environment §2](../setup-dev-environment/SKILL.md#2-environment-selection-algorithm-canonical) for the selection algorithm.

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
- **`cuml.__file__` points to a different worktree than the one you're editing**: the wrong env is active. Activate the env that belongs to the worktree you're editing and rebuild.
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
