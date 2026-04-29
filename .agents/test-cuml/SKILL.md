---
name: test-cuml
description: Run cuML test suites — C++ gtests, standard Python tests, dask tests, cuml.accel tests, upstream library tests (sklearn/umap/hdbscan) under cuml.accel, and integration tests. Use when the user asks to run, invoke, or debug cuML tests, pytest, ctest, run sklearn tests under cuml.accel, run dask tests, run integration tests, or wants to verify a code change works.
---

# Running cuML Tests

This skill covers all cuML test suites. The canonical source of truth for each suite is the corresponding `ci/run_*` and `ci/test_*` script — when in doubt, read those first.

## 0. Prerequisites

**Activate the worktree env before running any tests:**

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$(git rev-parse --show-toplevel)/.conda-env"
```

- After editing C++/CUDA/Cython code, rebuild before testing. See [.agents/build-cuml/SKILL.md](../build-cuml/SKILL.md).
- If you see `No module named pytest` (or `python` resolves to base Miniforge instead of the worktree env), the dev environment is not active — the worktree env includes `pytest` and the rest of the test stack.
- All Python tests require a GPU. Most suites support `pytest-xdist` parallelism (`-n auto` or `--numprocesses=N --dist=worksteal`).
- CI entry scripts such as `ci/run_cuml_singlegpu_pytests.sh` call `python` on your `PATH`; they assume an appropriate environment is already activated

---

## 1. C++ gtests (libcuml)

Test binaries are built by the default `./build.sh` targets and installed to `$CONDA_PREFIX/bin/gtests/libcuml/`.

**Canonical entry point:** [`ci/run_ctests.sh`](../../ci/run_ctests.sh) — changes into the install dir (or falls back to `cpp/build/latest`) and runs `ctest --output-on-failure --no-tests=error`.

```bash
# Run all gtests in parallel (matches CI)
./ci/run_ctests.sh -j9

# Run a single test suite by regex
./ci/run_ctests.sh -R SG_DBSCAN_TEST

# Run only prims tests
./ci/run_ctests.sh -R PRIMS_
```

**Direct binary invocation** (faster iteration on one algorithm):

```bash
cd "$CONDA_PREFIX/bin/gtests/libcuml"
./SG_DBSCAN_TEST --gtest_filter='*'
./SG_RF_TEST --gtest_filter='RandomForestClassifierTest*'
./SG_DBSCAN_TEST --gtest_list_tests   # list available test cases
```

Test binary names come from `cpp/tests/CMakeLists.txt` (`SG_*` for single-GPU, `PRIMS_*` for ml-prims, `MG_*` for multi-GPU). Prims tests install to `$CONDA_PREFIX/bin/gtests/libcuml_prims/`.

If binaries are missing, rebuild: `./build.sh` (default targets include the test build).

---

## 2. Standard Python tests (single-GPU, no dask)

**Path:** `python/cuml/tests/` (ignoring the `dask/` subdir).
**Canonical entry point:** [`ci/run_cuml_singlegpu_pytests.sh`](../../ci/run_cuml_singlegpu_pytests.sh)

```bash
# Run the full suite (CI-style, parallel)
./ci/run_cuml_singlegpu_pytests.sh --numprocesses=8 --dist=worksteal

# Quick targeted run
./ci/run_cuml_singlegpu_pytests.sh -k test_kmeans

# Single file, verbose
./ci/run_cuml_singlegpu_pytests.sh -v python/cuml/tests/test_kmeans.py

# Alternatively, run pytest directly from the tests dir
cd python/cuml/tests
pytest --ignore=dask -k test_linear_regression -x
```

**Test categories** (from [`python/cuml/tests/conftest.py`](../../python/cuml/tests/conftest.py)):

| Flag | Runs | Default |
| --- | --- | --- |
| `--run_unit` | `@pytest.mark.unit` tests | Yes (implied when no `--run_*` flag given) |
| `--run_quality` | `@pytest.mark.quality` tests | No |
| `--run_stress` | `@pytest.mark.stress` tests | No |
| `--run_memleak` | `@pytest.mark.memleak` tests | No |

`HYPOTHESIS_ENABLED=true` enables full Hypothesis search (CI nightly behavior; default in CI runs quality/stress profiles).

---

## 3. Dask tests (multi-GPU)

**Path:** `python/cuml/tests/dask/` — each test spins up a `LocalCUDACluster` (TCP by default, UCXX optional).
**Canonical entry point:** [`ci/run_cuml_dask_pytests.sh`](../../ci/run_cuml_dask_pytests.sh)

```bash
# Standard TCP run (default)
./ci/run_cuml_dask_pytests.sh -k test_dask_kmeans

# Full parallel CI run
./ci/run_cuml_dask_pytests.sh --numprocesses=8 --dist=worksteal

# UCXX-only tests (skips all non-UCX tests)
./ci/run_cuml_dask_pytests.sh --run_ucx
```

CI runs both TCP and UCXX passes — see [`ci/test_python_dask.sh`](../../ci/test_python_dask.sh).

**Note on the env:** The single-GPU CI job intentionally errors if `dask` is importable. Use a dask-enabled env (one that has `cuml[dask,test-dask]`: `dask-cudf`, `raft-dask`, `dask-cuda`, `dask-ml`) when running the dask suite.

---

## 4. cuml.accel tests (proxy estimator suite)

**Path:** `python/cuml/cuml_accel_tests/` — `conftest.py` calls `cuml.accel.install()` so all collected tests run with the proxy enabled. The `upstream/` subdir is in `collect_ignore` and must be run separately (see §5).
**Canonical entry point:** [`ci/run_cuml_singlegpu_accel_pytests.sh`](../../ci/run_cuml_singlegpu_accel_pytests.sh)

```bash
# Full suite (CI-style, parallel)
./ci/run_cuml_singlegpu_accel_pytests.sh --numprocesses=8 --dist=worksteal

# Targeted run
./ci/run_cuml_singlegpu_accel_pytests.sh -k test_estimator_proxy

# Direct pytest
cd python/cuml/cuml_accel_tests
pytest test_estimator_proxy.py -k LogisticRegression -x -v
```

The `integration/` subdir inside `cuml_accel_tests/` belongs to the integration suite (§6), not here.

---

## 5. Upstream tests (scikit-learn / umap / hdbscan under cuml.accel)

Each upstream library has a `run-tests.sh` in `python/cuml/cuml_accel_tests/upstream/<lib>/`. The script runs the library's own test suite with `-p cuml.accel` and an `xfail-list.yaml`.

Full workflow reference: [`python/cuml/cuml_accel_tests/upstream/README.md`](../../python/cuml/cuml_accel_tests/upstream/README.md).

### scikit-learn

[`upstream/scikit-learn/run-tests.sh`](../../python/cuml/cuml_accel_tests/upstream/scikit-learn/run-tests.sh) — uses `--pyargs sklearn`.

```bash
# Run all sklearn tests under cuml.accel
./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-tests.sh

# Targeted
./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-tests.sh -k test_kmeans

# Parallel + XML report
./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-tests.sh \
    -n auto --dist worksteal --junitxml=report.xml

# Run known-xfailed tests to see actual results
./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-tests.sh --runxfail
```

**`run-tests.sh` and the xfail list:** the script always passes `--pyargs sklearn` and [`--xfail-list=.../xfail-list.yaml`](../../python/cuml/cuml_accel_tests/upstream/scikit-learn/xfail-list.yaml). The `cuml.accel` pytest plugin marks xfails and **requires every condition-matched id in the YAML to match a collected test**; if you instead run e.g. `pytest -p cuml.accel --pyargs sklearn.neighbors.tests.test_kde --xfail-list=...`, most xfail entries have no corresponding collected test and you get `UnmatchedXfailTests` (treated as an error by [`python/cuml/cuml_accel_tests/upstream/pytest.ini`](../../python/cuml/cuml_accel_tests/upstream/pytest.ini)). For targeted runs **with** the official xfail list, keep using `run-tests.sh` and filter with `-k` (full sklearn collection, then deselect; still usually seconds of overhead). For a **fast local loop** on one module only, you may run `python -m pytest -p cuml.accel --pyargs sklearn.<submodule> ...` **without** `--xfail-list` — but tests listed as xfail in the YAML may then appear as real failures.

### umap

[`upstream/umap/run-tests.sh`](../../python/cuml/cuml_accel_tests/upstream/umap/run-tests.sh) — clones the umap repo at the tag matching the installed `umap.__version__` into `umap-upstream/`.

```bash
./python/cuml/cuml_accel_tests/upstream/umap/run-tests.sh
./python/cuml/cuml_accel_tests/upstream/umap/run-tests.sh -k test_umap_transform
```

### hdbscan

[`upstream/hdbscan/run-tests.sh`](../../python/cuml/cuml_accel_tests/upstream/hdbscan/run-tests.sh) — uses `--pyargs hdbscan.tests`.

```bash
./python/cuml/cuml_accel_tests/upstream/hdbscan/run-tests.sh
./python/cuml/cuml_accel_tests/upstream/hdbscan/run-tests.sh -k test_hdbscan
```

### Analyzing results

```bash
# Summary (pass/fail counts)
./python/cuml/cuml_accel_tests/upstream/summarize-results.py report.xml

# Verbose — show failure details
./python/cuml/cuml_accel_tests/upstream/summarize-results.py -v report.xml

# Enforce a minimum pass rate (e.g. 80%)
./python/cuml/cuml_accel_tests/upstream/summarize-results.py -f 80 report.xml

# Tracebacks for specific failures
./python/cuml/cuml_accel_tests/upstream/summarize-results.py \
    --format=traceback -k logistic report.xml

# Generate a new xfail list from results
./python/cuml/cuml_accel_tests/upstream/summarize-results.py \
    --format=xfail_list report.xml > new-xfail-list.yaml
```

For editing the xfail list (adding/updating reasons, markers, conditions), use `xfail_manager.py` — see the [README](../../python/cuml/cuml_accel_tests/upstream/README.md).

---

## 6. Integration tests (cudf.pandas + cuml.accel)

Runs the standard `python/cuml/tests/` suite (not dask) with `-p cudf.pandas` so pandas operations route through the cudf-pandas wrapper, plus `--quick_run` to keep runtime bounded.
**Canonical entry point:** [`ci/run_cuml_integration_pytests.sh`](../../ci/run_cuml_integration_pytests.sh)

```bash
# Full CI run
./ci/run_cuml_integration_pytests.sh --numprocesses=8 --dist=worksteal

# Targeted
./ci/run_cuml_integration_pytests.sh -k test_linear_regression

# Direct pytest equivalent
cd python/cuml/tests
pytest -p cudf.pandas --cache-clear --ignore=dask --quick_run .
```

Tests marked `@pytest.mark.cudf_pandas` are skipped unless `-p cudf.pandas` is loaded (enforced in `tests/conftest.py`).

CI script: [`ci/test_python_integration.sh`](../../ci/test_python_integration.sh).

---

## 7. Common gotchas

- **`ImportError` or wrong `cuml.__file__`**: another worktree's env is active. Run `conda activate "$(git rev-parse --show-toplevel)/.conda-env"` and rebuild. See the [build skill](../build-cuml/SKILL.md).
- **C++ test binaries not found**: the install dir `$CONDA_PREFIX/bin/gtests/libcuml/` is absent. Run `./build.sh` (default builds and installs the tests).
- **Dask import error in single-GPU env**: the single-GPU CI script (`test_python_singlegpu.sh`) intentionally fails if `dask` is installed. Use a dask-capable env for §3 tests.
- **`UnmatchedXfailTests` in upstream sklearn tests**: (1) the xfail list references a test id that no longer exists in the installed library version — triage and update the list (see the upstream [README](../../python/cuml/cuml_accel_tests/upstream/README.md)). (2) You passed `--xfail-list` while collecting only a **subset** of the suite (e.g. `--pyargs sklearn.neighbors.tests.test_kde`); use [`run-tests.sh`](../../python/cuml/cuml_accel_tests/upstream/scikit-learn/run-tests.sh) with `-k` instead, or drop `--xfail-list` for a quick narrow run.
- **xdist worker crash after CUDA error**: this is intentional — `tests/conftest.py` calls `os._exit(1)` on sticky CUDA errors so xdist spawns a clean worker. Don't attempt to suppress it.

---

## 8. Additional resources

- Build skill: [.agents/build-cuml/SKILL.md](../build-cuml/SKILL.md)
- Full build doc and manual cmake/test paths: [BUILD.md](../../BUILD.md)
- Upstream test workflow and xfail management: [python/cuml/cuml_accel_tests/upstream/README.md](../../python/cuml/cuml_accel_tests/upstream/README.md)
- Pytest markers and filterwarnings: [`python/cuml/pyproject.toml`](../../python/cuml/pyproject.toml) `[tool.pytest.ini_options]`
- Test suite conftest (markers, xdist hooks, Hypothesis profiles): [`python/cuml/tests/conftest.py`](../../python/cuml/tests/conftest.py)
- Dask conftest (cluster fixtures, `--run_ucx` option): [`python/cuml/tests/dask/conftest.py`](../../python/cuml/tests/dask/conftest.py)
