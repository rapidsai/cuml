# Agent Guide

Onboarding notes for AI coding agents working in the cuML repository.

## Repository overview

This is [cuML](https://github.com/rapidsai/cuml), the RAPIDS GPU-accelerated machine learning library. The repo contains:

- `cpp/` — `libcuml` C++/CUDA library and ML primitives.
- `python/cuml/` — `cuml` Python package (Cython + Python, scikit-learn-compatible API).
- `conda/environments/` — pinned conda env files for development and CI.
- `ci/` — CI scripts.
- `docs/` — Sphinx documentation.

## Building cuML

Before testing any local code change that touches C++/CUDA/Cython, the local install must be rebuilt in an active cuML development environment. Follow the dedicated build skill:

- [.agents/build-cuml/SKILL.md](.agents/build-cuml/SKILL.md) — `build.sh` usage, ccache, and common gotchas.

The full build reference is [BUILD.md](BUILD.md).

## Running tests

For running the C++ gtests, the standard pytest suite, dask tests, the cuml.accel suite, upstream-library tests under cuml.accel, and the integration tests, follow the dedicated test skill:

- [.agents/test-cuml/SKILL.md](.agents/test-cuml/SKILL.md) — active environment checks, `ci/run_*` entry points, common pytest args, and per-suite gotchas.

The CI scripts under [ci/](ci/) (`test_cpp.sh`, `test_python_singlegpu.sh`, `test_python_dask.sh`, `test_python_integration.sh`, `test_python_scikit_learn_tests.sh`, `test_python_cuml_accel_upstream.sh`) are the source of truth for how each suite is invoked.

## Contributing workflow

For pre-commit hooks, clang-tidy, branch naming, and the PR process, see [CONTRIBUTING.md](CONTRIBUTING.md).

Key conventions:

- Linter errors are auto-fixed by pre-commit hooks — don't fix them manually unless asked.

## Code review guidelines

When reviewing changes, agents should follow the layer-specific review rubrics:

- C++/CUDA changes: [cpp/agents.md](cpp/agents.md)
- Python changes: [python/agents.md](python/agents.md)

Both files focus on CRITICAL and HIGH issues only and target a sub-3% false-positive rate.
