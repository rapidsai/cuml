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

Before testing any local code change that touches C++/CUDA/Cython, the local install must be rebuilt. Follow the dedicated build skill:

- [.agents/build-cuml/SKILL.md](.agents/build-cuml/SKILL.md) — conda env setup, `build.sh` usage, ccache, and common gotchas.

The full build reference is [BUILD.md](BUILD.md).

## Contributing workflow

For pre-commit hooks, clang-tidy, branch naming, and the PR process, see [CONTRIBUTING.md](CONTRIBUTING.md).

Key conventions:

- Linter errors are auto-fixed by pre-commit hooks — don't fix them manually unless asked.
- Always activate the conda dev environment before running `build.sh`, `pytest`, or `pip`.

## Code review guidelines

When reviewing changes, agents should follow the layer-specific review rubrics:

- C++/CUDA changes: [cpp/agents.md](cpp/agents.md)
- Python changes: [python/agents.md](python/agents.md)

Both files focus on CRITICAL and HIGH issues only and target a sub-3% false-positive rate.
