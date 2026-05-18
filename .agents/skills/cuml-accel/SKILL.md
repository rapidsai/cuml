---
name: cuml-accel
description: >
  Enable cuml.accel GPU acceleration for scikit-learn pipelines and diagnose
  GPU vs CPU dispatch. Use when the user asks to run sklearn code on the GPU,
  mentions cuml.accel, or wants to check which estimators run on the GPU.
---

# cuml.accel for scikit-learn

## Enabling cuml.accel

There are two ways to enable cuml.accel: in-code or from the command line.

### In-code activation

Add this block **before any sklearn imports** so cuml.accel can patch them:

```python
try:
    import cuml.accel
    cuml.accel.install()
    _GPU = True
except ImportError:
    _GPU = False

print(f"Running on: {'GPU (cuml.accel)' if _GPU else 'CPU (sklearn)'}")

from sklearn.pipeline import Pipeline
# ... rest of sklearn imports ...
```

`cuml.accel.install()` must execute before `from sklearn...` — it monkey-patches
sklearn estimators with GPU-accelerated equivalents. The try/except provides a
graceful CPU fallback when cuml is not installed.

### CLI activation

Run any script under cuml.accel without modifying its source:

```bash
python -m cuml.accel script.py
```

This is useful for quick experiments or when the script shouldn't have a cuml
dependency baked in. The script's own sklearn imports are patched automatically.

## Diagnosing GPU vs CPU dispatch

Run the script with debug logging to see per-estimator dispatch decisions:

```bash
# if activation is in-code
CUML_ACCEL_LOG_LEVEL=DEBUG python script.py

# if using CLI activation
CUML_ACCEL_LOG_LEVEL=DEBUG python -m cuml.accel script.py
```

Each acceleratable estimator produces a log line:

- `[cuml.accel] \`Ridge.fit\` ran on GPU` — estimator ran on GPU.
- `[cuml.accel] \`Ridge.fit\` falling back to CPU: <reason>` — estimator has a
  cuml proxy but fell back. The reason string explains why.

### Resolving fallbacks

Read the reason from the log line to triage:

- **Input format issue** (reason mentions the input, e.g. sparse or dtype):
  transform the input before the estimator to match what cuml expects.
- **Unsupported parameter** (reason names a specific parameter): change the
  parameter, or accept the CPU fallback if that constraint is important to the
  pipeline.

### Estimators with no cuml proxy

Estimators that cuml.accel doesn't cover at all won't appear in the debug logs
— there's no "ran on CPU" line, they're simply absent. To spot them: compare the
estimator names in the pipeline against what appears in the log output. Anything
absent was never a candidate for GPU acceleration.

For the full list of accelerated estimators and their per-parameter/per-input
fallback conditions, see the upstream limitations doc for the installed cuml
version:

    https://raw.githubusercontent.com/rapidsai/cuml/refs/tags/v{CUML_VERSION}/docs/source/cuml-accel/limitations.rst

Replace `{CUML_VERSION}` with the installed version (e.g. `26.04.00`). Find it
with `python -c "import cuml; print(cuml.__version__)"`.

## Evaluating whether GPU acceleration helps

GPU acceleration adds kernel launch overhead. On small datasets or cheap
estimators the overhead can outweigh the speedup. Always measure before
recommending cuml.accel for a pipeline. Run the pipeline **both ways**
and compare wall-clock time.

Report both timings.

## Summarizing acceleration work

When reporting on GPU acceleration, categorize **every** estimator in the
pipeline into three buckets so nothing is silently unaccounted for:

- **On GPU**: appeared in logs as `ran on GPU`.
- **Falling back to CPU**: appeared in logs with `falling back to CPU: <reason>`.
- **Not a candidate**: no cuml proxy, absent from logs entirely.

Based on the timing results make a recommendation on whether to keep
cuml.accel enabled for the workload.
