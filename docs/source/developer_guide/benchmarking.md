# Benchmarking with the built-in harness

Use `python -m cuml.benchmark` to run estimator workloads and save warmup and
measurement observations in JSON. A YAML suite selects a backend, cases, and profiles.

## Run a suite

Run in an environment with cuML installed, the selected backend's dependencies,
and enough host/device memory. Suite loading requires PyYAML and msgspec
(`python -m pip install pyyaml msgspec` if they are missing).

```bash
# Start with all single-GPU cases at reduced size.
python -m cuml.benchmark --suite cuml_sg --profile smoke --output cuml.json

# Inspect available profiles; run CPU cases with per-repetition logging.
python -m cuml.benchmark --suite cuml_accel --help
python -m cuml.benchmark --suite sklearn_cpu --profile smoke -v --output cpu.json

# Run a custom manifest.
python -m cuml.benchmark --suite ./my-suite.yaml --profile standard --output custom.json
```

Built-in suites are packaged under `python/cuml/cuml/benchmark/suites/`:

| Suite | Manifest `implementation` | Execution |
| --- | --- | --- |
| `cuml_sg` (default) | `cuml` | Native single-GPU cuML estimators |
| `cuml_mg` | `cuml.dask` | Distributed cuML on a local multi-GPU cluster |
| `cuml_accel` | `cuml.accel` | Accelerated scikit-learn-compatible estimators |
| `sklearn_cpu` | `scikit-learn` | CPU estimators, including registered UMAP/HDBSCAN cases |

The default profile is `standard`. All built-ins define `standard` (one warmup,
three measurements, full rows) and `smoke` (one warmup, one measurement, 1% rows).
Smoke retains every case and feature column; it is an execution check, not a
performance baseline, and can still need substantial memory. You can define additional
named profiles in your suite manifest and select them with `--profile`.

There are no CLI filters for estimators, operations, or cases, and no CLI
parameter/count overrides. To run a subset or change a workload, copy a packaged
manifest, edit its `cases` or `profiles`, and pass its path to `--suite`.

Backend caveats:
- `cuml_accel` activates acceleration at process startup automatically, restarting
  the process if needed. It requires at least one warmup. Warmups collect dispatch
  evidence; any recorded CPU fallback fails the case and stops its repetitions.
  Measurement repetitions do not use the accelerator profiler.
- `cuml_mg` requires at least two visible GPUs and the Dask/Dask-CUDA dependencies.
  The harness creates a local cluster using the visible GPUs. CSR inputs and
  case timeouts are not supported for this backend.
- Estimator availability and accepted constructor parameters depend on the
  installed packages; catalog validation does not guarantee every combination runs.

## Define a suite

Save this complete manifest as `my-suite.yaml` for the custom command above:

```yaml
version: 2
name: small-kmeans
implementation: cuml
profiles:
  standard:
    warmups: 1
    repetitions: 3
    size_scale: 1.0
    timeout_sec: 60
cases:
  - estimator: KMeans
    dataset:
      kind: blobs
      shape: {rows: 1024, features: 8}
      dtype: float32
      format: dense
      parameters: {centers: 5, cluster_std: 1.0}
    operation: fit_predict
    input_selection: [X]
    parameters: {n_clusters: 5, random_state: 42}
```

A suite manifest brings together a few core concepts:

- **Implementation:** the backend used to run all cases in the suite.
- **Profiles:** named execution settings controlling warmups, measurements,
  dataset size scaling, and optional timeouts.
- **Cases:** the workloads to benchmark. Each case selects an estimator, its
  constructor parameters, and the operation to measure, such as `fit_predict`.
- **Datasets:** synthetic inputs defined by a generator, shape, data type, and
  representation. Dataset parameters configure generation separately from
  estimator parameters.
- **Input selection:** the inputs passed to the operation, such as features
  (`X`) or features and targets (`X` and `y`). Inference cases also describe
  the training data and inputs used to fit the estimator before measurement.

The manifest uses `version: 2`. For the complete structure, get the suite JSON
Schema from `cuml.benchmark.suite.suite_manifest_json_schema()`; see the
[API and schema reference](../api/cuml.benchmark).

## Timing and artifacts

Save a run's results to a JSON artifact with `--output`:

```bash
python -m cuml.benchmark --suite ./my-suite.yaml --profile standard --output results.json
```

The artifact contains run metadata and a list of case results. This illustrative
excerpt shows selected fields from a successful case; other fields and
observations are omitted:

```json
{
  "algorithm": "KMeans",
  "operation": {"name": "fit_predict", "lifecycle": "fit"},
  "outcome": {"status": "success"},
  "observations": [
    {
      "role": "measurement",
      "outcome": {"status": "success"},
      "timings": [{"name": "wall_time", "value": 0.012, "unit": "s"}]
    }
  ]
}
```

- **Run metadata:** records the command, suite/profile, system, and software
  used for the experiment.
- **Case results:** describe each workload, its implementation, and whether it
  succeeded or failed. Workload IDs help match cases across backend runs.
- **Observations:** record individual warmup and measurement repetitions.
  Use successful measurement observations for performance summaries; the
  harness saves raw timings rather than calculating aggregate statistics.
- **Timing:** synchronized wall time in seconds for the selected estimator
  operation. Data preparation and setup are excluded, so these timings differ
  from the overall elapsed time reported in the console.

Results are saved after each case, including failures. Without `--output`, the
CLI chooses a timestamped filename in the current directory. An explicit
`--output` **replaces an existing file unless `--resume` is specified**.

Artifacts use `schema_version: 2`. For the complete structure, see
`cuml/benchmark/schemas/benchmark-result.schema.json` or retrieve it with
`cuml.benchmark.schemas.benchmark_result_schema()` in the
[API and schema reference](../api/cuml.benchmark).

## Resume a run

```bash
python -m cuml.benchmark --suite cuml_sg --profile smoke --output cuml.json --resume
```

`--resume` requires an explicit `--output` pointing to an existing compatible
artifact. It retains successful case results, retries failed cases from scratch,
and runs missing cases. The original run ID is preserved. Matching workload IDs
alone are insufficient: schema version, methodology, suite metadata (including
path, implementation, profile and execution plan), software, and system metadata
must match. Changing counts, timeouts, suite location, packages, or hardware can
prevent resume. Use a new output for a changed experiment.

## Add or modify cases

Start from a packaged suite or its neighboring `example_*.yaml` manifests.
These small examples exercise matching workloads across backends, including
supervised fitting, data types, sparse inputs, and inference. They are templates
and execution checks, not performance baselines or comprehensive coverage.
Run them by passing their file path to `--suite`.

Check the backend's estimator catalog in `registry.py` and generator constraints
in `datasets.py`. Set estimator random-state parameters when supported; the data
seed does not set estimator randomness. Keep workloads identical across backends,
avoid duplicate resolved cases, and test small workloads first. Extend packaged
manifests and catalogs together; check `python/cuml/tests/test_benchmark.py`.

## NVTX profiling

`cuml.benchmark.nvtx_benchmark` is available as a standalone Nsight Systems
profiling utility. It requires `nsys` version 2021.4 or later and profiles NVTX
ranges in the command supplied as its single argument:

```bash
python -m cuml.benchmark.nvtx_benchmark \
  "python -m cuml.benchmark --suite cuml_sg --profile smoke --output cuml.json"
```

It is independent of the harness's observation timing and JSON serialization.
