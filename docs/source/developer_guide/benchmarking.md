# Benchmarking with the built-in harness

Use `python -m cuml.benchmark` to run estimator workloads and save warmup and
measurement observations in JSON. A YAML suite defines shared workloads, their
implementation backends, and execution profiles.

## Run a suite

Run in an environment with cuML installed, the selected backends' dependencies,
and enough host/device memory. Suite loading requires PyYAML and msgspec
(`python -m pip install pyyaml msgspec` if they are missing).

```bash
# Run native single-GPU cuML at reduced size.
python -m cuml.benchmark --suite estimators --profile smoke --output results

# Run only native single-GPU cuML and the CPU baseline.
python -m cuml.benchmark --suite estimators --profile smoke \
  --implementation cuml --implementation scikit-learn --output comparison

# Inspect profiles; run accelerated cases with per-repetition logging.
python -m cuml.benchmark --suite estimators --help
python -m cuml.benchmark --implementation cuml.accel --profile smoke -v

# Run a custom manifest.
python -m cuml.benchmark --suite ./my-suite.yaml --output custom-results
```

The `estimators` built-in (the default suite) is packaged in
`python/cuml/cuml/benchmark/suites/estimators.yaml`. Refer to the manifest for
its workloads, implementation backends, and profiles.

By default, the CLI runs benchmarks using only `cuml`. Use `--implementation`
to select another backend and compare implementations of the same algorithms.
Repeat the option to select multiple backends; explicit selections replace the
default. Selected backends run sequentially in separate processes, in manifest
order. Execution continues after a backend failure; the command exits nonzero
if any selected backend fails.

The default profile is `standard`. Select a profile with `--profile`.
Copy the manifest and edit its cases or profiles to customize workloads,
parameters, and execution counts.

Backend requirements:
- `cuml.accel` requires at least one warmup. Warmups collect dispatch evidence;
  any recorded CPU fallback fails the case and stops its repetitions.
  Profiling is limited to warmups.
- `cuml.dask` requires at least two visible GPUs and Dask/Dask-CUDA dependencies.
  The harness creates one local cluster for the backend run.
- Estimator availability and accepted constructor parameters depend on installed
  packages.

## Define a suite

Save this manifest as `my-suite.yaml`:

```yaml
version: 2
name: small-kmeans
implementations: [cuml, scikit-learn, cuml.accel]
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

A suite brings together:

- **Implementations:** a nonempty, unique list of execution backends.
- **Profiles:** named warmup/measurement counts, dataset size scaling, and optional
  timeouts.
- **Cases:** estimator workloads with constructor parameters and measured
  operations, such as `fit_predict`.
- **Datasets:** synthetic inputs defined by generator, shape, data type, and
  representation. Generator parameters are separate from estimator parameters.
- **Input selection:** arguments passed to the operation (`X`, `y`, or both).
  Inference cases also specify training rows and inputs for the initial fit.

Cases inherit the suite's implementations. To restrict a case, give it a
nonempty subset, for example:

```yaml
  - estimator: AgglomerativeClustering
    implementations: [cuml, scikit-learn]
    # ... dataset, operation, input_selection, and parameters ...
```

See the [API and schema reference](../api/cuml.benchmark) for suite validation
and the generated schema.

## Timing and results

The output directory contains one JSON results file per selected backend:

```text
results/
  cuml.json
  scikit-learn.json
  cuml.accel.json
  cuml.dask.json
```

Without `--output`, the CLI chooses an unused timestamped directory in the current
working directory. An explicit output directory may already exist. **Without
`--resume`, results files for selected backends are replaced**; unrelated files and
unselected backends' results files are left untouched. If startup fails before a
results file is created, the failure is reported in the console and exit status.
Check the command's exit status to confirm the invocation succeeded.

Each results file contains:

- **Run metadata:** command, suite/profile, backend execution plan, system, and
  software used for that backend run.
- **Case results:** workload descriptors, implementation, and success/failure.
  Workload IDs are backend-independent, so matching cases can be joined across
  results files. Each backend run has its own run ID.
- **Observations:** individual warmup and measurement repetitions. Use successful
  measurement observations to calculate performance summaries from raw timings.
- **Timing:** synchronized wall time in seconds for the selected estimator
  operation. Preparation and setup are excluded, so these timings differ from
  total elapsed console time.

Each backend saves its results file after every case, including failures.
See the [API and schema reference](../api/cuml.benchmark) for the results schema.

## Resume a run

```bash
python -m cuml.benchmark --suite estimators --profile smoke \
  --output results --resume
```

`--resume` requires an explicit, existing output directory. For each selected
backend, it retains successful cases, retries failed cases from scratch, and
runs missing cases. A missing backend results file starts a new backend run. Other
backends' results files are preserved when selecting a subset or adding another
backend.

Existing results files must be compatible: schema version, methodology, suite
metadata (including path, backend, profile, and execution plan), software, and
system metadata must match. Original backend run IDs are preserved. Changing
counts, timeouts, suite location, packages, or hardware can prevent resume.
Use a new output directory for a changed experiment.

## Add or modify cases

Start from `suites/estimators.yaml` or the smaller `suites/example.yaml`.
The example exercises matching workloads across native cuML, CPU, and accel,
including supervised fitting, data types, sparse inputs, and inference. Use it
as a template and execution check.

Check estimator catalogs in `registry.py` and generator constraints in
`datasets.py`. Extend packaged manifests and catalogs together; the tests in
`python/cuml/tests/test_benchmark.py` check coverage and workload identity.

## NVTX profiling

`cuml.benchmark.nvtx_benchmark` is a standalone Nsight Systems utility. It
requires `nsys` version 2021.4 or later and profiles NVTX ranges in the command
supplied as its single argument:

```bash
python -m cuml.benchmark.nvtx_benchmark \
  "python -m cuml.benchmark --implementation cuml --profile smoke --output results"
```

It is independent of the harness's observation timing and JSON serialization.
