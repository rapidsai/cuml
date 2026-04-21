# cuML Benchmark Suite

This document describes how to run the cuML benchmark suite. The tools support two execution modes:

- **Full mode**: `python -m cuml.benchmark` — requires cuML and GPU dependencies; runs GPU + CPU benchmarks.
- **Standalone mode**: `python run_benchmarks.py` — works without cuML installed (e.g. from the repo); runs CPU-only benchmarks when cuML is not available.

The benchmark runner also supports YAML manifests. A manifest is the declarative source of truth for a benchmark suite, while CLI flags can be used to filter or override selected fields at runtime.

## Running the benchmarks

### Full mode (cuML installed)

When cuML is installed, use the module entry point:

```bash
python -m cuml.benchmark --dataset classification LogisticRegression --csv results.csv
```

If a GPU is available, this runs both GPU (cuML) and CPU implementations and reports speedup.

To run a YAML-defined suite:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/single_gpu.yaml \
  --profile default \
  --csv results.csv
```

To run the tiny harness-validation manifest:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/test.yaml \
  --profile default \
  --skip-gpu
```

### Standalone mode (from the repository)

From the `python/cuml/cuml/benchmark/` directory, you can run without installing cuML:

```bash
cd python/cuml/cuml/benchmark/
python run_benchmarks.py --dataset classification LogisticRegression
```

When cuML is not installed, only CPU (e.g. scikit-learn) benchmarks run. The same script works with cuML installed for full GPU+CPU runs.

Standalone mode also supports YAML manifests:

```bash
cd python/cuml/cuml/benchmark/
python run_benchmarks.py \
  --config configs/test.yaml \
  --profile default \
  --skip-gpu
```

The following Python packages are required for standalone mode:

```bash
pip install numpy pandas scikit-learn scipy
```

### CPU-only mode

To run only CPU benchmarks (e.g. when no GPU or to compare CPU implementations only):

```bash
python -m cuml.benchmark --skip-gpu --dataset classification LogisticRegression
# or: python run_benchmarks.py --skip-gpu --dataset classification LogisticRegression
```

### GPU-only mode

To run only GPU (cuML) benchmarks:

```bash
python -m cuml.benchmark --skip-cpu --dataset classification LogisticRegression
```

**Note:** Do not use both `--skip-gpu` and `--skip-cpu`; that would run no benchmarks. The script will report an error.

## Common options

| Option | Description |
|--------|-------------|
| `--dataset` | Dataset name: e.g. `blobs`, `classification`, `regression`, `higgs`. Use `--print-datasets` to list all. |
| `--config` | Path to a YAML benchmark manifest. |
| `--profile` | Named profile to select from a YAML benchmark manifest. |
| `--skip-gpu` | Skip GPU/cuML benchmarks (CPU only). |
| `--skip-cpu` | Skip CPU benchmarks (GPU/cuML only). |
| `--csv [FILE]` | Save results to a CSV file. |
| `--min-rows`, `--max-rows`, `--num-sizes` | Control sample sizes for scaling benchmarks. |
| `--input-dimensions` | Feature dimensions to test (e.g. `16 256`). |
| `--print-algorithms` | List available algorithms and exit. |
| `--print-datasets` | List available datasets and exit. |
| `--print-status` | Print GPU/cuML availability and exit. |

## Examples

Run a single algorithm at default sizes:

```bash
python -m cuml.benchmark --dataset classification LogisticRegression
```

Run with parameter sweeps and save to CSV:

```bash
python -m cuml.benchmark --dataset classification \
  --max-rows 100000 --min-rows 10000 \
  --dataset-param-sweep n_classes=[2,8] \
  --cuml-param-sweep n_estimators=[10,100] \
  --csv results.csv \
  RandomForestClassifier
```

Run multiple algorithms (use `--` before algorithm names when passing `--input-dimensions`):

```bash
python -m cuml.benchmark --dataset blobs --num-sizes 1 \
  --input-dimensions 16 256 -- \
  DBSCAN KMeans PCA UMAP
```

Run with a real dataset at its default size:

```bash
python -m cuml.benchmark --dataset higgs --default-size \
  RandomForestClassifier LogisticRegression
```

Run a manifest but restrict execution to one algorithm:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/single_gpu.yaml \
  --profile default \
  LogisticRegression
```

Run a manifest and override only the row count from the CLI:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/test.yaml \
  --profile default \
  --num-rows 500 \
  --skip-gpu
```

In config mode, only explicitly provided CLI flags override the manifest. Parser defaults do not silently replace YAML values.

## YAML manifests

A manifest defines a benchmark suite, default settings for the suite, and the individual benchmark entries to run.

The repository currently includes:

- `python/cuml/cuml/benchmark/configs/test.yaml`: tiny smoke/harness-validation suite
- `python/cuml/cuml/benchmark/configs/single_gpu.yaml`: canonical single-GPU regression suite with multiple profiles

### Top-level schema

Each manifest is a YAML mapping with these top-level fields:

- `version`: integer schema version. Currently `1`.
- `suite`: metadata about the suite.
- `profiles`: optional named selectors that include benchmarks by tag.
- `defaults`: optional fields applied to every benchmark unless overridden.
- `benchmarks`: list of benchmark entries.

Example:

```yaml
version: 1

suite:
  name: single_gpu
  tier: single_gpu
  description: Canonical single-GPU regression manifest

profiles:
  default:
    include_tags: [default-profile]

defaults:
  input_type: cupy
  dtype: fp32
  n_reps: 3
  random_state: 42
  test_split: 0.1
  run_cpu: true
  run_gpu: true
  raise_on_error: true

benchmarks:
  - id: logreg_fit_small
    algorithm: LogisticRegression
    dataset: classification
    operation: fit
    rows: [100000]
    features: [16]
    tags: [default-profile, linear, classification]
```

### `suite`

`suite` must contain:

- `name`: short suite identifier
- `tier`: suite category such as `test` or `single_gpu`
- `description`: human-readable description

### `profiles`

Profiles allow a single manifest to define multiple run surfaces such as PR, extended, or nightly.

In `single_gpu.yaml`, the manifest is organized along two axes:

- width class: `narrow`, `medium`, `wide`
- runtime tier: `default`, `extended`, `nightly`

The width class changes the feature shape of the workload. The runtime tier changes only the row count. Every algorithm should therefore appear as a full `width x tier` grid.

Each profile currently has:

- `include_tags`: list of tags; a benchmark is included when it shares at least one tag with the profile

Example:

```yaml
profiles:
  default:
    include_tags: [default-profile]
  extended:
    include_tags: [default-profile, extended-profile]
  nightly:
    include_tags: [default-profile, extended-profile, nightly-only]
```

This means:

- `default` includes `narrow`, `medium`, and `wide` workloads for every algorithm at the smallest row counts
- `extended` includes the same `narrow`, `medium`, and `wide` workloads with larger row counts
- `nightly` includes the same `narrow`, `medium`, and `wide` workloads with the largest row counts

Example:

```yaml
- id: logreg_fit_narrow_default
  algorithm: LogisticRegression
  dataset: classification
  operation: fit
  rows: [500000]
  features: [16]
  tags: [default-profile, narrow, linear]

- id: logreg_fit_narrow_extended
  algorithm: LogisticRegression
  dataset: classification
  operation: fit
  rows: [1000000]
  features: [16]
  tags: [extended-profile, narrow, linear]

- id: logreg_fit_wide_default
  algorithm: LogisticRegression
  dataset: classification
  operation: fit
  rows: [150000]
  features: [512]
  tags: [default-profile, wide, linear]
```

### YAML anchors and merge keys

To avoid repeating the same benchmark definition three times for `default`, `extended`, and `nightly`, `single_gpu.yaml` uses standard YAML anchors and merge keys.

Example:

```yaml
- &logreg_fit_narrow_default
  id: logreg_fit_narrow_default
  algorithm: LogisticRegression
  dataset: classification
  operation: fit
  rows: [500000]
  features: [16]
  tags: [default-profile, narrow, linear, classification]

- <<: *logreg_fit_narrow_default
  id: logreg_fit_narrow_extended
  rows: [1000000]
  tags: [extended-profile, narrow, linear, classification]
```

How this works:

- `&logreg_fit_narrow_default` defines an anchor name for the first mapping
- `*logreg_fit_narrow_default` references that anchored mapping later
- `<<:` is the YAML merge key, which copies the anchored fields into the new entry
- any fields written after `<<:` override the copied values

So the extended entry above inherits:

- `algorithm: LogisticRegression`
- `dataset: classification`
- `operation: fit`
- `features: [16]`

and then overrides:

- `id`
- `rows`
- `tags`

This keeps the manifest compact while preserving the intended rule that width changes shape and tier changes row count.

### `defaults`

`defaults` can provide common values for benchmark entries, including:

- `dataset`
- `input_type`
- `dtype`
- `n_reps`
- `random_state`
- `test_split`
- `run_cpu`, `run_gpu`
- `raise_on_error`
- `enabled`
- `tags`
- `params`, `cuml_params`, `cpu_params`, `dataset_params`
- `param_grid`, `cuml_param_grid`, `cpu_param_grid`, `dataset_param_grid`
- `comparison`
- `metadata`

Benchmark entries override scalar defaults and merge dictionary-style fields.

For GPU-oriented suites such as `single_gpu.yaml`, prefer a GPU-native `input_type` such as `cupy` or `cudf`. This avoids benchmarking large host/device or cuDF-to-NumPy conversion costs when the goal is to measure the estimator itself on GPU inputs.

### Benchmark entry schema

Each entry in `benchmarks` must define:

- `algorithm`: algorithm name from the benchmark registry

It can also define:

- `id`: stable benchmark identifier; defaults to `algorithm` if omitted
- `dataset`: dataset generator name such as `classification`, `regression`, or `blobs`
- `operation`: benchmarked operation such as `fit`, `predict`, `transform`, `fit_transform`, `fit_predict`, `fit_kneighbors`, or `kneighbors`
- `rows` and `features`: lists used as a Cartesian product
- `shapes`: explicit paired `(rows, features)` combinations when you do not want a Cartesian product
- `default_size`: use the dataset's natural default size instead of explicit dimensions
- `params`: shared estimator parameters
- `cuml_params`: GPU-only estimator parameters
- `cpu_params`: CPU-only estimator parameters
- `dataset_params`: dataset generator parameters
- `param_grid`, `cuml_param_grid`, `cpu_param_grid`, `dataset_param_grid`: parameter sweeps expanded as Cartesian products
- `n_reps`, `input_type`, `dtype`, `random_state`, `test_split`, `run_cpu`, `run_gpu`, `raise_on_error`
- `tags`: labels used by profiles
- `enabled`, `skip_reason`
- `comparison`, `metadata`

### Choosing dimensions

Use `rows` plus `features` when you want the Cartesian product:

```yaml
- id: pca_fit_small
  algorithm: PCA
  dataset: blobs
  operation: fit
  rows: [100000]
  features: [32, 256]
```

This runs `(100000, 32)` and `(100000, 256)`.

Use `shapes` when you want explicit pairs instead:

```yaml
- id: paired_shapes_example
  algorithm: LogisticRegression
  dataset: classification
  operation: fit
  shapes:
    - rows: 10000
      features: 32
    - rows: 50000
      features: 256
```

This runs only `(10000, 32)` and `(50000, 256)`.

### Creating your own manifest

Start from `test.yaml` if you want a tiny suite, or from `single_gpu.yaml` if you want a richer example with profiles.

This minimal custom manifest is a good starting point:

```yaml
version: 1

suite:
  name: my_benchmarks
  tier: custom
  description: Example custom benchmark suite

profiles:
  default:
    include_tags: [default-profile]
  extended:
    include_tags: [default-profile, extended-profile]

defaults:
  input_type: cupy
  dtype: fp32
  n_reps: 2
  random_state: 42
  test_split: 0.1
  run_cpu: true
  run_gpu: true
  raise_on_error: true

benchmarks:
  - id: my_logreg_narrow_default
    algorithm: LogisticRegression
    dataset: classification
    operation: fit
    rows: [500000]
    features: [16]
    tags: [default-profile, narrow, linear]

  - id: my_logreg_narrow_extended
    algorithm: LogisticRegression
    dataset: classification
    operation: fit
    rows: [1000000]
    features: [16]
    tags: [extended-profile, narrow, linear]

  - id: my_logreg_wide_default
    algorithm: LogisticRegression
    dataset: classification
    operation: fit
    rows: [150000]
    features: [512]
    tags: [default-profile, wide, linear]

  - id: my_kmeans_medium_default
    algorithm: KMeans
    dataset: blobs
    operation: fit_predict
    rows: [50000]
    features: [128]
    dataset_params:
      centers: 8
    tags: [default-profile, medium, clustering]
```

Save it anywhere and run it with:

```bash
python -m cuml.benchmark --config /path/to/my_benchmarks.yaml --profile default
```

If you also define `extended` or `nightly` profiles, keep the same algorithm and width-class entries, and increase only `rows` for those companion entries.

### CLI overrides in config mode

When `--config` is provided, the manifest remains the primary definition of the benchmark suite. CLI flags can still override selected fields for ad hoc runs.

Common useful overrides are:

- `--profile`
- positional algorithm names to restrict which manifest entries run
- `--skip-gpu` or `--skip-cpu`
- `--num-rows`, `--num-features`, `--input-dimensions`, `--default-size`
- `--n-reps`, `--dtype`, `--input-type`, `--test-split`
- `--param-sweep`, `--cuml-param-sweep`, `--cpu-param-sweep`, `--dataset-param-sweep`

Example:

```bash
python -m cuml.benchmark \
  --config python/cuml/cuml/benchmark/configs/single_gpu.yaml \
  --profile default \
  --skip-gpu \
  --num-rows 50000 \
  LogisticRegression
```

This keeps the manifest-selected benchmark entry but overrides the row count and execution mode for that run.

## Input types

With GPU/cuML you can use `--input-type` such as `numpy`, `pandas`, or `cudf`. Without GPU, only `numpy` and `pandas` are valid; the script will warn and switch to `numpy` if needed.
