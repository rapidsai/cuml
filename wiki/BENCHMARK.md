# cuML Benchmark Suite

This document describes how to run the cuML benchmark suite. The tools support two execution modes:

- **Full mode**: `python -m cuml.benchmark` — requires cuML and GPU dependencies; runs GPU + CPU benchmarks.
- **Standalone mode**: `python run_benchmark.py` — works without cuML installed (e.g. from the repo); runs CPU-only benchmarks when cuML is not available.

## Running the benchmarks

### Full mode (cuML installed)

When cuML is installed, use the module entry point:

```bash
python -m cuml.benchmark --dataset classification LogisticRegression --csv results.csv
```

If a GPU is available, this runs both GPU (cuML) and CPU implementations and reports speedup.

### Standalone mode (from the repository)

From the `python/cuml/cuml/benchmark/` directory, you can run without installing cuML:

```bash
cd python/cuml/cuml/benchmark/
python run_benchmark.py --dataset classification LogisticRegression
```

When cuML is not installed, only CPU (e.g. scikit-learn) benchmarks run. The same script works with cuML installed for full GPU+CPU runs.

### Legacy script

You can still run the original script directly (same options):

```bash
cd python/cuml/cuml/benchmark/
python run_benchmarks.py --dataset classification LogisticRegression
```

### CPU-only mode

To run only CPU benchmarks (e.g. when no GPU or to compare CPU implementations only):

```bash
python -m cuml.benchmark --skip-gpu --dataset classification LogisticRegression
# or: python run_benchmark.py --skip-gpu --dataset classification LogisticRegression
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
| `--skip-gpu` | Skip GPU/cuML benchmarks (CPU only). |
| `--skip-cpu` | Skip CPU benchmarks (GPU/cuML only). |
| `--csv [FILE]` | Append results to a CSV file. |
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

## Input types

With GPU/cuML you can use `--input-type` such as `numpy`, `pandas`, or `cudf`. Without GPU, only `numpy` and `pandas` are valid; the script will warn and switch to `numpy` if needed.
