# cuML Benchmark Suite

This document describes how to run the cuML benchmark suite. The benchmark tools can be used both when cuML is installed (GPU + CPU comparison) and when cuML is not installed (CPU-only, e.g. scikit-learn).

## Running the benchmarks

### From the repository (standalone)

From the `python/cuml/cuml/benchmark/` directory:

```bash
cd python/cuml/cuml/benchmark/
python run_benchmarks.py --dataset classification LogisticRegression
```

You can run without having cuML installed; in that case only CPU (e.g. scikit-learn) benchmarks run.

### When cuML is installed

If cuML is installed and a GPU is available, the script runs both GPU (cuML) and CPU implementations and reports speedup. Example:

```bash
python run_benchmarks.py --dataset classification LogisticRegression --csv results.csv
```

### CPU-only mode

To run only CPU benchmarks (e.g. when no GPU or to compare CPU implementations only):

```bash
python run_benchmarks.py --skip-gpu --dataset classification LogisticRegression
```

### GPU-only mode

To run only GPU (cuML) benchmarks:

```bash
python run_benchmarks.py --skip-cpu --dataset classification LogisticRegression
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
python run_benchmarks.py --dataset classification LogisticRegression
```

Run with parameter sweeps and save to CSV:

```bash
python run_benchmarks.py --dataset classification \
  --max-rows 100000 --min-rows 10000 \
  --dataset-param-sweep n_classes=[2,8] \
  --cuml-param-sweep n_estimators=[10,100] \
  --csv results.csv \
  RandomForestClassifier
```

Run multiple algorithms (use `--` before algorithm names when passing `--input-dimensions`):

```bash
python run_benchmarks.py --dataset blobs --num-sizes 1 \
  --input-dimensions 16 256 -- \
  DBSCAN KMeans PCA UMAP
```

Run with a real dataset at its default size:

```bash
python run_benchmarks.py --dataset higgs --default-size \
  RandomForestClassifier LogisticRegression
```

## Input types

With GPU/cuML you can use `--input-type` such as `numpy`, `pandas`, or `cudf`. Without GPU, only `numpy` and `pandas` are valid; the script will warn and switch to `numpy` if needed.
