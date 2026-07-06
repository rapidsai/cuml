# Benchmark Runner

This directory contains the cuML benchmark runner and supporting utilities for
measuring machine learning algorithm performance across GPU (cuML) and CPU
(scikit-learn) implementations.

The benchmark runner can execute individual algorithms, compare CPU and GPU
performance, sweep parameters, save results to CSV, and run benchmark suites
defined in YAML configuration files.

## Prerequisites

Before running benchmarks, ensure that:

- cuML is installed for GPU benchmarking.
- scikit-learn is available for CPU benchmarking.
- You are running the benchmark from this directory or through the package entry point.

## Features

- Benchmark cuML (GPU) and scikit-learn (CPU) implementations
- Run GPU-only or CPU-only benchmarks
- Execute individual algorithms or benchmark suites
- Save benchmark results to CSV
- Use YAML configuration files
- Perform parameter sweeps
- Benchmark multiple dataset sizes and feature dimensions

## Running the Benchmark

The benchmark runner can be executed either as a Python module:

```bash
python -m cuml.benchmark
```

or directly from this directory:

```bash
python run_benchmarks.py
```

## Running Benchmarks

Run a benchmark using both GPU and CPU (when cuML is available):

```bash
python run_benchmarks.py --dataset classification LogisticRegression
```

Run only CPU benchmarks:

```bash
python run_benchmarks.py --skip-gpu --dataset classification LogisticRegression
```

Run only GPU benchmarks:

```bash
python run_benchmarks.py --skip-cpu --dataset classification LogisticRegression
```

Use a real dataset:

```bash
python run_benchmarks.py --dataset higgs --default-size RandomForestClassifier
```

## Configuration Files

Benchmark suites can also be executed from YAML configuration files.

Available configurations include:

- `configs/single_gpu.yaml`
- `configs/test.yaml`

Example:

```bash
python run_benchmarks.py --config configs/single_gpu.yaml
```

## Useful Commands

List available algorithms:

```bash
python run_benchmarks.py --print-algorithms
```

List available datasets:

```bash
python run_benchmarks.py --print-datasets
```

Display GPU/CPU status:

```bash
python run_benchmarks.py --print-status
```

## Saving Results

Save benchmark results to CSV:

```bash
python run_benchmarks.py --csv results.csv LogisticRegression
```

## Common Command-Line Options

| Option            | Description                          |
| ----------------- | ------------------------------------ |
| `--dataset`       | Select dataset                       |
| `--config`        | Load YAML benchmark configuration    |
| `--profile`       | Select profile from YAML config      |
| `--csv`           | Save benchmark results               |
| `--skip-gpu`      | Run CPU benchmarks only              |
| `--skip-cpu`      | Run GPU benchmarks only              |
| `--backends`      | Specify CPU/GPU backends             |
| `--dtype`         | Dataset precision (`fp32` or `fp64`) |
| `--rmm-allocator` | Choose GPU memory allocator          |

## Getting Help

To view the complete list of supported command-line options, run:

```bash
python run_benchmarks.py --help
```