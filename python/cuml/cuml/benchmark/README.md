# Benchmark Runner

This directory contains the cuML benchmark runner and supporting utilities for
benchmarking machine learning algorithms with cuML and scikit-learn.

The benchmark runner can be executed directly from the command line or through
YAML configuration files.

## Prerequisites

Before running benchmarks, ensure that:

- cuML is installed for GPU benchmarking.
- scikit-learn is available for CPU benchmarking.
- You are running the benchmark from this directory or through the package entry point.

## Quick Start

Run a benchmark directly:

```bash
python run_benchmarks.py --dataset classification LogisticRegression
```

Run a benchmark from a configuration file:

```bash
python run_benchmarks.py --config configs/single_gpu.yaml
```

Available configuration files include:

- `configs/single_gpu.yaml`
- `configs/test.yaml`

## Getting Help

To see all available command-line options and usage examples, run:

```bash
python run_benchmarks.py --help
```