# UMAP Testing and Embedding Quality Assessment Tools

This directory provides comprehensive tools for both UMAP implementation validation and embedding quality assessment. It serves data scientists, researchers, and developers who need to evaluate the quality of UMAP embeddings or compare different UMAP implementations.

## Overview

The tools in this directory serve three main purposes:

1. **Implementation Testing** (`test_umap.py`): Rigorous validation of cuML UMAP against reference implementations
2. **Embedding Quality Assessment** (`umap_metrics.py`): Comprehensive evaluation tools for measuring the quality of any UMAP embedding
3. **Comparison Implementation** (`run_umap_debug.py`): Detailed analysis comparing cuML UMAP with the reference implementation and allowing debugging

### For Data Scientists

These tools provide **standardized metrics** to evaluate how well your UMAP embeddings preserve data structure. Use them to **quantify embedding quality**, **optimize parameters**, and **generate publication-ready reports** with comprehensive visualizations.

### For Researchers and Developers

These tools enable **rigorous implementation comparison** and provide detailed algorithmic insights including **accuracy benchmarking**, **pipeline debugging**, and **topological analysis** using persistent homology.

## Necessary Dependencies

The following dependencies are **NOT** present in the conda environment and need to be installed separately:

#### Required for Nearest Neighbors search
```bash
conda install -c rapidsai cuvs
```

#### Required for Geodesic Distance Computation
```bash
conda install -c rapidsai cugraph
```

#### Required for Topology Preservation Metrics
```bash
pip install ripser
```

#### Required for Web Report Generation
```bash
pip install plotly
```

## Data Requirements

### Real Dataset Testing

Most of the tests require datasets to be present on disk.

Please first download them,
```bash
conda install -c rapidsai cuvs-bench
python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize
python -m cuvs_bench.get_dataset --dataset fashion-mnist-784-euclidean --normalize
python -m cuvs_bench.get_dataset --dataset gist-960-euclidean --normalize
python -m cuvs_bench.get_dataset --dataset glove-25-angular --normalize
python -m cuvs_bench.get_dataset --dataset mnist-784-euclidean --normalize
python -m cuvs_bench.get_dataset --dataset sift-128-euclidean --normalize
```

Then, allow the datasets to be found by setting the `DATASET_DIR` environment variable:
```bash
export DATASET_DIR=/path/to/benchmark/datasets
```

Expected dataset format:
- Binary files with `.fbin` extension for base vectors
- Datasets should follow the standard ANN benchmark format

## Files Description

### Core Testing Files

- **`test_umap.py`**: Main test suite for UMAP functionality with real-world datasets
- **`umap_metrics.py`**: Comprehensive metrics computation library for UMAP quality assessment
- **`run_umap_debug.py`**: Interactive debugging tool for comparing reference vs cuML implementations
- **`toy_datasets.py`**: Synthetic and real dataset generators for testing
- **`web_results_generation.py`**: Web-based interactive report generation

### Standard Testing (`test_umap.py`)

This file contains tests for real-world datasets commonly used in nearest neighbor search benchmarks:

- **Deep Image 96 Angular**: High-dimensional image features with cosine similarity
- **Fashion-MNIST 784 Euclidean**: Fashion item image embeddings
- **GIST 960 Euclidean**: Image descriptor vectors
- **MNIST 784 Euclidean**: Handwritten digit embeddings
- **SIFT 128 Euclidean**: Scale-invariant feature transform descriptors

#### Key Test Features

- **KNN Accuracy Validation**: Compares k-nearest neighbor search results between cuML and reference implementations, measuring neighbor recall and distance accuracy across different metrics (euclidean, cosine, etc.)
- **Fuzzy Simplicial Set Verification**: Validates the construction of fuzzy simplicial sets by comparing edge weights, graph topology, and membership probabilities between implementations
- **Spectral Initialization Testing**: Compares spectral embedding initialization methods, ensuring consistent starting points for the optimization process
- **Embedding Quality Assessment**: Measures final embedding quality using trustworthiness, continuity, and other established manifold learning metrics
- **Parameter Robustness Testing**: Validates performance across different UMAP parameters (n_neighbors, min_dist, n_components) and dataset characteristics
- **Implementation Consistency**: Ensures cuML produces statistically equivalent results to the reference implementation within acceptable tolerances
- **Performance Regression Detection**: Catches performance degradations or quality regressions in cuML updates

#### Running Tests

```bash
DATASET_DIR=datasets pytest python/cuml/umap_dev_tools/test_umap.py -v
```

### Embedding Quality Assessment (`run_umap_debug.py`)

Interactive tool for UMAP embedding quality assessment and implementation comparison. Provides **comprehensive quality metrics**, **standardized evaluation benchmarks**, and **publication-ready reports**. Also enables **pipeline debugging** and **detailed implementation analysis** across multiple test datasets.

#### Available Datasets

**Synthetic**: Swiss Roll, S-Curve, Sphere, Torus, Gaussian Blobs
**Real**: Iris, Wine, Breast Cancer, Digits, Diabetes

#### Usage Examples

```bash
# Quality assessment with web report
python python/cuml/umap_dev_tools/run_umap_debug.py --implementation cuml --dataset "Swiss Roll" --web-report

# Compare cuML vs reference implementation
python python/cuml/umap_dev_tools/run_umap_debug.py --implementation both --dataset "Swiss Roll" --web-report

# Quick quality check (no web report)
python python/cuml/umap_dev_tools/run_umap_debug.py --dataset "Swiss Roll" --implementation cuml

# List available datasets
python python/cuml/umap_dev_tools/run_umap_debug.py --list-datasets
```

### Quality Metrics Library (`umap_metrics.py`)

This module provides a comprehensive suite of scientifically-validated metrics for assessing UMAP embedding quality. These metrics are based on established literature in manifold learning and dimensionality reduction.

#### Local Structure Preservation
These metrics evaluate how well your embedding preserves local neighborhoods and nearest-neighbor relationships:

- **Trustworthiness**: Quantifies how many of the k-nearest neighbors in the embedding were also k-nearest neighbors in the original space (higher is better, range: 0-1)
- **Continuity**: Measures how many of the k-nearest neighbors in the original space remain k-nearest neighbors in the embedding (higher is better, range: 0-1)

#### Global Structure Preservation
These metrics assess how well large-scale data relationships are maintained:

- **Geodesic Spearman Correlation**: Rank correlation between geodesic distances in original space and Euclidean distances in embedding space
- **Geodesic Pearson Correlation (DEMaP)**: Linear correlation between geodesic and embedded distances - the Distance-based Embedding quality Metric
- **Global Structure Score**: Combined measure of how well overall data topology is preserved

#### Fuzzy Simplicial Set Analysis
For researchers and developers, these metrics analyze the intermediate graph representations:

- **KL Divergence**: Information-theoretic comparison between high-dimensional and low-dimensional fuzzy graphs
- **Jaccard Index**: Proportion of edges that overlap between fuzzy simplicial sets
- **Row-sum L1 Error**: Per-node membership mass differences between graph representations

#### Topology Preservation
Advanced topological analysis using computational topology:

- **Persistent Homology**: Analysis of topological features (holes, connected components) across scales
- **Betti Numbers**: Count of topological features - H0 (connected components) and H1 (loops/cycles)
- **Topological Similarity**: Comparison of persistent diagrams between original and embedded data

#### Interpreting the Metrics

**For Data Scientists:**
- **Trustworthiness & Continuity > 0.9**: Excellent local structure preservation
- **Trustworthiness & Continuity > 0.8**: Good preservation, suitable for most analyses
- **Trustworthiness & Continuity < 0.7**: Poor preservation, consider parameter tuning
- **DEMaP > 0.7**: Good global structure preservation
- **Similar Betti numbers**: Good topological preservation

### Web Report Generation (`web_results_generation.py`)

Creates interactive HTML reports with:

- **Embedding Visualizations**: 2D scatter plots with original data coloring
- **Spectral Initialization Plots**: Visualization of initial embedding states
- **Quality Metrics Tables**: Comprehensive metric comparisons
- **Implementation Comparisons**: Side-by-side reference vs cuML analysis
