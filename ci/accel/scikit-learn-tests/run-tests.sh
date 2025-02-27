#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

usage() {
    echo "Usage: $0 [options] [-- pytest-options]"
    echo ""
    echo "Options:"
    echo "  -h, --help             Show this help message"
    echo "  -p, --path             Path to scikit-learn source (default: ./scikit-learn)"
    echo "  --relevant-only        Run only tests relevant for GPU acceleration"
    exit 1
}

# Parse command line arguments
PYTEST_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --)
            shift
            PYTEST_ARGS="$@"
            break
            ;;
        -h|--help)
            usage
            ;;
        -p|--path)
            SCIKIT_LEARN_SRC_PATH="$2"
            shift 2
            ;;
        --relevant-only)
            RELEVANT_TESTS_ONLY=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

set -eu

SCIKIT_LEARN_SRC_PATH="${SCIKIT_LEARN_SRC_PATH:-./scikit-learn}"
RELEVANT_TESTS_ONLY="${RELEVANT_TESTS_ONLY:-}"
PYTEST_ARGS="${PYTEST_ARGS:-}"

# Check if scikit-learn is built
if [ ! -d "${SCIKIT_LEARN_SRC_PATH}" ]; then
    echo "Error: scikit-learn source not found at ${SCIKIT_LEARN_SRC_PATH}"
    echo "Please run build.sh first"
    exit 1
fi

cd ${SCIKIT_LEARN_SRC_PATH}

# Run tests
if [ -n "${RELEVANT_TESTS_ONLY}" ]; then
    # Select tests that correspond to intercepted estimators in cuML
    pytest -p cuml.accel -v \
        sklearn/cluster/tests/test_k_means.py \
        sklearn/cluster/tests/test_dbscan.py \
        sklearn/decomposition/tests/test_pca.py \
        sklearn/decomposition/tests/test_truncated_svd.py \
        sklearn/linear_model/tests/test_base.py \
        sklearn/linear_model/tests/test_logistic.py \
        sklearn/linear_model/tests/test_ridge.py \
        sklearn/linear_model/tests/test_coordinate_descent.py \
        sklearn/manifold/tests/test_t_sne.py \
        sklearn/neighbors/tests/test_neighbors.py \
        sklearn/ensemble/tests/test_forest.py \
        ${PYTEST_ARGS}
else
    pytest -p cuml.accel -v \
        ${PYTEST_ARGS}
fi
