#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# This script runs scikit-learn tests with cuML acceleration enabled.

usage() {
    echo "Usage: $0 [options] [-- pytest-options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                     Show this help message"
    echo "  -v, --verbose                 Increase output verbosity"
    echo "      --scikit-learn-version    Scikit-learn version to test (default: 1.5.2)"
    echo "  -p, --path                    Path to scikit-learn source (default: ./scikit-learn)"
    echo "  --relevant-only               Run only tests relevant for GPU acceleration"
    echo "  -s, --skip-build              Skip building scikit-learn"
    echo "  -r, --report FILE             Parse existing report instead of running tests"
    echo "  -t, --threshold VALUE         Minimum pass rate threshold [0-100] (default: 0)"
    exit 2
}

# Parse command line arguments
THRESHOLD=0
VERBOSE=0
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
        --scikit-learn-version)
            SCIKIT_LEARN_VERSION="$2"
            shift 2
            ;;
        -p|--path)
            SCIKIT_LEARN_SRC_PATH="$2"
            shift 2
            ;;
        --relevant-only)
            RELEVANT_TESTS_ONLY=1
            shift
            ;;
        -s|--skip-build)
            SKIP_BUILD=1
            shift
            ;;
        -r|--report)
            REPORT_FILE="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate threshold
if ! [[ "$THRESHOLD" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [ "$(echo "$THRESHOLD > 100" | bc -l)" -eq 1 ]; then
    echo "Error: Threshold must be a number between 0 and 100"
    exit 1
fi

# Environment variables:
#   SCIKIT_LEARN_VERSION   - Version of scikit-learn to test (default: 1.5.2)
#   SCIKIT_LEARN_SRC_PATH  - Path to scikit-learn source (default: ./scikit-learn)
#   SKIP_BUILD            - If set, skip building scikit-learn (assumes it's already built)
#   REPORT_FILE           - If set, only parse this XML report instead of running tests
#
# Example usage:
#   ./run-cuml-accel-tests.sh                    # Run all tests
#   SCIKIT_LEARN_VERSION=1.4.0 ./run-cuml-accel-tests.sh  # Test specific version
#   SKIP_BUILD=1 ./run-cuml-accel-tests.sh      # Skip building scikit-learn
#   REPORT_FILE=results.xml ./run-cuml-accel-tests.sh # Only parse existing report

set -eu

SCIKIT_LEARN_VERSION="${SCIKIT_LEARN_VERSION:-1.5.2}"
SCIKIT_LEARN_SRC_PATH="${SCIKIT_LEARN_SRC_PATH:-./scikit-learn}"
SKIP_BUILD="${SKIP_BUILD:-}"
RELEVANT_TESTS_ONLY="${RELEVANT_TESTS_ONLY:-}"
PYTEST_ARGS="${PYTEST_ARGS:-}"

# Collect some information about the environment.
ORIGINAL_PWD=$(pwd)
HEAD_REF=$(git rev-parse HEAD)
CURRENT_DATETIME=$(date +%Y%m%d_%H%M%S)

# Clone scikit-learn if not already present and checkout to the desired version.
if [ ! -d "${SCIKIT_LEARN_SRC_PATH}" ]; then
    git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git ${SCIKIT_LEARN_SRC_PATH}
fi

cd ${SCIKIT_LEARN_SRC_PATH}

if [ -n "${SCIKIT_LEARN_VERSION}" ]; then
    git fetch --depth 1 origin tag ${SCIKIT_LEARN_VERSION}
    git checkout ${SCIKIT_LEARN_VERSION}
fi

# Build scikit-learn if not skipped
if [ -z "${SKIP_BUILD}" ]; then
    # Install dependencies required for building scikit-learn that are not yet present.
    conda install -c conda-forge meson-python

    # Build scikit-learn
    pip install --editable . --verbose --no-build-isolation --config-settings editable-verbose=true
fi

# Run tests.
RESULTS_FILE="${CURRENT_DATETIME}_${HEAD_REF}_results.xml"
if [ -n "${RELEVANT_TESTS_ONLY}" ]; then
    # Select tests that correspond to intercepted estimators in cuML
    TEST_CMD="pytest -p cuml.accel -v \
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
        --junitxml=${ORIGINAL_PWD}/${RESULTS_FILE}"
else
    TEST_CMD="pytest -p cuml.accel -v --junitxml=${ORIGINAL_PWD}/${RESULTS_FILE}"
fi

if [ -n "${PYTEST_ARGS}" ]; then
    TEST_CMD="${TEST_CMD} ${PYTEST_ARGS}"
fi

eval "${TEST_CMD}"
