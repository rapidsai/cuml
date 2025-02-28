#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

usage() {
    echo "Usage: $0 [options] [-- pytest-options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -p, --path                 Path to scikit-learn source (default: ./scikit-learn)"
    echo "  --select [minimal|relevant|all]   Select test group (default: all)"
    exit 1
}

# Parse command line arguments
PYTEST_ARGS=""
SELECT="all"  # default selection

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
        --select)
            SELECT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

set -eu

SCIKIT_LEARN_SRC_PATH="${SCIKIT_LEARN_SRC_PATH:-./scikit-learn}"
PYTEST_ARGS="${PYTEST_ARGS:-}"

# Define the path to the test selection YAML config
TEST_SELECTION_CONFIG="$(dirname "$0")/test-selection.yaml"

# Validate that the YAML config file exists
if [ ! -f "${TEST_SELECTION_CONFIG}" ]; then
    echo "Error: Test selection config file not found at ${TEST_SELECTION_CONFIG}"
    exit 1
fi

# Extract test patterns from the YAML config based on the SELECT value
TEST_PATTERNS=$(yq -e ".$SELECT[]" "${TEST_SELECTION_CONFIG}")

if [ -z "${TEST_PATTERNS}" ]; then
    echo "Error: No test patterns found for selection '${SELECT}' in ${TEST_SELECTION_CONFIG}"
    exit 1
fi

# Build an array of test file patterns
readarray -t patterns <<< "${TEST_PATTERNS}"

# Check if scikit-learn is built
if [ ! -d "${SCIKIT_LEARN_SRC_PATH}" ]; then
    echo "Error: scikit-learn source not found at ${SCIKIT_LEARN_SRC_PATH}"
    echo "Please run build.sh first"
    exit 1
fi

cd "${SCIKIT_LEARN_SRC_PATH}"

# Run tests with selected patterns
if [ "${patterns[0]}" != "**" ]; then
    pytest -p cuml.accel -v "${patterns[@]}" ${PYTEST_ARGS}
else
    pytest -p cuml.accel -v ${PYTEST_ARGS}
fi
