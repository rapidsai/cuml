#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                     Show this help message"
    echo "      --scikit-learn-version    Scikit-learn version to test (default: 1.5.2)"
    echo "  -p, --path                    Path to scikit-learn source (default: ./scikit-learn)"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

set -eu

SCIKIT_LEARN_VERSION="${SCIKIT_LEARN_VERSION:-1.5.2}"
SCIKIT_LEARN_SRC_PATH="${SCIKIT_LEARN_SRC_PATH:-./scikit-learn}"

# Clone scikit-learn if not already present and checkout version
if [ ! -d "${SCIKIT_LEARN_SRC_PATH}" ]; then
    git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git ${SCIKIT_LEARN_SRC_PATH}
fi

cd ${SCIKIT_LEARN_SRC_PATH}

if [ -n "${SCIKIT_LEARN_VERSION}" ]; then
    git fetch --depth 1 origin tag ${SCIKIT_LEARN_VERSION}
    git checkout ${SCIKIT_LEARN_VERSION}
fi

# Build scikit-learn
python -m pip install \
    --editable . \
    --verbose \
    --no-build-isolation \
    --config-settings editable-verbose=true
