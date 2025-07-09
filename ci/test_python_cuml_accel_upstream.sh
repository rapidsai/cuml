#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test script outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run UMAP tests
rapids-logger "UMAP test suite"
timeout 15m ./python/cuml/cuml_accel_tests/upstream/umap/run-tests.sh

# The upstream HDBSCAN tests require scikit-learn < 1.6. Skip the HDBSCAN
# tests if an incompatible version is installed.
hdbscan_sklearn_version_check=0
python -c "
import sklearn
import sys
from packaging.version import Version
sys.exit(Version(sklearn.__version__) >= Version('1.6'))
" || hdbscan_sklearn_version_check=$?

if [ $hdbscan_sklearn_version_check -eq 0 ]; then
    rapids-logger "HDBSCAN test suite"
    timeout 15m ./python/cuml/cuml_accel_tests/upstream/hdbscan/run-tests.sh
else
    rapids-logger "Skipping HDBSCAN test suite, scikit-learn version not compatible"
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
