#!/bin/bash
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
##############################################
# cuML GPU build and test script for CI      #
##############################################

set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CONDA_ARTIFACT_PATH=${WORKSPACE}/ci/artifacts/cuml/cpu/.conda-bld/

# Set home to the job's workspace
export HOME=$WORKSPACE

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Parse git describe
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Install dependencies"
gpuci_conda_retry install -c conda-forge -c rapidsai -c rapidsai-nightly -c nvidia \
      "cudatoolkit=${CUDA_REL}" \
      "cudf=${MINOR_VERSION}" \
      "rmm=${MINOR_VERSION}" \
      "libcumlprims=${MINOR_VERSION}" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "ucx-py=${MINOR_VERSION}" \
      "ucx-proc=*=gpu" \
      "xgboost=1.3.3dev.rapidsai${MINOR_VERSION}" \
      "rapids-build-env=${MINOR_VERSION}.*" \
      "rapids-notebook-env=${MINOR_VERSION}.*" \
      "rapids-doc-env=${MINOR_VERSION}.*"

# https://docs.rapids.ai/maintainers/depmgmt/
# gpuci_conda_retry remove --force rapids-build-env rapids-notebook-env
# gpuci_conda_retry install -y "your-pkg=1.0.0"

gpuci_logger "Install contextvars if needed"
py_ver=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
if [ "$py_ver" == "3.6" ];then
    conda install contextvars
fi

gpuci_logger "Install the main version of dask and distributed"
set -x
pip install "git+https://github.com/dask/distributed.git@main" --upgrade --no-deps
pip install "git+https://github.com/dask/dask.git@main" --upgrade --no-deps
set +x

gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

gpuci_logger "Adding ${CONDA_PREFIX}/lib to LD_LIBRARY_PATH"

export LD_LIBRARY_PATH_CACHED=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_logger "Building doxygen C++ docs"
    $WORKSPACE/build.sh cppdocs -v

    ################################################################################
    # BUILD - Build libcuml, cuML, and prims from source
    ################################################################################

    gpuci_logger "Build from source"
    $WORKSPACE/build.sh clean libcuml cuml prims bench -v --codecov

    gpuci_logger "Resetting LD_LIBRARY_PATH"

    cd $WORKSPACE

    ################################################################################
    # TEST - Run GoogleTest and py.tests for libcuml and cuML
    ################################################################################
    if hasArg --skip-tests; then
        gpuci_logger "Skipping Tests"
        exit 0
    fi

    gpuci_logger "Check GPU usage"
    nvidia-smi

    gpuci_logger "GoogleTest for libcuml"
    set -x
    cd $WORKSPACE/cpp/build
    GTEST_OUTPUT="xml:${WORKSPACE}/test-results/libcuml_cpp/" ./test/ml

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_CACHED
    export LD_LIBRARY_PATH_CACHED=""

    gpuci_logger "Python pytest for cuml"
    cd $WORKSPACE/python

    pytest --cache-clear --basetemp=${WORKSPACE}/cuml-cuda-tmp --junitxml=${WORKSPACE}/junit-cuml.xml -v -s -m "not memleak" --durations=50 --timeout=300 --ignore=cuml/test/dask --ignore=cuml/raft --cov-config=.coveragerc --cov=cuml --cov-report=xml:${WORKSPACE}/python/cuml/cuml-coverage.xml --cov-report term

    timeout 7200 sh -c "pytest cuml/test/dask --cache-clear --basetemp=${WORKSPACE}/cuml-mg-cuda-tmp --junitxml=${WORKSPACE}/junit-cuml-mg.xml -v -s -m 'not memleak' --durations=50 --timeout=300 --cov-config=.coveragerc --cov=cuml --cov-report=xml:${WORKSPACE}/python/cuml/cuml-dask-coverage.xml --cov-report term"


    ################################################################################
    # TEST - Run notebook tests
    ################################################################################
    set +e -Eo pipefail
    EXITCODE=0
    trap "EXITCODE=1" ERR

    gpuci_logger "Notebook tests"
    ${WORKSPACE}/ci/gpu/test-notebooks.sh 2>&1 | tee nbtest.log
    python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log

    ################################################################################
    # TEST - Run GoogleTest for ml-prims
    ################################################################################

    gpuci_logger "Run ml-prims test"
    cd $WORKSPACE/cpp/build
    GTEST_OUTPUT="xml:${WORKSPACE}/test-results/prims/" ./test/prims

    ################################################################################
    # TEST - Run GoogleTest for ml-prims, but with cuda-memcheck enabled
    ################################################################################

    if [ "$BUILD_MODE" = "branch" ] && [ "$BUILD_TYPE" = "gpu" ]; then
        gpuci_logger "GoogleTest for ml-prims with cuda-memcheck enabled..."
        cd $WORKSPACE/cpp/build
        python ../scripts/cuda-memcheck.py -tool memcheck -exe ./test/prims
    fi
else
    #Project Flash
    export LIBCUML_BUILD_DIR="$WORKSPACE/ci/artifacts/cuml/cpu/conda_work/cpp/build"
    export LD_LIBRARY_PATH="$LIBCUML_BUILD_DIR:$LD_LIBRARY_PATH"

    if hasArg --skip-tests; then
        gpuci_logger "Skipping Tests"
        exit 0
    fi

    gpuci_logger "Check GPU usage"
    nvidia-smi

    gpuci_logger "Update binaries"
    cd $LIBCUML_BUILD_DIR
    chrpath -d libcuml.so
    chrpath -d libcuml++.so
    patchelf --replace-needed `patchelf --print-needed libcuml++.so | grep faiss` libfaiss.so libcuml++.so

    gpuci_logger "GoogleTest for libcuml"
    cd $LIBCUML_BUILD_DIR
    chrpath -d ./test/ml
    patchelf --replace-needed `patchelf --print-needed ./test/ml | grep faiss` libfaiss.so ./test/ml
    GTEST_OUTPUT="xml:${WORKSPACE}/test-results/libcuml_cpp/" ./test/ml

    CONDA_FILE=`find ${CONDA_ARTIFACT_PATH} -name "libcuml*.tar.bz2"`
    CONDA_FILE=`basename "$CONDA_FILE" .tar.bz2` #get filename without extension
    CONDA_FILE=${CONDA_FILE//-/=} #convert to conda install
    gpuci_logger "Installing $CONDA_FILE"
    conda install -c ${CONDA_ARTIFACT_PATH} "$CONDA_FILE"

    gpuci_logger "Building cuml"
    "$WORKSPACE/build.sh" -v cuml --codecov

    gpuci_logger "Python pytest for cuml"
    cd $WORKSPACE/python

    pytest --cache-clear --basetemp=${WORKSPACE}/cuml-cuda-tmp --junitxml=${WORKSPACE}/junit-cuml.xml -v -s -m "not memleak" --durations=50 --timeout=300 --ignore=cuml/test/dask --ignore=cuml/raft --cov-config=.coveragerc --cov=cuml --cov-report=xml:${WORKSPACE}/python/cuml/cuml-coverage.xml --cov-report term

    timeout 7200 sh -c "pytest cuml/test/dask --cache-clear --basetemp=${WORKSPACE}/cuml-mg-cuda-tmp --junitxml=${WORKSPACE}/junit-cuml-mg.xml -v -s -m 'not memleak' --durations=50 --timeout=300 --cov-config=.coveragerc --cov=cuml --cov-report=xml:${WORKSPACE}/python/cuml/cuml-dask-coverage.xml --cov-report term"

    ################################################################################
    # TEST - Run notebook tests
    ################################################################################

    gpuci_logger "Notebook tests"
    set +e -Eo pipefail
    EXITCODE=0
    trap "EXITCODE=1" ERR

    ${WORKSPACE}/ci/gpu/test-notebooks.sh 2>&1 | tee nbtest.log
    python ${WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log

    ################################################################################
    # TEST - Run GoogleTest for ml-prims
    ################################################################################

    gpuci_logger "Run ml-prims test"
    cd $LIBCUML_BUILD_DIR
    chrpath -d ./test/prims
    patchelf --replace-needed `patchelf --print-needed ./test/prims | grep faiss` libfaiss.so ./test/prims
    GTEST_OUTPUT="xml:${WORKSPACE}/test-results/prims/" ./test/prims

    ################################################################################
    # TEST - Run GoogleTest for ml-prims, but with cuda-memcheck enabled
    ################################################################################

    if [ "$BUILD_MODE" = "branch" ] && [ "$BUILD_TYPE" = "gpu" ]; then
        logger "GoogleTest for ml-prims with cuda-memcheck enabled..."
        cd $WORKSPACE/ci/artifacts/cuml/cpu/conda_work/cpp/build
        python ../scripts/cuda-memcheck.py -tool memcheck -exe ./test/prims
    fi

    gpuci_logger "Building doxygen C++ docs"
    #Need to run in standard directory, not our artifact dir
    unset LIBCUML_BUILD_DIR
    $WORKSPACE/build.sh cppdocs -v

fi

if [ -n "${CODECOV_TOKEN}" ]; then

    # NOTE: The code coverage upload needs to work for both PR builds and normal
    # branch builds (aka `branch-0.XX`). Ensure the following settings to the
    # codecov CLI will work with and without a PR
    gpuci_logger "Uploading Code Coverage to codecov.io"

    # Directory containing reports
    REPORT_DIR="${WORKSPACE}/python/cuml"

    # Base name to use in Codecov UI
    CODECOV_NAME=${JOB_BASE_NAME:-"${OS},py${PYTHON},cuda${CUDA}"}

    # Codecov args needed by both calls
    EXTRA_CODECOV_ARGS="-c"

    # Save the OS PYTHON and CUDA flags
    EXTRA_CODECOV_ARGS="${EXTRA_CODECOV_ARGS} -e OS,PYTHON,CUDA"

    # If we have REPORT_HASH, use that instead. This fixes an issue where
    # CodeCov uses a local merge commit created by Jenkins. Since this commit
    # never gets pushed, it causes issues in Codecov
    if [ -n "${REPORT_HASH}" ]; then
        EXTRA_CODECOV_ARGS="${EXTRA_CODECOV_ARGS} -C ${REPORT_HASH}"
    fi

    # Append the PR ID. This is needed when running the build inside docker for
    # PR builds
    if [ -n "${PR_ID}" ]; then
        EXTRA_CODECOV_ARGS="${EXTRA_CODECOV_ARGS} -P ${PR_ID}"
    fi

    # Set the slug since this does not work in jenkins.
    export CODECOV_SLUG="${PR_AUTHOR:-"rapidsai"}/cuml"

    # Upload the two reports with separate flags. Delete the report on success
    # to prevent further CI steps from re-uploading
    curl -s https://codecov.io/bash | bash -s -- -F non-dask -f ${REPORT_DIR}/cuml-coverage.xml -n "$CODECOV_NAME,non-dask" ${EXTRA_CODECOV_ARGS}
    curl -s https://codecov.io/bash | bash -s -- -F dask -f ${REPORT_DIR}/cuml-dask-coverage.xml -n "$CODECOV_NAME,dask" ${EXTRA_CODECOV_ARGS}
fi

return ${EXITCODE}
