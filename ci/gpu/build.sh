#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-8}
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
unset GIT_DESCRIBE_TAG

# ucx-py version
export UCX_PY_VERSION='0.30.*'

# configure numba threading library
export NUMBA_THREADING_LAYER=workqueue

# Whether to install dask nightly or stable packages
export INSTALL_DASK_MAIN=1

# Dask version to install when `INSTALL_DASK_MAIN=0`
export DASK_STABLE_VERSION="2022.12.0"

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
gpuci_mamba_retry install -c conda-forge -c rapidsai -c rapidsai-nightly -c nvidia \
      "cudatoolkit=${CUDA_REL}" \
      "cudf=${MINOR_VERSION}" \
      "rmm=${MINOR_VERSION}" \
      "libcumlprims=${MINOR_VERSION}" \
      "libraft-headers=${MINOR_VERSION}" \
      "libraft-distance=${MINOR_VERSION}" \
      "libraft-nn=${MINOR_VERSION}" \
      "pylibraft=${MINOR_VERSION}" \
      "raft-dask=${MINOR_VERSION}" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "ucx-py=${UCX_PY_VERSION}" \
      "ucx-proc=*=gpu" \
      "xgboost=1.7.1dev.rapidsai${MINOR_VERSION}" \
      "rapids-build-env=${MINOR_VERSION}.*" \
      "rapids-notebook-env=${MINOR_VERSION}.*" \
      "shap>=0.37,<=0.39"

if [ "$(arch)" = "x86_64" ]; then
    gpuci_mamba_retry install -c conda-forge -c rapidsai -c rapidsai-nightly -c nvidia \
        "rapids-doc-env=${MINOR_VERSION}.*"
fi

# https://docs.rapids.ai/maintainers/depmgmt/
# gpuci_conda_retry remove --force rapids-build-env rapids-notebook-env
# gpuci_mamba_retry install -y "your-pkg=1.0.0"

gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

function install_dask {
    # Install the conda-forge or nightly version of dask and distributed
    gpuci_logger "Install the conda-forge or nightly version of dask and distributed"
    set -x
    if [[ "${INSTALL_DASK_MAIN}" == 1 ]]; then
        gpuci_logger "gpuci_mamba_retry install -c dask/label/dev 'dask/label/dev::dask' 'dask/label/dev::distributed'"
        gpuci_mamba_retry install -c dask/label/dev "dask/label/dev::dask" "dask/label/dev::distributed"
        conda list
    else
        gpuci_logger "gpuci_mamba_retry install conda-forge::dask=={$DASK_STABLE_VERSION} conda-forge::distributed=={$DASK_STABLE_VERSION} conda-forge::dask-core=={$DASK_STABLE_VERSION} --force-reinstall"
        gpuci_mamba_retry install conda-forge::dask==$DASK_STABLE_VERSION conda-forge::distributed==$DASK_STABLE_VERSION conda-forge::dask-core==$DASK_STABLE_VERSION --force-reinstall
    fi
    set +x
}

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_logger "Building doxygen C++ docs"
    $WORKSPACE/build.sh cppdocs -v

    ################################################################################
    # BUILD - Build libcuml, cuML, and prims from source
    ################################################################################

    gpuci_logger "Build from source"
    $WORKSPACE/build.sh clean libcuml cuml prims bench -v --codecov

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

    set -x
    install_dask
    pip install "git+https://github.com/hdbscan/hdbscan.git@master" --force-reinstall --upgrade --no-deps
    set +x

    gpuci_logger "Python pytest for cuml"
    cd $WORKSPACE/python

    pytest --cache-clear --basetemp=${WORKSPACE}/cuml-cuda-tmp --junitxml=${WORKSPACE}/junit-cuml.xml -v -s -m "not memleak" --durations=50 --timeout=300 --ignore=cuml/tests/dask --ignore=cuml/raft --cov-config=.coveragerc --cov=cuml --cov-report=xml:${WORKSPACE}/python/cuml/cuml-coverage.xml --cov-report term

    timeout 7200 sh -c "pytest cuml/tests/dask --cache-clear --basetemp=${WORKSPACE}/cuml-mg-cuda-tmp --junitxml=${WORKSPACE}/junit-cuml-mg.xml -v -s -m 'not memleak' --durations=50 --timeout=300 --cov-config=.coveragerc --cov=cuml --cov-report=xml:${WORKSPACE}/python/cuml/cuml-dask-coverage.xml --cov-report term"


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
    if hasArg --skip-tests; then
        gpuci_logger "Skipping Tests"
        exit 0
    fi

    gpuci_logger "Check GPU usage"
    nvidia-smi

    gpuci_logger "Installing libcuml and libcuml-tests"
    gpuci_mamba_retry install -y -c ${CONDA_ARTIFACT_PATH} libcuml libcuml-tests

    gpuci_logger "Running libcuml test binaries"
    GTEST_ARGS="xml:${WORKSPACE}/test-results/libcuml_cpp/"
    for gt in "$CONDA_PREFIX/bin/gtests/libcuml/"*; do
        test_name=$(basename $gt)
        echo "Running gtest $test_name"
        ${gt} ${GTEST_ARGS}
        echo "Ran gtest $test_name : return code was: $?, test script exit code is now: $EXITCODE"
    done

    # TODO: Move boa install to gpuci/rapidsai
    gpuci_mamba_retry install boa

    gpuci_logger "Building and installing cuml"
    export CONDA_BLD_DIR="$WORKSPACE/.conda-bld"
    export VERSION_SUFFIX=""
    gpuci_conda_retry mambabuild --croot ${CONDA_BLD_DIR} conda/recipes/cuml -c ${CONDA_ARTIFACT_PATH} --python=${PYTHON}
    gpuci_mamba_retry install cuml -c "${CONDA_BLD_DIR}" -c "${CONDA_ARTIFACT_PATH}"

    set -x

    install_dask
    pip install "git+https://github.com/dask/dask-glm@main" --force-reinstall --no-deps
    pip install "git+https://github.com/scikit-learn-contrib/hdbscan.git@master" --force-reinstall --upgrade --no-deps
    pip install sparse

    set +x

    gpuci_logger "Python pytest for cuml"
    cd $WORKSPACE/python/cuml/tests

    pytest --cache-clear --basetemp=${WORKSPACE}/cuml-cuda-tmp --junitxml=${WORKSPACE}/junit-cuml.xml -v -s -m "not memleak" --durations=50 --timeout=300 --ignore=dask --cov-config=.coveragerc --cov=cuml --cov-report=xml:${WORKSPACE}/python/cuml/cuml-coverage.xml --cov-report term

    timeout 7200 sh -c "pytest dask --cache-clear --basetemp=${WORKSPACE}/cuml-mg-cuda-tmp --junitxml=${WORKSPACE}/junit-cuml-mg.xml -v -s -m 'not memleak' --durations=50 --timeout=300 --cov-config=.coveragerc --cov=cuml --cov-report=xml:${WORKSPACE}/python/cuml/cuml-dask-coverage.xml --cov-report term"

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
    GTEST_ARGS="xml:${WORKSPACE}/test-results/prims/"
    for gt in "$CONDA_PREFIX/bin/gtests/libcuml_prims/"*; do
        test_name=$(basename $gt)
        echo "Running gtest $test_name"
        ${gt} ${GTEST_ARGS}
        echo "Ran gtest $test_name : return code was: $?, test script exit code is now: $EXITCODE"
    done


    ################################################################################
    # TEST - Run GoogleTest for ml-prims, but with cuda-memcheck enabled
    ################################################################################

    if [ "$BUILD_MODE" = "branch" ] && [ "$BUILD_TYPE" = "gpu" ]; then
        logger "GoogleTest for ml-prims with cuda-memcheck enabled..."
        cd $WORKSPACE/ci/artifacts/cuml/cpu/conda_work/cpp/build
        python ../scripts/cuda-memcheck.py -tool memcheck -exe ./test/prims
    fi

    if [ "$(arch)" = "x86_64" ]; then
        gpuci_logger "Building doxygen C++ docs"
        #Need to run in standard directory, not our artifact dir
        unset LIBCUML_BUILD_DIR
        $WORKSPACE/build.sh cppdocs -v

        if [ "$CUDA_REL" != "11.0" ]; then
            gpuci_logger "Building python docs"
            $WORKSPACE/build.sh pydocs
        fi
    fi

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
