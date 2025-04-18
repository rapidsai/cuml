#!/bin/bash

# Copyright (c) 2019-2025, NVIDIA CORPORATION.

# cuml build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd "$(dirname $0)"; pwd)

VALIDTARGETS="clean libcuml cuml cuml-cpu cpp-mgtests prims bench prims-bench cppdocs pydocs"
VALIDFLAGS="-v -g -n --allgpuarch --singlegpu --nolibcumltest --nvtx --show_depr_warn --codecov --ccache --configure-only -h --help "
VALIDARGS="${VALIDTARGETS} ${VALIDFLAGS}"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean             - remove all existing build artifacts and configuration (start over)
   libcuml           - build the cuml C++ code only. Also builds the C-wrapper library
                       around the C++ code.
   cuml              - build the cuml Python package
   cuml-cpu          - build the cuml CPU Python package
   cpp-mgtests       - build libcuml mnmg tests. Builds MPI communicator, adding MPI as dependency.
   prims             - build the ml-prims tests
   bench             - build the libcuml C++ benchmark
   prims-bench       - build the ml-prims C++ benchmark
   cppdocs           - build the C++ API doxygen documentation
   pydocs            - build the general and Python API documentation
 and <flag> is:
   -v                - verbose build mode
   -g                - build for debug
   -n                - no install step
   -h                - print this text
   --allgpuarch      - build for all supported GPU architectures
   --singlegpu       - Build libcuml and cuml without multigpu components
   --nolibcumltest   - disable building libcuml C++ tests for a faster build
   --nvtx            - Enable nvtx for profiling support
   --show_depr_warn  - show cmake deprecation warnings
   --codecov         - Enable code coverage support by compiling with Cython linetracing
                       and profiling enabled (WARNING: Impacts performance)
   --ccache          - Use ccache to cache previous compilations
   --configure-only  - Invoke CMake without actually building
   --nocloneraft     - CMake will clone RAFT even if it is in the environment, use this flag to disable that behavior
   --static-treelite - Force CMake to use the Treelite static libs, cloning and building them if necessary

 default action (no args) is to build and install 'libcuml', 'cuml', and 'prims' targets only for the detected GPU arch

 The following environment variables are also accepted to allow further customization:
   PARALLEL_LEVEL         - Number of parallel threads to use in compilation.
   CUML_EXTRA_CMAKE_ARGS  - Extra arguments to pass directly to cmake. Values listed in environment
                            variable will override existing arguments. Example:
                            CUML_EXTRA_CMAKE_ARGS=\"-DBUILD_CUML_C_LIBRARY=OFF\" ./build.sh
   CUML_EXTRA_PYTHON_ARGS - Extra arguments to pass directly to pip install
"
LIBCUML_BUILD_DIR=${LIBCUML_BUILD_DIR:=${REPODIR}/cpp/build}
CUML_BUILD_DIR=${REPODIR}/python/cuml/build
PYTHON_DEPS_CLONE=${REPODIR}/python/external_repositories
BUILD_DIRS="${LIBCUML_BUILD_DIR} ${CUML_BUILD_DIR} ${PYTHON_DEPS_CLONE}"

# Set defaults for vars modified by flags to this script
VERBOSE=""
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_ALL_GPU_ARCH=0
SINGLEGPU_CPP_FLAG=""
CUML_EXTRA_PYTHON_ARGS=${CUML_EXTRA_PYTHON_ARGS:=""}
NVTX=OFF
CCACHE=OFF
CLEAN=0
BUILD_DISABLE_DEPRECATION_WARNINGS=ON
BUILD_CUML_STD_COMMS=ON
BUILD_CUML_TESTS=ON
BUILD_CUML_MG_TESTS=OFF
BUILD_STATIC_TREELITE=OFF
CMAKE_LOG_LEVEL=WARNING

# Set defaults for vars that may not have been defined externally
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX:=$LIBCUML_BUILD_DIR/install}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}

# Default to Ninja if generator is not specified
export CMAKE_GENERATOR="${CMAKE_GENERATOR:=Ninja}"

# Allow setting arbitrary cmake args via the $CUML_ADDL_CMAKE_ARGS variable. Any
# values listed here will override existing arguments. For example:
# CUML_EXTRA_CMAKE_ARGS="-DBUILD_CUML_C_LIBRARY=OFF" ./build.sh
# Will disable building the C library even though it is hard coded to ON
CUML_EXTRA_CMAKE_ARGS=${CUML_EXTRA_CMAKE_ARGS:=""}

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Check for valid usage
if (( NUMARGS != 0 )); then
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option or formatting, check --help: ${a}"
        exit 1
    fi
    done
fi

function completeBuild {
    (( NUMARGS == 0 )) && return
    for a in ${ARGS}; do
        if (echo " ${VALIDTARGETS} " | grep -q " ${a} "); then
          false; return
        fi
    done
    true
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

if hasArg clean; then
    CLEAN=1
fi

if hasArg cpp-mgtests; then
    BUILD_CUML_MG_TESTS=ON
fi

# Long arguments
LONG_ARGUMENT_LIST=(
    "verbose"
    "debug"
    "no-install"
    "allgpuarch"
    "singlegpu"
    "nvtx"
    "show_depr_warn"
    "codecov"
    "ccache"
    "nolibcumltest"
    "nocloneraft"
    "configure-only"
)

# Short arguments
ARGUMENT_LIST=(
    "v"
    "g"
    "n"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s," "${LONG_ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "$(printf "%s" "${ARGUMENT_LIST[@]}")" \
    -- "$@"
)

if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

eval set -- "$opts"

while true; do
    case "$1" in
        -h)
            show_help
            exit 0
            ;;
        -v | --verbose )
            VERBOSE_FLAG="-v"
            CMAKE_LOG_LEVEL=VERBOSE
            ;;
        -g | --debug )
            BUILD_TYPE=RelWithDebInfo
            ;;
        -n | --no-install )
            INSTALL_TARGET=""
            ;;
        --allgpuarch )
            BUILD_ALL_GPU_ARCH=1
            ;;
        --singlegpu )
            CUML_EXTRA_PYTHON_ARGS="${CUML_EXTRA_PYTHON_ARGS} --singlegpu"
            SINGLEGPU_CPP_FLAG=ON
            ;;
        --nvtx )
            NVTX=ON
            ;;
        --show_depr_warn )
            BUILD_DISABLE_DEPRECATION_WARNINGS=OFF
            ;;
        --codecov )
            CUML_EXTRA_PYTHON_ARGS="${CUML_EXTRA_PYTHON_ARGS} --linetrace=1 --profile"
            ;;
        --ccache )
            CCACHE=ON
            ;;
        --nolibcumltest )
            BUILD_CUML_TESTS=OFF
            ;;
        --nocloneraft )
            DISABLE_FORCE_CLONE_RAFT=ON
            ;;
        --static-treelite )
            BUILD_STATIC_TREELITE=ON
            ;;
        --)
            shift
            break
            ;;
    esac
    shift
done


# If clean given, run it prior to any other steps
if (( ${CLEAN} == 1 )); then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
        if [ -d ${bd} ]; then
            find ${bd} -mindepth 1 -delete
            rmdir ${bd} || true
        fi
    done

    # Clean up python artifacts
    find ${REPODIR}/python/ | grep -E "(__pycache__|\.pyc|\.pyo|\.so|\_skbuild)$"  | xargs rm -rf

    # Remove Doxyfile
    rm -rf ${REPODIR}/cpp/Doxyfile

    # Remove .benchmark dirs and .pytest_cache
    find ${REPODIR}/ | grep -E "(\.pytest_cache|\.benchmarks)$"  | xargs rm -rf
fi


################################################################################
# Configure for building all C++ targets
if completeBuild || hasArg libcuml || hasArg prims || hasArg bench || hasArg prims-bench || hasArg cppdocs || hasArg cpp-mgtests; then
    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        CUML_CMAKE_CUDA_ARCHITECTURES="NATIVE"
        echo "Building for the architecture of the GPU in the system..."
    else
        CUML_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
        echo "Building for *ALL* supported GPU architectures..."
    fi

    mkdir -p ${LIBCUML_BUILD_DIR}
    cd ${LIBCUML_BUILD_DIR}

    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${CUML_CMAKE_CUDA_ARCHITECTURES} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DBUILD_CUML_C_LIBRARY=ON \
          -DSINGLEGPU=${SINGLEGPU_CPP_FLAG} \
          -DCUML_ALGORITHMS="ALL" \
          -DBUILD_CUML_TESTS=${BUILD_CUML_TESTS} \
          -DBUILD_CUML_MPI_COMMS=${BUILD_CUML_MG_TESTS} \
          -DBUILD_CUML_MG_TESTS=${BUILD_CUML_MG_TESTS} \
          -DCUML_USE_TREELITE_STATIC=${BUILD_STATIC_TREELITE} \
          -DNVTX=${NVTX} \
          -DUSE_CCACHE=${CCACHE} \
          -DDISABLE_DEPRECATION_WARNINGS=${BUILD_DISABLE_DEPRECATION_WARNINGS} \
          -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
          -DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_LOG_LEVEL} \
          ${CUML_EXTRA_CMAKE_ARGS} \
          ..
fi

# If `./build.sh cuml` is called, don't build C/C++ components
if (! hasArg --configure-only) && (completeBuild || hasArg libcuml || hasArg prims || hasArg bench || hasArg cpp-mgtests); then
    cd ${LIBCUML_BUILD_DIR}
    if [ -n "${INSTALL_TARGET}" ]; then
      cmake --build ${LIBCUML_BUILD_DIR} -j${PARALLEL_LEVEL} ${build_args} --target ${INSTALL_TARGET} ${VERBOSE_FLAG}
    else
      cmake --build ${LIBCUML_BUILD_DIR} -j${PARALLEL_LEVEL} ${build_args} ${VERBOSE_FLAG}
    fi
fi

if (! hasArg --configure-only) && hasArg cppdocs; then
    cd ${LIBCUML_BUILD_DIR}
    cmake --build ${LIBCUML_BUILD_DIR} --target docs_cuml
fi


# Build and (optionally) install the cuml Python package
if (! hasArg --configure-only) && (completeBuild || hasArg cuml || hasArg pydocs); then
    # Replace spaces with semicolons in SKBUILD_EXTRA_CMAKE_ARGS
    SKBUILD_EXTRA_CMAKE_ARGS=$(echo ${SKBUILD_EXTRA_CMAKE_ARGS} | sed 's/ /;/g')

    SKBUILD_CMAKE_ARGS="-DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_LOG_LEVEL};${SKBUILD_EXTRA_CMAKE_ARGS}" \
        python -m pip install --no-build-isolation --no-deps --config-settings rapidsai.disable-cuda=true ${REPODIR}/python/cuml

    if hasArg pydocs; then
        cd ${REPODIR}/docs
        make html
    fi
fi

if hasArg cuml-cpu; then
    SKBUILD_CMAKE_ARGS="-DCUML_CPU=ON;-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE" \
        python -m pip install --no-build-isolation --no-deps --config-settings rapidsai.disable-cuda=true ${REPODIR}/python/cuml
fi
