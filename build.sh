#!/bin/bash

# Copyright (c) 2019-2022, NVIDIA CORPORATION.

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
REPODIR=$(cd $(dirname $0); pwd)

VALIDTARGETS="clean libcuml libcuml_c cuml cppmgtests cppexamples prims bench prims-bench cppdocs pydocs"
VALIDFLAGS="-v -g -n --allgpuarch --singlegpu --nolibcumltest --nvtx --show_depr_warn --codecov -h --help --cachetool"
VALIDARGS="${VALIDTARGETS} ${VALIDFLAGS}"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean             - remove all existing build artifacts and configuration (start over)
   libcuml           - build the libcuml++.so C++ library
   libcuml_c         - build the libcuml.so C library containing C wrappers around libcuml++.so
   cuml              - build the cuml Python package
   cppmgtests        - build libcuml++ mnmg tests. Builds MPI communicator, adding MPI as dependency.
   cppexamples       - build libcuml++ examples.
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
   --ccache          - Use ccache to speed up rebuilds. Deprecated, use '--cachectool ccache' insted.
   --cachetool:      - Specify one of sccache | ccache for speeding up builds and rebuilds.
   --nocloneraft     - CMake will clone RAFT even if it is in the environment, use this flag to disable that behavior
   --static-faiss    - Force CMake to use the FAISS static libs, cloning and building them if necessary
   --static-treelite - Force CMake to use the Treelite static libs, cloning and building them if necessary

 default action (no args) is to build and install 'libcuml', 'cuml', and 'prims' targets only for the detected GPU arch

 The following environment variables are also accepted to allow further customization:
   PARALLEL_LEVEL         - Number of parallel threads to use in compilation.
   CUML_EXTRA_CMAKE_ARGS  - Extra arguments to pass directly to cmake. Values listed in environment
                            variable will override existing arguments. Example:
                            CUML_EXTRA_CMAKE_ARGS=\"-DBUILD_CUML_C_LIBRARY=OFF\" ./build.sh
   CUML_EXTRA_PYTHON_ARGS - Extra argument to pass directly to python setup.py
"
LIBCUML_BUILD_DIR=${LIBCUML_BUILD_DIR:=${REPODIR}/cpp/build}
CUML_BUILD_DIR=${REPODIR}/python/build
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
CACHE_TOOL=""
CLEAN=0
BUILD_DISABLE_DEPRECATION_WARNING=ON
BUILD_LIBCUML_CPP=OFF
BUILD_LIBCUML_C=OFF
BUILD_LIBCUML_TESTS=ON
BUILD_LIBCUML_MG_TESTS=OFF
BUILD_PRIMS_TESTS=OFF
BUILD_LIBCUML_BENCH=OFF
BUILD_LIBCUML_EXAMPLES=OFF
BUILD_PRIMS_BENCH=OFF
BUILD_CUML_STD_COMMS=ON
BUILD_STATIC_FAISS=OFF
BUILD_STATIC_TREELITE=OFF
CMAKE_LOG_LEVEL=WARNING

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}

# Default to Ninja if generator is not specified
export CMAKE_GENERATOR="${CMAKE_GENERATOR:=Ninja}"

# Allow setting arbitrary cmake args via the $CUML_ADDL_CMAKE_ARGS variable. Any
# values listed here will override existing arguments. For example:
# CUML_EXTRA_CMAKE_ARGS="-DBUILD_CUML_C_LIBRARY=OFF" ./build.sh
# Will disable building the C library even though it is hard coded to ON
CUML_EXTRA_CMAKE_ARGS=${CUML_EXTRA_CMAKE_ARGS:=""}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function completeBuild {
    (( ${NUMARGS} == 0 )) && return
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
    "nolibcumltest"
    "nocloneraft"
    "ccache"
    "cachetool:"
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
            BUILD_TYPE=Debug
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
            BUILD_DISABLE_DEPRECATION_WARNING=OFF
            ;;
        --codecov )
            CUML_EXTRA_PYTHON_ARGS="${CUML_EXTRA_PYTHON_ARGS} --linetrace=1 --profile"
            ;;
        --ccache )
            echo "WARNING: Flag --ccache is deprecated, please use '--ccachetool ccache' instead."
            CACHE_TOOL="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
            ;;
        --cachetool )
            CACHE_TOOL="-DCMAKE_C_COMPILER_LAUNCHER=$2 -DCMAKE_CXX_COMPILER_LAUNCHER=$2 -DCMAKE_CUDA_COMPILER_LAUNCHER=$2"
            echo ${CACHE_TOOL}
            shift
            ;;
        --build_metrics )
            BUILD_REPORT_METRICS=ON
            ;;
        --nolibcumltest )
            BUILD_LIBCUML_TESTS=OFF
            ;;
        --nocloneraft )
            DISABLE_FORCE_CLONE_RAFT=ON
            ;;
        --static-faiss )
            BUILD_STATIC_FAISS=ON
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

    cd ${REPODIR}/python
    python setup.py clean --all
    cd ${REPODIR}
fi

if hasArg libcuml || completeBuild; then
    BUILD_LIBCUML_CPP=ON
fi
if hasArg libcuml_c || completeBuild; then
    BUILD_LIBCUML_C=ON
fi
if hasArg cpp-mgtests || completeBuild; then
    BUILD_LIBCUML_MG_TESTS=ON
fi
if hasArg cppexamples || completeBuild; then
    BUILD_LIBCUML_EXAMPLES=ON
fi
if hasArg prims || completeBuild; then
    BUILD_PRIMS_TESTS=ON
fi
if hasArg bench || completeBuild; then
    BUILD_LIBCUML_BENCH=ON
fi
if hasArg prims-bench || completeBuild; then
    BUILD_PRIMS_BENCH=ON
fi

################################################################################
# Configure for building all C++ targets
if completeBuild || hasArg libcuml || hasarg cuml_c || hasArg prims || hasArg bench || hasArg prims-bench || hasArg cppdocs || hasArg cpp-mgtests; then
    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        CUML_CMAKE_CUDA_ARCHITECTURES="NATIVE"
        echo "Building for the architecture of the GPU in the system..."
    else
        CUML_CMAKE_CUDA_ARCHITECTURES="ALL"
        echo "Building for *ALL* supported GPU architectures..."
    fi

    mkdir -p ${LIBCUML_BUILD_DIR}
    cd ${LIBCUML_BUILD_DIR}

    cmake -S $REPODIR/cpp -B ${LIBCUML_BUILD_DIR} \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${CUML_CMAKE_CUDA_ARCHITECTURES} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DBUILD_CUML_C_LIBRARY=${BUILD_LIBCUML_C} \
          -DBUILD_CUML_CPP_LIBRARY=${BUILD_LIBCUML_CPP} \
          -DBUILD_CUML_TESTS=${BUILD_LIBCUML_TESTS} \
          -DBUILD_CUML_MG_TESTS=${BUILD_LIBCUML_MG_TESTS} \
          -DBUILD_PRIMS_TESTS=${BUILD_PRIMS_TESTS} \
          -DBUILD_CUML_EXAMPLES=${BUILD_LIBCUML_EXAMPLES} \
          -DBUILD_CUML_BENCH=${BUILD_LIBCUML_BENCH} \
          -DBUILD_CUML_PRIMS_BENCH=${BUILD_PRIMS_BENCH} \
          -DSINGLEGPU=${SINGLEGPU_CPP_FLAG} \
          -DBUILD_CUML_MPI_COMMS=${BUILD_CUML_MG_TESTS} \
          -DCUML_USE_FAISS_STATIC=${BUILD_STATIC_FAISS} \
          -DCUML_USE_TREELITE_STATIC=${BUILD_STATIC_TREELITE} \
          -DNVTX=${NVTX} \
          -DDISABLE_DEPRECATION_WARNING=${BUILD_DISABLE_DEPRECATION_WARNING} \
          -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
          -DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_LOG_LEVEL} \
          ${CACHE_TOOL} \
          ${CUML_EXTRA_CMAKE_ARGS} \
          ..

    cd ${LIBCUML_BUILD_DIR}
    compile_start=$(date +%s)
    cmake --build . -j${PARALLEL_LEVEL} ${VERBOSE_FLAG}
    compile_end=$(date +%s)
    compile_total=$(( compile_end - compile_start ))

    echo "Total Compilation Time: ${compile_total}"

    if [[ ${INSTALL_TARGET} != "" ]]; then
        cmake --build . -j${PARALLEL_LEVEL} --target install ${VERBOSE_FLAG}
    fi
fi


if hasArg cppdocs; then
    cd ${LIBCUML_BUILD_DIR}
    cmake --build ${LIBCUML_BUILD_DIR} --target docs_cuml
fi


# Build and (optionally) install the cuml Python package
if completeBuild || hasArg cuml || hasArg pydocs; then
    cd ${REPODIR}/python
    if [[ ${INSTALL_TARGET} != "" ]]; then
        python setup.py build_ext -j${PARALLEL_LEVEL:-1} ${CUML_EXTRA_PYTHON_ARGS} --library-dir=${LIBCUML_BUILD_DIR} install --single-version-externally-managed --record=record.txt
    else
        python setup.py build_ext -j${PARALLEL_LEVEL:-1} ${CUML_EXTRA_PYTHON_ARGS} --library-dir=${LIBCUML_BUILD_DIR}
    fi

    if hasArg pydocs; then
        cd ${REPODIR}/docs
        make html
    fi
fi
