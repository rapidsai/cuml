#!/bin/bash

# Copyright (c) 2019-2021, NVIDIA CORPORATION.

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

VALIDTARGETS="clean libcuml cuml cpp-mgtests prims bench prims-bench cppdocs pydocs"
VALIDFLAGS="-v -g -n --allgpuarch --buildfaiss --buildgtest --singlegpu --nvtx --show_depr_warn --codecov --ccache -h --help "
VALIDARGS="${VALIDTARGETS} ${VALIDFLAGS}"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean            - remove all existing build artifacts and configuration (start over)
   libcuml          - build the cuml C++ code only. Also builds the C-wrapper library
                      around the C++ code.
   cuml             - build the cuml Python package
   cpp-mgtests      - build libcuml mnmg tests. Builds MPI communicator, adding MPI as dependency.
   prims            - build the ML prims tests
   bench            - build the cuml C++ benchmark
   prims-bench      - build the ml-prims C++ benchmark
   cppdocs          - build the C++ API doxygen documentation
   pydocs           - build the general and Python API documentation
 and <flag> is:
   -v               - verbose build mode
   -g               - build for debug
   -n               - no install step
   --allgpuarch     - build for all supported GPU architectures
   --buildfaiss     - build faiss statically into libcuml
   --buildgtest     - build googletest library
   --singlegpu      - Build libcuml and cuml without multigpu components
   --nvtx           - Enable nvtx for profiling support
   --show_depr_warn - show cmake deprecation warnings
   --codecov        - Enable code coverage support by compiling with Cython linetracing
                      and profiling enabled (WARNING: Impacts performance)
   --ccache         - Use ccache to cache previous compilations
   -h               - print this text

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
CCACHE=OFF
CLEAN=0
BUILD_DISABLE_DEPRECATION_WARNING=ON
BUILD_CUML_STD_COMMS=ON
BUILD_CPP_MG_TESTS=OFF
BUILD_STATIC_FAISS=OFF

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}

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

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    for a in ${ARGS}; do
        if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
            echo "Invalid option: ${a}"
            exit 1
        fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE=1
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg --singlegpu; then
    CUML_EXTRA_PYTHON_ARGS="${CUML_EXTRA_PYTHON_ARGS} --singlegpu"
    SINGLEGPU_CPP_FLAG=ON
fi
if hasArg cpp-mgtests; then
    BUILD_CPP_MG_TESTS=ON
fi
if hasArg --buildfaiss; then
    BUILD_STATIC_FAISS=ON
fi
if hasArg --buildgtest; then
    BUILD_GTEST=ON
fi
if hasArg --nvtx; then
    NVTX=ON
fi
if hasArg --show_depr_warn; then
    BUILD_DISABLE_DEPRECATION_WARNING=OFF
fi
if hasArg --codecov; then
    CUML_EXTRA_PYTHON_ARGS="${CUML_EXTRA_PYTHON_ARGS} --linetrace=1 --profile"
fi
if hasArg --ccache; then
    CCACHE=ON
fi
if hasArg clean; then
    CLEAN=1
fi

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

################################################################################
# Configure for building all C++ targets
if completeBuild || hasArg libcuml || hasArg prims || hasArg bench || hasArg prims-bench || hasArg cppdocs || hasArg cpp-mgtests; then
    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        GPU_ARCH=""
        echo "Building for the architecture of the GPU in the system..."
    else
        GPU_ARCH="-DGPU_ARCHS=ALL"
        echo "Building for *ALL* supported GPU architectures..."
    fi

    mkdir -p ${LIBCUML_BUILD_DIR}
    cd ${LIBCUML_BUILD_DIR}

    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DBLAS_LIBRARIES=${INSTALL_PREFIX}/lib/libopenblas.so.0 \
          ${GPU_ARCH} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DBUILD_CUML_C_LIBRARY=ON \
          -DSINGLEGPU=${SINGLEGPU_CPP_FLAG} \
          -DWITH_UCX=ON \
          -DBUILD_CUML_MPI_COMMS=${BUILD_CPP_MG_TESTS} \
          -DBUILD_CUML_MG_TESTS=${BUILD_CPP_MG_TESTS} \
          -DBUILD_STATIC_FAISS=${BUILD_STATIC_FAISS} \
          -DNVTX=${NVTX} \
          -DPARALLEL_LEVEL=${PARALLEL_LEVEL} \
          -DUSE_CCACHE=${CCACHE} \
          -DNCCL_PATH=${INSTALL_PREFIX} \
          -DDISABLE_DEPRECATION_WARNING=${BUILD_DISABLE_DEPRECATION_WARNING} \
          -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
          ${CUML_EXTRA_CMAKE_ARGS} \
          ..
fi

# Run all make targets at once

MAKE_TARGETS=
if hasArg libcuml; then
    MAKE_TARGETS="${MAKE_TARGETS}cuml++ cuml ml"
fi
if hasArg cpp-mgtests; then
    MAKE_TARGETS="${MAKE_TARGETS} ml_mg"
fi
if hasArg prims; then
    MAKE_TARGETS="${MAKE_TARGETS} prims"
fi
if hasArg bench; then
    MAKE_TARGETS="${MAKE_TARGETS} sg_benchmark"
fi
if hasArg prims-bench; then
    MAKE_TARGETS="${MAKE_TARGETS} prims_benchmark"
fi

# If `./build.sh cuml` is called, don't build C/C++ components
if completeBuild || hasArg libcuml || hasArg prims || hasArg bench || hasArg cpp-mgtests; then
    cd ${LIBCUML_BUILD_DIR}
    build_args="--target ${MAKE_TARGETS} ${INSTALL_TARGET}"
    if [ ! -z ${VERBOSE} ]
    then
      build_args="-v ${build_args}"
    fi
    if [ ! -z ${PARALLEL_LEVEL} ]
    then
      build_args="-j${PARALLEL_LEVEL} ${build_args}"
    fi
    cmake --build ${LIBCUML_BUILD_DIR} ${build_args}
fi

if hasArg cppdocs; then
    cd ${LIBCUML_BUILD_DIR}
    cmake --build ${LIBCUML_BUILD_DIR} --target doc
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
