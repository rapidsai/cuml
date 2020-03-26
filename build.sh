#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION.

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

VALIDARGS="clean libcuml cuml prims bench prims-bench -v -g -n --allgpuarch --singlegpu --nvtx --show_depr_warn -h --help"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean            - remove all existing build artifacts and configuration (start over)
   libcuml          - build the cuml C++ code only. Also builds the C-wrapper library
                      around the C++ code.
   cuml             - build the cuml Python package
   prims            - build the ML prims tests
   bench            - build the cuml C++ benchmark
   prims-bench      - build the ml-prims C++ benchmark
 and <flag> is:
   -v               - verbose build mode
   -g               - build for debug
   -n               - no install step
   --allgpuarch     - build for all supported GPU architectures
   --singlegpu      - Build cuml without multigpu support (multigpu requires libcumlprims)
   --nvtx           - Enable nvtx for profiling support
   --show_depr_warn - show cmake deprecation warnings
   -h               - print this text

 default action (no args) is to build and install 'libcuml', 'cuml', and 'prims' targets only for the detected GPU arch
"
LIBCUML_BUILD_DIR=${REPODIR}/cpp/build
CUML_BUILD_DIR=${REPODIR}/python/build
PYTHON_DEPS_CLONE=${REPODIR}/python/external_repositories
BUILD_DIRS="${LIBCUML_BUILD_DIR} ${CUML_BUILD_DIR} ${PYTHON_DEPS_CLONE}"

# Set defaults for vars modified by flags to this script
VERBOSE=""
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_ALL_GPU_ARCH=0
SINGLEGPU=""
NVTX=OFF
CLEAN=0
BUILD_DISABLE_DEPRECATION_WARNING=ON

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}
BUILD_ABI=${BUILD_ABI:=ON}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
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
    SINGLEGPU="--singlegpu"
fi
if hasArg --nvtx; then
    NVTX=ON
fi
if hasArg --show_depr_warn; then
    BUILD_DISABLE_DEPRECATION_WARNING=OFF
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
fi

################################################################################
# Configure for building all C++ targets
if (( ${NUMARGS} == 0 )) || hasArg libcuml || hasArg prims || hasArg bench || hasArg prims-bench; then
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
          -DCMAKE_CXX11_ABI=${BUILD_ABI} \
          -DBLAS_LIBRARIES=${INSTALL_PREFIX}/lib/libopenblas.so.0 \
          ${GPU_ARCH} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DBUILD_CUML_C_LIBRARY=ON \
          -DBUILD_CUML_STD_COMMS=ON \
          -DWITH_UCX=ON \
          -DBUILD_CUML_MPI_COMMS=OFF \
          -DNVTX=${NVTX} \
          -DPARALLEL_LEVEL=${PARALLEL_LEVEL} \
          -DNCCL_PATH=${INSTALL_PREFIX} \
          -DDISABLE_DEPRECATION_WARNING=${BUILD_DISABLE_DEPRECATION_WARNING} \
          -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} ..

fi

# Run all make targets at once

MAKE_TARGETS=
if hasArg libcuml; then
    MAKE_TARGETS="${MAKE_TARGETS}cuml++ cuml ml ml_mg"
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
if (( ${NUMARGS} == 0 )) || hasArg libcuml || hasArg prims || hasArg bench; then
# If there are no targets specified when calling build.sh, it will
# just call `make -j`. This avoids a lot of extra printing
    cd ${LIBCUML_BUILD_DIR}
    make -j${PARALLEL_LEVEL} ${MAKE_TARGETS} VERBOSE=${VERBOSE} ${INSTALL_TARGET}

fi


# Build and (optionally) install the cuml Python package
if (( ${NUMARGS} == 0 )) || hasArg cuml; then

    cd ${REPODIR}/python
    if [[ ${INSTALL_TARGET} != "" ]]; then
  python setup.py build_ext --inplace ${SINGLEGPU}
  python setup.py install --single-version-externally-managed --record=record.txt ${SINGLEGPU}
    else
  python setup.py build_ext --inplace --library-dir=${LIBCUML_BUILD_DIR} ${SINGLEGPU}
    fi
fi
