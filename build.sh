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

VALIDARGS="clean libcuml cuml prims -v -g -n --buildAllGPUArch --multigpu -h --help"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean             - remove all existing build artifacts and configuration (start over)
   libcuml           - build the cuml C++ code only
   cuml              - build the cuml Python package
   prims             - build the ML prims tests
 and <flag> is:
   -v                - verbose build mode
   -g                - build for debug
   -n                - no install step
   --buildAllGPUArch - build for all supported GPU architectures
   --multigpu        - Build cuml with multigpu support (requires libcumlMG and CUDA >=10.0)
   -h                - print this text

 default action (no args) is to build and install 'libcuml', 'cuml', and 'prims' targets only for the detected GPU arch
"
LIBCUML_BUILD_DIR=${REPODIR}/cpp/build
CUML_BUILD_DIR=${REPODIR}/python/build
BUILD_DIRS="${LIBCUML_BUILD_DIR} ${CUML_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE=""
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_ALL_GPU_ARCH=0
MULTIGPU=""

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
if hasArg --buildAllGPUArch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg --multigpu; then
    MULTIGPU=--multigpu
fi

# Various build options use nvidia-smi for querying the system
HAS_NVIDIA_SMI=1
which nvidia-smi > /dev/null
if (( $? != 0 )); then
    HAS_NVIDIA_SMI=0
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
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
if (( ${NUMARGS} == 0 )) || hasArg libcuml || hasArg prims; then

    # Configure the build for the GPU type of this machine, or all (GPU_ARCH="")
    # if it cannot be detected.
    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        if (( ${HAS_NVIDIA_SMI} )); then
            GPU="$(nvidia-smi | awk '{print $4}' | sed '8!d')"
            if [[ $GPU == *"P100"* ]]; then
                GPU_ARCH="-DGPU_ARCHS=\"60\""
                echo "Building for Pascal..."
            elif [[ $GPU == *"V100"* ]]; then
                GPU_ARCH="-DGPU_ARCHS=\"70\""
                echo "Building for Volta..."
            elif [[ $GPU == *"T4"* ]]; then
                GPU_ARCH="-DGPU_ARCHS=\"75\""
                echo "Building for Turing..."
            fi
        else
            echo "nvidia-smi was not found on PATH and is needed for detecting the GPU arch"
            echo "Ensure nvidia-smi is on PATH or use --buildAllGPUArch"
            exit 1
        fi
    else
        GPU_ARCH=""
        echo "Building for *ALL* GPU architectures..."
    fi

    mkdir -p ${LIBCUML_BUILD_DIR}
    cd ${LIBCUML_BUILD_DIR}

    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CXX11_ABI=${BUILD_ABI} \
          -DBLAS_LIBRARIES=${INSTALL_PREFIX}/lib/libopenblas.a \
          ${GPU_ARCH} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..
fi

# Build and (optionally) install libcuml + tests
if (( ${NUMARGS} == 0 )) || hasArg libcuml; then

    cd ${LIBCUML_BUILD_DIR}
    make -j${PARALLEL_LEVEL} cuml++ ml ml_mg VERBOSE=${VERBOSE} ${INSTALL_TARGET}
fi

# Build and (optionally) install the cuml Python package
if (( ${NUMARGS} == 0 )) || hasArg cuml; then

    cd ${REPODIR}/python
    if [[ ${INSTALL_TARGET} != "" ]]; then
	python setup.py build_ext --inplace ${MULTIGPU}
	python setup.py install --single-version-externally-managed --record=record.txt ${MULTIGPU}
    else
	python setup.py build_ext --inplace --library-dir=${LIBCUML_BUILD_DIR} ${MULTIGPU}
    fi
fi

# Build the ML prims tests
if (( ${NUMARGS} == 0 )) || hasArg prims; then

    cd ${LIBCUML_BUILD_DIR}
    make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} prims
fi
