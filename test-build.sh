#!/bin/bash
#
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set -e  # Exit on error
set -x  # Print commands

################################################################################
# test-build.sh
#
# This script tests the manual build workflow from BUILD.md on a clean
# Docker container. It's designed to run inside:
#   nvidia/cuda:12.6.0-devel-ubuntu22.04
#
# Usage:
#   docker run --gpus all -it --rm \
#     -v $(pwd):/workspace \
#     nvidia/cuda:12.6.0-devel-ubuntu22.04 \
#     bash /workspace/test-build.sh
#
# Or interactively:
#   docker run --gpus all -it --rm \
#     -v $(pwd):/workspace \
#     nvidia/cuda:12.6.0-devel-ubuntu22.04 \
#     bash
#   # Then inside container:
#   cd /workspace && bash test-build.sh
################################################################################

echo "=========================================="
echo "cuML Build Test Script"
echo "=========================================="
echo "Testing manual build workflow from BUILD.md"
echo ""

# Detect if we're in a container
if [ ! -f /.dockerenv ] && [ ! -f /run/.containerenv ]; then
    echo "WARNING: This script is designed to run inside a Docker container"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

################################################################################
# Step 1: Install system prerequisites
################################################################################
echo "=========================================="
echo "Step 1: Installing system prerequisites"
echo "=========================================="

export DEBIAN_FRONTEND=noninteractive

# First, ensure we have basic tools and update apt cache
echo "Updating apt cache..."
apt-get update || {
    echo "WARNING: apt-get update failed, trying to fix..."
    # Sometimes the CUDA images have stale repositories
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    apt-get update
}

echo "Installing system prerequisites..."
apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential || {
    echo "ERROR: Failed to install system prerequisites"
    echo "This might be a repository configuration issue in the Docker image"
    exit 1
}

################################################################################
# Step 2: Install Miniforge (conda)
################################################################################
echo "=========================================="
echo "Step 2: Installing Miniforge"
echo "=========================================="

if [ ! -d "$HOME/miniforge3" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p $HOME/miniforge3
    rm /tmp/miniforge.sh
fi

# Initialize conda
echo "Initializing conda..."
source $HOME/miniforge3/etc/profile.d/conda.sh || {
    echo "ERROR: Failed to initialize conda"
    exit 1
}
conda --version
echo "Conda initialized successfully"

################################################################################
# Step 3: Create conda environment with dependencies
################################################################################
echo "=========================================="
echo "Step 3: Creating conda environment"
echo "=========================================="
echo "Installing dependencies based ONLY on BUILD.md (not using environment yaml files)"
echo ""

# Determine which directory we're in
if [ -d "/workspace" ] && [ -f "/workspace/BUILD.md" ]; then
    CUML_ROOT="/workspace"
    echo "Using mounted workspace at /workspace"
elif [ -f "BUILD.md" ]; then
    CUML_ROOT="$(pwd)"
    echo "Using current directory: $CUML_ROOT"
else
    echo "ERROR: Cannot find BUILD.md. Please run this script from cuML root or mount it to /workspace"
    exit 1
fi

cd "$CUML_ROOT"

# Get the cuML version to install matching RAPIDS libraries
CUML_VERSION=$(grep -oP 'version = "\K[^"]+' python/cuml/pyproject.toml || echo "25.02")
CUML_MAJOR_MINOR=$(echo $CUML_VERSION | cut -d. -f1,2)
echo "Detected cuML version: $CUML_VERSION (will install RAPIDS libraries version $CUML_MAJOR_MINOR)"
echo ""

# Create environment with Python only
# BUILD.md specifies: Python (>= 3.10 and <= 3.13)
echo "Creating base conda environment with Python 3.11..."
conda create -n cuml_test_build python=3.11 -y

# Activate the environment (need to temporarily disable exit-on-error for conda activate)
echo "Activating conda environment..."
set +e  # Temporarily disable exit on error
conda activate cuml_test_build
ACTIVATE_STATUS=$?
set -e  # Re-enable exit on error
if [ $ACTIVATE_STATUS -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

echo ""
echo "Installing dependencies from BUILD.md:"
echo "  Software Dependencies:"
echo "    - cmake >= 3.30.4"
echo "    - ninja"
echo "    - gcc >= 13.0"
echo "    - Cython >= 3.0.0"
echo "  RAPIDS Ecosystem Libraries (version $CUML_MAJOR_MINOR):"
echo "    - librmm"
echo "    - libraft-headers"
echo "    - libcuvs"
echo "    - libcumlprims"
echo "    - cudf"
echo "  Other External Libraries:"
echo "    - treelite (4.4.1)"
echo "    - rapids-logger (0.2.x)"
echo ""

# Install all dependencies based on BUILD.md
# Always use rapidsai stable channel
RAPIDS_CHANNEL="rapidsai"
echo "Using rapidsai stable channel for version $CUML_MAJOR_MINOR"

echo "Running conda install (this may take several minutes)..."
conda install -y -c conda-forge -c nvidia -c ${RAPIDS_CHANNEL} \
    "cmake>=3.30.4" \
    ninja \
    "gcc_linux-64>=13.0" \
    "gxx_linux-64>=13.0" \
    "cython>=3.0.0" \
    cuda-version=12.6 \
    "librmm=${CUML_MAJOR_MINOR}" \
    "libraft-headers=${CUML_MAJOR_MINOR}" \
    "libcuvs=${CUML_MAJOR_MINOR}" \
    "libcumlprims=${CUML_MAJOR_MINOR}" \
    "cudf=${CUML_MAJOR_MINOR}" \
    "treelite=4.4.1" \
    "rapids-logger=0.2" || {
    echo "ERROR: Failed to install conda dependencies"
    echo "This might indicate:"
    echo "  - BUILD.md is missing dependencies"
    echo "  - Package versions are incompatible"
    echo "  - Network connectivity issues"
    exit 1
}

echo ""
echo "Conda environment created and activated successfully"
echo ""
echo "Verifying key dependencies are installed:"
which cmake && cmake --version | head -1
which ninja && echo "ninja: $(ninja --version)"
which gcc && gcc --version | head -1
python -c "import cython; print(f'cython: {cython.__version__}')"

echo ""
echo "Installed packages (first 30):"
conda list | head -30

################################################################################
# Step 4: Build libcuml++ (C++/CUDA library) - Manual Process
################################################################################
echo "=========================================="
echo "Step 4: Building libcuml++ (manual process)"
echo "=========================================="

cd "$CUML_ROOT/cpp"

# Clean any previous builds
if [ -d "build" ]; then
    echo "Removing previous build directory"
    rm -rf build
fi

mkdir build
cd build

# Set CUDA_BIN_PATH if needed (though it should be in PATH already)
if [ -z "$CUDA_BIN_PATH" ]; then
    export CUDA_BIN_PATH=${CUDA_HOME:-/usr/local/cuda}
fi

echo "CUDA_BIN_PATH: $CUDA_BIN_PATH"
echo "CMAKE version:"
cmake --version
echo "GCC version:"
gcc --version | head -1
echo "NVCC version:"
nvcc --version | grep release

# Configure with cmake
# Using GPU_ARCHS for faster build (Volta only for testing)
# Disable building tests and examples to speed up the build
echo "Running cmake configuration..."
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DGPU_ARCHS="70" \
    -DBUILD_CUML_TESTS=OFF \
    -DBUILD_CUML_MG_TESTS=OFF \
    -DBUILD_CUML_EXAMPLES=OFF \
    -DBUILD_CUML_BENCH=OFF \
    -DBUILD_PRIMS_TESTS=OFF || {
    echo "ERROR: CMake configuration failed"
    echo "This might indicate missing dependencies or configuration issues"
    exit 1
}

# Build
echo ""
echo "Building libcuml++ and libcuml..."
PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(nproc)}
echo "Using $PARALLEL_LEVEL parallel jobs"
make -j${PARALLEL_LEVEL} || {
    echo "ERROR: Build failed"
    echo "Try reducing PARALLEL_LEVEL or check build logs for details"
    exit 1
}

# Install
echo ""
echo "Installing libcuml++ and libcuml..."
make install || {
    echo "ERROR: Installation failed"
    exit 1
}

echo "libcuml++ build and installation complete"

# Verify libraries were installed
echo "Checking installed libraries:"
ls -lh $CONDA_PREFIX/lib/libcuml* || echo "WARNING: libcuml libraries not found in expected location"

################################################################################
# Step 5: Build cuml Python package
################################################################################
echo "=========================================="
echo "Step 5: Building cuml Python package"
echo "=========================================="

cd "$CUML_ROOT/python"

echo "Building cuml Python package..."
python setup.py build_ext --inplace || {
    echo "ERROR: Python package build failed"
    exit 1
}

echo ""
echo "Installing cuml Python package..."
python setup.py install || {
    echo "ERROR: Python package installation failed"
    exit 1
}

################################################################################
# Step 6: Verify installation
################################################################################
echo "=========================================="
echo "Step 6: Verifying installation"
echo "=========================================="

# Check if cuml can be imported
python -c "import cuml; print(f'cuML version: {cuml.__version__}')" || {
    echo "ERROR: Failed to import cuml"
    exit 1
}

# Try to get CUDA runtime info
python -c "import cuml; from cuml.internals.memory_utils import get_global_output_type; print('cuML import successful')" || {
    echo "ERROR: cuML import failed with error"
    exit 1
}

echo ""
echo "=========================================="
echo "BUILD TEST COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - System prerequisites: ✓"
echo "  - Miniforge/conda: ✓"
echo "  - Conda environment: ✓"
echo "  - libcuml++ build: ✓"
echo "  - cuml Python package: ✓"
echo "  - Import test: ✓"
echo ""
echo "The manual build workflow from BUILD.md has been validated."
echo ""

################################################################################
# Optional: Run basic tests (if requested)
################################################################################
if [ "$RUN_TESTS" = "1" ]; then
    echo "=========================================="
    echo "Running basic tests"
    echo "=========================================="

    cd "$CUML_ROOT/python"

    # Run a simple test
    python -c "
import cuml
from cuml.datasets import make_classification
from cuml.linear_model import LogisticRegression

print('Creating synthetic dataset...')
X, y = make_classification(n_samples=100, n_features=20, n_informative=10)

print('Training logistic regression model...')
model = LogisticRegression()
model.fit(X, y)

print('Making predictions...')
predictions = model.predict(X)
print(f'Predictions shape: {predictions.shape}')
print('Basic functionality test: ✓')
"
fi

exit 0
