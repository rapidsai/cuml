# Testing cuML Build Instructions

This directory contains `test-build.sh`, a script to validate the manual build workflow from BUILD.md in a clean Docker container.

## Quick Start

### Option 1: Run script directly (from cuML repo root)

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvidia/cuda:12.6.0-devel-ubuntu22.04 \
  bash /workspace/test-build.sh
```

### Option 2: Interactive testing

```bash
# Start the container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvidia/cuda:12.6.0-devel-ubuntu22.04 \
  bash

# Inside the container, run the script
cd /workspace
./test-build.sh
```

### Option 3: With basic functional tests

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -e RUN_TESTS=1 \
  nvidia/cuda:12.6.0-devel-ubuntu22.04 \
  bash /workspace/test-build.sh
```

## What the script does

The `test-build.sh` script validates all steps from the "Manual Process" section in BUILD.md:

1. **Installs system prerequisites** - git, wget, build-essential
2. **Installs Miniforge** - conda package manager
3. **Creates conda environment** - installs **only** the dependencies explicitly listed in BUILD.md:
   - CUDA toolkit
   - gcc >= 13.0
   - cmake >= 3.30.4
   - ninja
   - Python 3.11
   - Cython >= 3.0.0
   - RAPIDS ecosystem libraries (librmm, libraft-headers, libcuvs, libcumlprims, cudf) - matching cuML version
   - External libraries (treelite 4.4.1, rapids-logger 0.2.x)

   **Note:** The script deliberately avoids using `conda/environments/all_*.yaml` files to ensure BUILD.md documentation is complete and accurate.

4. **Builds libcuml++** - C++/CUDA library using the manual cmake workflow
5. **Builds Python package** - cuml Python package using setup.py
6. **Verifies installation** - imports cuml and checks basic functionality

## Requirements

- Docker with NVIDIA GPU support (nvidia-docker or Docker with `--gpus` flag)
- At least one NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- Sufficient disk space (~10GB for build artifacts)
- Internet connection (for downloading conda packages)

## Customization

You can customize the build by setting environment variables:

```bash
# Use more/fewer parallel jobs
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -e PARALLEL_LEVEL=4 \
  nvidia/cuda:12.6.0-devel-ubuntu22.04 \
  bash /workspace/test-build.sh

# Enable basic functional tests
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -e RUN_TESTS=1 \
  nvidia/cuda:12.6.0-devel-ubuntu22.04 \
  bash /workspace/test-build.sh
```

## Expected Output

On success, you should see:

```
==========================================
BUILD TEST COMPLETED SUCCESSFULLY!
==========================================

Summary:
  - System prerequisites: ✓
  - Miniforge/conda: ✓
  - Conda environment: ✓
  - libcuml++ build: ✓
  - cuml Python package: ✓
  - Import test: ✓

The manual build workflow from BUILD.md has been validated.
```

## Troubleshooting

### Out of memory errors
- Reduce parallel jobs: `-e PARALLEL_LEVEL=2`
- Use a machine with more RAM (recommended: 16GB+)

### GPU not found during tests
- Ensure `--gpus all` is in the docker run command
- Verify nvidia-docker is properly installed: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`

### Build failures due to missing dependencies
- **This is the key test!** If the build fails because of a missing dependency, it means BUILD.md is incomplete
- Check the error message to identify the missing package
- Add the missing dependency to BUILD.md
- Update the `conda install` command in the test script accordingly
- Re-run the test

### RAPIDS version mismatch
- The script auto-detects the cuML version and installs matching RAPIDS libraries
- For development branches, it uses `rapidsai-nightly` channel
- If packages are not found, you may need to adjust the RAPIDS_CHANNEL in the script

### Build failures (other causes)
- Check that you're mounting the correct directory
- Ensure the cuML repository is complete (not a partial checkout)
- Check disk space: `df -h`

## Alternative Docker Images

You can test with different base images:

```bash
# CUDA 12.5
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvidia/cuda:12.5.0-devel-ubuntu22.04 \
  bash /workspace/test-build.sh

# CUDA 13.0 (if available)
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvidia/cuda:13.0.0-devel-ubuntu22.04 \
  bash /workspace/test-build.sh

# Ubuntu 20.04
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvidia/cuda:12.6.0-devel-ubuntu20.04 \
  bash /workspace/test-build.sh
```

## Notes

- The script uses `GPU_ARCHS="70"` (Volta only) to speed up compilation during testing
- Tests and benchmarks are disabled by default to reduce build time
- The script uses `set -e` and will exit on first error
- Use `set -x` output for debugging failed builds
