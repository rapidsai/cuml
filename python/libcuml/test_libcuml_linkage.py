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

import os
import subprocess
import sys
from pathlib import Path

import pytest


def find_libcuml_so():
    """Find libcuml++.so in the virtual environment."""
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        pytest.skip("VIRTUAL_ENV not set")

    # Search for libcuml++.so in the virtual environment
    venv_path = Path(venv)
    libcuml_paths = list(venv_path.rglob("libcuml++.so"))

    if not libcuml_paths:
        pytest.fail(f"libcuml++.so not found in {venv}")

    # Return the first match (should only be one in a proper install)
    return libcuml_paths[0]


def parse_ldd_output(ldd_output):
    """Parse ldd output and return a dictionary of library -> path."""
    libs = {}
    for line in ldd_output.split("\n"):
        line = line.strip()
        if "=>" in line:
            parts = line.split("=>")
            if len(parts) == 2:
                lib_name = parts[0].strip()
                path_and_addr = parts[1].strip()
                # Extract path (remove address in parentheses)
                if "(" in path_and_addr:
                    path = path_and_addr.split("(")[0].strip()
                else:
                    path = path_and_addr
                libs[lib_name] = path
    return libs


def test_libcuml_linkage():
    """Test that libcuml++.so links to the correct library paths."""
    # Find libcuml++.so
    libcuml_so_path = find_libcuml_so()
    print(f"Found libcuml++.so at: {libcuml_so_path}")

    # Import libcuml to ensure it loads successfully
    import libcuml  # noqa: F401

    # Run ldd on libcuml++.so
    result = subprocess.run(
        ["ldd", str(libcuml_so_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    ldd_output = result.stdout
    print(f"ldd output:\n{ldd_output}")

    # Parse ldd output
    linked_libs = parse_ldd_output(ldd_output)

    # Get virtual environment path
    venv = os.environ.get("VIRTUAL_ENV")
    venv_path = Path(venv)

    # Define expected library paths based on CMakeLists.txt
    # For CTK 13+: nvidia/cu13/lib and nvidia/nccl/lib
    # For CTK 12 and below: individual nvidia/{library}/lib directories
    # The libcuml++.so is at: {venv}/lib/python{X.Y}/site-packages/libcuml/lib64/libcuml++.so
    # So relative paths from lib64 are: ../../nvidia/{cu13|library}/lib

    # Determine CUDA Toolkit version from environment or by auto-detection
    rapids_cuda_version = os.environ.get("RAPIDS_CUDA_VERSION", "")

    if rapids_cuda_version:
        # Parse major version from RAPIDS_CUDA_VERSION (e.g., "13.0" -> 13)
        try:
            cuda_major_version = int(rapids_cuda_version.split(".")[0])
            is_ctk_13_plus = cuda_major_version >= 13
            print(
                f"Using CUDA version from environment: {rapids_cuda_version} (CTK {cuda_major_version})"
            )
        except (ValueError, IndexError):
            print(
                f"Warning: Could not parse RAPIDS_CUDA_VERSION='{rapids_cuda_version}', falling back to auto-detection"
            )
            is_ctk_13_plus = any(
                "nvidia/cu13/lib" in path
                for path in linked_libs.values()
                if path != "not found"
            )
    else:
        # Auto-detect by checking library paths
        # CTK 13+ uses nvidia/cu13/lib, earlier versions use individual directories
        print("RAPIDS_CUDA_VERSION not set, auto-detecting from library paths")
        is_ctk_13_plus = any(
            "nvidia/cu13/lib" in path
            for path in linked_libs.values()
            if path != "not found"
        )

    # Define the libraries we expect to find with their expected path patterns
    # These are non-system libraries that should be in the venv
    if is_ctk_13_plus:
        print("Detected CTK 13+ library layout")
        expected_libs = {
            # CTK 13 libs (in nvidia/cu13/lib)
            "libcufft.so.12": "nvidia/cu13/lib",
            "libcusolver.so.12": "nvidia/cu13/lib",
            "libcublas.so.13": "nvidia/cu13/lib",
            "libcublasLt.so.13": "nvidia/cu13/lib",
            "libcusparse.so.12": "nvidia/cu13/lib",
            "libnvJitLink.so.13": "nvidia/cu13/lib",
            "libcurand.so.10": "nvidia/cu13/lib",
            # NCCL (in nvidia/nccl/lib)
            "libnccl.so.2": "nvidia/nccl/lib",
            # cumlprims_mg (in libcuml/lib64)
            "libcumlprims_mg.so": "libcuml/lib64",
        }
    else:
        print("Detected CTK 12 or earlier library layout")
        expected_libs = {
            # CTK 12 and earlier (in individual nvidia/{library}/lib directories)
            "libcufft.so.11": "nvidia/cufft/lib",
            "libcusolver.so.11": "nvidia/cusolver/lib",
            "libcublas.so.12": "nvidia/cublas/lib",
            "libcublasLt.so.12": "nvidia/cublas/lib",
            "libcusparse.so.12": "nvidia/cusparse/lib",
            "libnvJitLink.so.12": "nvidia/nvjitlink/lib",
            "libcurand.so.10": "nvidia/curand/lib",
            # NCCL (in nvidia/nccl/lib)
            "libnccl.so.2": "nvidia/nccl/lib",
            # cumlprims_mg (in libcuml/lib64)
            "libcumlprims_mg.so": "libcuml/lib64",
        }

    # Track failures
    failures = []

    for lib_name, expected_path_suffix in expected_libs.items():
        if lib_name not in linked_libs:
            failures.append(f"Library {lib_name} not found in ldd output")
            continue

        actual_path = linked_libs[lib_name]

        # Check if library was found
        if actual_path == "not found":
            # Some libraries like librmm.so and librapids_logger.so may not be found
            # but are loaded dynamically at runtime, skip these
            if lib_name in ["librmm.so", "librapids_logger.so"]:
                continue
            failures.append(f"Library {lib_name} => not found")
            continue

        # Verify the path contains the expected suffix relative to venv
        if not actual_path.startswith(str(venv_path)):
            failures.append(
                f"Library {lib_name} is not in virtual environment:\n"
                f"  Expected path to start with: {venv_path}\n"
                f"  Actual path: {actual_path}"
            )
            continue

        # Check that the path contains the expected suffix
        if expected_path_suffix not in actual_path:
            failures.append(
                f"Library {lib_name} does not have expected path suffix:\n"
                f"  Expected suffix: {expected_path_suffix}\n"
                f"  Actual path: {actual_path}"
            )

    # Report all failures at once
    if failures:
        failure_msg = "\n\n".join(failures)
        pytest.fail(f"Library linkage validation failed:\n\n{failure_msg}")

    print("All library linkage checks passed!")


if __name__ == "__main__":
    # Allow running the test directly for debugging
    sys.exit(pytest.main([__file__, "-v", "-s"]))
