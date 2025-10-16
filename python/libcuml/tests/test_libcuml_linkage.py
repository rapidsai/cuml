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
    """Find libcuml++.so in the installation directory.

    Returns tuple of (libcuml_so_path, installation_root)
    """
    # Try to find libcuml++.so using the libcuml module location first
    try:
        import libcuml

        libcuml_module_path = Path(libcuml.__file__).parent
        # libcuml++.so should be in lib64 subdirectory
        libcuml_so = libcuml_module_path / "lib64" / "libcuml++.so"

        # If libcuml++.so doesn't exist at the module location, it means we're importing
        # the source code instead of the installed wheel. Try site-packages directly.
        if not libcuml_so.exists():
            for site_pkg in sys.path:
                if "site-packages" in site_pkg:
                    potential_path = (
                        Path(site_pkg) / "libcuml" / "lib64" / "libcuml++.so"
                    )
                    if potential_path.exists():
                        libcuml_so = potential_path
                        libcuml_module_path = potential_path.parent.parent
                        break

        if libcuml_so.exists():
            # Find the installation root by walking up from site-packages
            # Typical structure: {installation_root}/lib/pythonX.Y/site-packages/libcuml
            # or just: {installation_root}/lib/site-packages/libcuml (for system Python)

            # Check if VIRTUAL_ENV is set and use that as the root
            venv = os.environ.get("VIRTUAL_ENV")
            if venv:
                installation_root = Path(venv)
            else:
                installation_root = Path(sys.prefix)

            return libcuml_so, installation_root
    except (ImportError, AttributeError):
        pass

    # Fallback: try VIRTUAL_ENV if set
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        venv_path = Path(venv)
        libcuml_paths = list(venv_path.rglob("libcuml++.so"))
        if libcuml_paths:
            return libcuml_paths[0], venv_path

    pytest.fail("libcuml++.so not found. Please ensure libcuml is installed.")


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
    libcuml_so_path, installation_root = find_libcuml_so()
    print(f"Found libcuml++.so at: {libcuml_so_path}")
    print(f"Installation root: {installation_root}")

    import libcuml  # noqa: F401

    result = subprocess.run(
        ["ldd", str(libcuml_so_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    ldd_output = result.stdout
    print(f"ldd output:\n{ldd_output}")

    linked_libs = parse_ldd_output(ldd_output)

    # Determine CUDA version from CI container environment variable
    rapids_cuda_version = os.environ.get("RAPIDS_CUDA_VERSION", "")
    cuda_major_version = int(rapids_cuda_version.split(".")[0])
    is_ctk_13_plus = cuda_major_version >= 13

    # Define expected library paths
    # For CTK 13: nvidia/cu13/lib and nvidia/nccl/lib
    # For CTK 12: individual nvidia/{library}/lib directories
    # The libcuml++.so is at: site-packages/libcuml/lib64/libcuml++.so
    # So relative paths from lib64 are: ../../nvidia/{cu13|library}/lib
    if is_ctk_13_plus:
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

    failures = []

    for lib_name, expected_path_suffix in expected_libs.items():
        if lib_name not in linked_libs:
            failures.append(f"Library {lib_name} not found in ldd output")
            continue

        actual_path = linked_libs[lib_name]

        if actual_path == "not found":
            # librmm.so and librapids_logger.so may not be found
            # but are loaded dynamically at runtime, skip these
            if lib_name in ["librmm.so", "librapids_logger.so"]:
                continue
            failures.append(f"Library {lib_name} => not found")
            continue

        if not actual_path.startswith(str(installation_root)):
            failures.append(
                f"Library {lib_name} is not in installation directory:\n"
                f"  Expected path to start with: {installation_root}\n"
                f"  Actual path: {actual_path}"
            )
            continue

        if expected_path_suffix not in actual_path:
            failures.append(
                f"Library {lib_name} does not have expected path suffix:\n"
                f"  Expected suffix: {expected_path_suffix}\n"
                f"  Actual path: {actual_path}"
            )

    if failures:
        failure_msg = "\n\n".join(failures)
        pytest.fail(f"Library linkage validation failed:\n\n{failure_msg}")
