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

import importlib

from cuda.bindings import runtime
from cuml.internals import logger
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.memory_utils import set_global_output_type
from cuml.internals.safe_imports import UnavailableError, gpu_only_import

rmm = gpu_only_import("rmm")


def _install_for_library(library_name):
    importlib.import_module(f"cuml.accel._wrappers.{library_name}", __name__)


def _is_concurrent_managed_access_supported():
    """Check the availability of concurrent managed access (UVM).

    Note that WSL2 does not support managed memory.
    """

    # Ensure CUDA is initialized before checking cudaDevAttrConcurrentManagedAccess
    runtime.cudaFree(0)

    device_id = 0
    err, supports_managed_access = runtime.cudaDeviceGetAttribute(
        runtime.cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess, device_id
    )
    if err != runtime.cudaError_t.cudaSuccess:
        logger.error(
            f"Failed to check cudaDevAttrConcurrentManagedAccess with error {err}"
        )
        return False
    return supports_managed_access != 0


def install(disable_uvm=False):
    """Enable cuML Accelerator Mode."""
    logger.set_level(logger.level_enum.info)
    logger.set_pattern("%v")

    if not disable_uvm:
        if _is_concurrent_managed_access_supported():
            logger.debug("cuML: Enabling managed memory...")
            rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
        else:
            logger.warn("cuML: Could not enable managed memory.")

    logger.debug("cuML: Installing accelerator...")
    libraries_to_accelerate = ["sklearn", "umap", "hdbscan"]
    accelerated_libraries = []
    failed_to_accelerate = []
    for library_name in libraries_to_accelerate:
        try:
            logger.debug(
                f"cuML: Attempt to install accelerator for {library_name}..."
            )
            _install_for_library(library_name)
        except (
            ModuleNotFoundError,
            UnavailableError,
        ) as error:  # underlying package not installed (expected)
            logger.debug(
                f"cuML: Did not install accelerator for {library_name}, the underlying library is not installed: {error}"
            )
        except Exception as error:  # something else went wrong
            failed_to_accelerate.append(library_name)
            logger.error(
                f"cuML: Failed to install accelerator for {library_name}: {error}."
            )
        else:
            accelerated_libraries.append(library_name)
            logger.info(f"cuML: Installed accelerator for {library_name}.")

    GlobalSettings().accelerated_libraries = accelerated_libraries
    GlobalSettings().accelerator_loaded = any(accelerated_libraries)
    GlobalSettings().accelerator_active = any(accelerated_libraries)

    if any(accelerated_libraries) and not any(failed_to_accelerate):
        logger.info("cuML: Successfully initialized accelerator.")
    elif any(accelerated_libraries) and any(failed_to_accelerate):
        logger.warn(
            "cuML: Accelerator initialized, but failed to initialize for some libraries."
        )
    elif not any(accelerated_libraries) and not any(failed_to_accelerate):
        logger.warn(
            "cuML: Accelerator failed to initialize, because none of the underlying libraries are installed."
        )

    set_global_output_type("numpy")


def pytest_load_initial_conftests(early_config, parser, args):
    # https://docs.pytest.org/en/7.1.x/reference/\
    # reference.html#pytest.hookspec.pytest_load_initial_conftests
    try:
        install()
    except RuntimeError:
        raise RuntimeError(
            "An existing plugin has already loaded sklearn. Interposing failed."
        )
