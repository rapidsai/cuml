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

from cuda.bindings import runtime

from cuml.internals import logger
from cuml.internals.memory_utils import set_global_output_type
from cuml.accel.accelerator import Accelerator


ACCELERATED_MODULES = [
    "hdbscan",
    "sklearn.cluster",
    "sklearn.decomposition",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.manifold",
    "sklearn.neighbors",
    "umap",
]

ACCEL = Accelerator(["sklearn", "umap", "hdbscan", "cuml", "treelite"])
for module in ACCELERATED_MODULES:
    ACCEL.register(module, f"cuml.accel._wrappers.{module}")


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


def enabled() -> bool:
    """Returns whether the accelerator is enabled."""
    return ACCEL.enabled


def install(disable_uvm=False):
    """Enable cuML Accelerator Mode."""
    logger.set_level(logger.level_enum.info)
    logger.set_pattern("%v")

    if not disable_uvm:
        if _is_concurrent_managed_access_supported():
            import rmm

            logger.debug("cuML: Enabling managed memory...")
            rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
        else:
            logger.warn("cuML: Could not enable managed memory.")

    ACCEL.install()
    set_global_output_type("numpy")

    logger.info("cuML: Accelerator installed.")


def pytest_load_initial_conftests(early_config, parser, args):
    # https://docs.pytest.org/en/7.1.x/reference/\
    # reference.html#pytest.hookspec.pytest_load_initial_conftests
    try:
        install()
    except RuntimeError:
        raise RuntimeError(
            "An existing plugin has already loaded sklearn. Interposing failed."
        )
