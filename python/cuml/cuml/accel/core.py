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

from cuml.accel.accelerator import Accelerator
from cuml.internals import logger
from cuml.internals.memory_utils import set_global_output_type

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


def _exclude_from_acceleration(module: str) -> bool:
    """Whether a module should be excluded from acceleration"""
    parts = module.split(".")
    name = parts[0]
    if name == "sklearn":
        # Exclude all of sklearn _except_ for modules like `sklearn.*.tests.test_*`
        # since we want to be able to run the sklearn test suite with cuml.accel
        # enabled
        return len(parts) < 2 or parts[-2] != "tests"

    # Exclude any module under these packages
    return name in ("umap", "hdbscan", "cuml", "treelite")


ACCEL = Accelerator(exclude=_exclude_from_acceleration)
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

            mr = rmm.mr.get_current_device_resource()
            if isinstance(mr, rmm.mr.ManagedMemoryResource):
                # Nothing to do
                pass
            elif not isinstance(mr, rmm.mr.CudaMemoryResource):
                logger.debug(
                    "cuML: A non-default memory resource is already configured, "
                    "skipping enabling managed memory."
                )
            else:
                rmm.mr.set_current_device_resource(
                    rmm.mr.ManagedMemoryResource()
                )
                logger.debug("cuML: Enabled managed memory.")
        else:
            logger.debug(
                "cuML: Could not enable managed memory on this platform."
            )

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
