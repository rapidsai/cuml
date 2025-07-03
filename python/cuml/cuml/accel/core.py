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
from __future__ import annotations

import enum
from typing import Literal

from cuda.bindings import runtime

from cuml.accel.accelerator import Accelerator
from cuml.internals.memory_utils import set_global_output_type


class Logger:
    """A simple logger for use in `cuml.accel` only.

    This outputs logs on the same stream (stdout) that `cuml.internals.logger` uses, but
    critically lets us set the log level for `cuml.accel` logs separately from the level
    used by the rest of cuml. This could use python's `logging` directly, but then
    we'd have to setup a formatter and handler within our application, and the user
    experience would be identical, just with added complexity. Using a simple wrapper
    around `print` calls (what `cuml.internals.logger` effectively is) should be sufficient
    for our current needs.
    """

    class _Level(enum.IntEnum):
        ERROR = 40
        WARN = 30
        INFO = 20
        DEBUG = 10

    def __init__(self):
        self.level = Logger._Level.WARN

    def __repr__(self):
        return "<Logger level={self.level.name!r}>"

    def _log(self, msg):
        print(f"[cuml.accel] {msg}")

    def set_level(self, level: str) -> None:
        """Set the logger level.

        Parameters
        ----------
        level : {'error', 'warn', 'info', 'debug'}
            The log level to set.
        """
        self.level = Logger._Level[level.upper()]

    def error(self, msg: str) -> None:
        """Log a message at ERROR level."""
        if self.level <= Logger._Level.ERROR:
            self._log(msg)

    def warn(self, msg: str) -> None:
        """Log a message at WARN level."""
        if self.level <= Logger._Level.WARN:
            self._log(msg)

    def info(self, msg: str) -> None:
        """Log a message at INFO level."""
        if self.level <= Logger._Level.INFO:
            self._log(msg)

    def debug(self, msg: str) -> None:
        """Log a message at DEBUG level."""
        if self.level <= Logger._Level.DEBUG:
            self._log(msg)


logger = Logger()


ACCELERATED_MODULES = [
    "hdbscan",
    "sklearn.cluster",
    "sklearn.decomposition",
    "sklearn.ensemble",
    "sklearn.kernel_ridge",
    "sklearn.linear_model",
    "sklearn.manifold",
    "sklearn.neighbors",
    "sklearn.svm",
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


def install(
    disable_uvm: bool = False,
    log_level: Literal["error", "warn", "info", "debug"] = "warn",
) -> None:
    """Enable `cuml.accel`.

    Parameters
    ----------
    disable_uvm : bool, optional
        Whether to disable UVM.
    log_level : {"error", "warn", "info", "debug"}, optional
        The log level to set for the `cuml.accel` logger. Defaults to `"warn"`,
        set to `"info"` or `"debug"` to get more information about what methods
        `cuml.accel` accelerated for a given run.
    """
    logger.set_level(log_level)

    if not disable_uvm:
        if _is_concurrent_managed_access_supported():
            import rmm

            mr = rmm.mr.get_current_device_resource()
            if isinstance(mr, rmm.mr.ManagedMemoryResource):
                # Nothing to do
                pass
            elif not isinstance(mr, rmm.mr.CudaMemoryResource):
                logger.debug(
                    "A non-default memory resource is already configured, "
                    "skipping enabling managed memory."
                )
            else:
                rmm.mr.set_current_device_resource(
                    rmm.mr.ManagedMemoryResource()
                )
                logger.debug("Enabled managed memory.")
        else:
            logger.debug("Could not enable managed memory on this platform.")

    ACCEL.install()
    set_global_output_type("numpy")

    logger.info("Accelerator installed.")


def pytest_load_initial_conftests(early_config, parser, args):
    # https://docs.pytest.org/en/7.1.x/reference/\
    # reference.html#pytest.hookspec.pytest_load_initial_conftests
    try:
        install()
    except RuntimeError:
        raise RuntimeError(
            "An existing plugin has already loaded sklearn. Interposing failed."
        )
