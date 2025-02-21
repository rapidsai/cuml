#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

from .magics import load_ipython_extension

from cuml.internals import logger
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.memory_utils import set_global_output_type

__all__ = ["load_ipython_extension", "install"]


def _install_for_library(library_name):
    importlib.import_module(f"._wrappers.{library_name}", __name__)


def install():
    """Enable cuML Accelerator Mode."""
    logger.set_level(logger.level_enum.info)
    logger.set_pattern("%v")

    logger.info("cuML: Installing accelerator...")
    libraries_to_accelerate = ["sklearn", "umap", "hdbscan"]
    accelerated_libraries = []
    failed_to_accelerate = []
    for library_name in libraries_to_accelerate:
        logger.debug(f"cuML: Installing accelerator for {library_name}...")
        try:
            _install_for_library(library_name)
        except ModuleNotFoundError as error:  # underlying package not installed (expected)
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

    if GlobalSettings().accelerator_loaded:
        if any(failed_to_accelerate):
            logger.warn(
                f"cuML: Accelerator initialized, some installations failed: {', '.join(failed_to_accelerate)}"
            )
        else:
            logger.info("cuML: Accelerator successfully initialized.")
    else:
        logger.warn("cuML: Accelerator failed to initialize.")

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
