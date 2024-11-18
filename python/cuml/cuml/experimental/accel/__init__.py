#
# Copyright (c) 2024, NVIDIA CORPORATION.
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


from .magics import load_ipython_extension

from cuml.internals import logger
from cuml.internals.global_settings import GlobalSettings

__all__ = ["load_ipython_extension", "install"]


def install():
    """
    Enable cuML Accelerator Mode.
    """
    from .module_accelerator import ModuleAccelerator

    logger.set_level(logger.level_info)
    logger.set_pattern("%v")

    logger.info("cuML: Installing experimental accelerator...")
    loader = ModuleAccelerator.install("sklearn", "cuml", "sklearn")
    loader_umap = ModuleAccelerator.install("umap", "cuml", "umap")
    loader_hdbscan = ModuleAccelerator.install("hdbscan", "cuml", "hdbscan")
    GlobalSettings().accelerator_loaded = all(
        var is not None for var in [loader, loader_umap, loader_hdbscan]
    )

    if GlobalSettings().accelerator_loaded:
        logger.info(
            "cuML: experimental accelerator successfully initialized..."
        )
    else:
        logger.info("cuML: experimental accelerator failed to initialize...")


def pytest_load_initial_conftests(early_config, parser, args):
    # https://docs.pytest.org/en/7.1.x/reference/\
    # reference.html#pytest.hookspec.pytest_load_initial_conftests
    try:
        install()
    except RuntimeError:
        raise RuntimeError(
            "An existing plugin has already loaded sklearn. Interposing failed."
        )
