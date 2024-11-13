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


import cuml
import importlib
import inspect
import os

from cuml.internals.global_settings import GlobalSettings
from cuml.internals.mem_type import MemoryType
from cuml.internals import logger
from cuml.internals.safe_imports import gpu_only_import, cpu_only_import
from typing import Optional


# currently we just use this dictionary for debugging purposes
patched_classes = {}


def intercept(
    original_module: str,
    accelerated_module: str,
    original_class_name: str,
    accelerated_class_name: Optional[str] = None,
):

    if accelerated_class_name is None:
        accelerated_class_name = original_class_name
    # Import the original host module and cuML
    module_a = cpu_only_import(original_module)
    module_b = gpu_only_import(accelerated_module)

    # Store a reference to the original (CPU) class
    if original_class_name in patched_classes:
        original_class_a = patched_classes[original_class_name]
    else:
        original_class_a = getattr(module_a, original_class_name)
        patched_classes[original_class_name] = original_class_a

    # Get the class from cuML so ProxyEstimator inherits from it
    class_b = getattr(module_b, accelerated_class_name)

    # todo (dgd): add environment variable to disable this
    class ProxyEstimatorMeta(cuml.internals.base_helpers.BaseMetaClass):
        def __repr__(cls):
            return repr(original_class_a)

    class ProxyEstimator(class_b, metaclass=ProxyEstimatorMeta):
        def __init__(self, *args, **kwargs):
            self._cpu_model_class = (
                original_class_a  # Store a reference to the original class
            )
            # print("HYPPPPP")
            kwargs, self._gpuaccel = self._hyperparam_translator(**kwargs)
            super().__init__(*args, **kwargs)

            self._cpu_hyperparams = list(
                inspect.signature(
                    self._cpu_model_class.__init__
                ).parameters.keys()
            )

        def __repr__(self):
            return f"wrapped {self._cpu_model_class}"

        def __str__(self):
            return f"ProxyEstimator of {self._cpu_model_class}"

        def __getstate__(self):
            if not hasattr(self, "_cpu_model"):
                self.import_cpu_model()
                self.build_cpu_model()

                self.gpu_to_cpu()

            return self._cpu_model.__dict__.copy()

        def __reduce__(self):
            import pickle
            from .module_accelerator import disable_module_accelerator

            with disable_module_accelerator():
                filename = self.__class__.__name__ + "_sklearn"
                with open(filename, "wb") as f:
                    pickle.dump(self._cpu_model_class, f)

            return (
                reconstruct_proxy,
                (
                    original_module,
                    accelerated_module,
                    original_class_name,
                    self.__getstate__(),
                ),
            )

        def __setstate__(self, state):
            print(f"state: {state}")
            self._cpu_model_class = (
                original_class_a  # Store a reference to the original class
            )
            super().__init__()
            self.import_cpu_model()
            self._cpu_model = self._cpu_model_class()
            self._cpu_model.__dict__.update(state)
            self.cpu_to_gpu()
            self.output_type = "numpy"
            self.output_mem_type = MemoryType.host

    logger.debug(
        f"Created proxy estimator: ({module_b}, {original_class_name}, {ProxyEstimator})"
    )
    setattr(module_b, original_class_name, ProxyEstimator)

    # This is currently needed for pytest only
    if "PYTEST_CURRENT_TEST" in os.environ:
        setattr(module_a, original_class_name, ProxyEstimator)

    return ProxyEstimator


def reconstruct_proxy(original_module, new_module, class_name_a, args, kwargs):
    "Function needed to pickle since ProxyEstimator is"
    # We probably don't need to intercept again here, since we already stored
    # the variables in _wrappers
    cls = intercept(
        original_module=original_module,
        accelerated_module=new_module,
        original_class_name=class_name_a,
    )

    return cls(*args, **kwargs)
