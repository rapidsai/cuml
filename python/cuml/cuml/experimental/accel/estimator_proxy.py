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

from cuml.internals.global_settings import GlobalSettings
from cuml.internals.mem_type import MemoryType
from cuml.internals import logger
from typing import Optional


patched_classes = {}


class EstimatorInterceptor:
    def __init__(
        self, original_module, new_module, class_name_a, class_name_b
    ):
        self.original_module = original_module
        self.new_module = new_module
        self.class_name_a = class_name_a
        self.class_name_b = class_name_b

    def load_and_replace(self):
        # Import the original host module and cuML
        module_a = importlib.import_module(self.original_module)
        module_b = importlib.import_module(self.new_module)

        # Store a reference to the original (CPU) class
        if self.class_name_a in patched_classes:
            original_class_a = patched_classes[self.class_name_a]
        else:
            original_class_a = getattr(module_a, self.class_name_a)
            patched_classes[self.class_name_a] = original_class_a

        # Get the class from cuML so ProxyEstimator inherits from it
        class_b = getattr(module_b, self.class_name_b)

        # todo: add environment variable to disable this
        class ProxyEstimatorMeta(cuml.internals.base_helpers.BaseMetaClass):
            def __repr__(cls):
                return repr(original_class_a)

        class ProxyEstimator(class_b, metaclass=ProxyEstimatorMeta):
            def __init__(self, *args, **kwargs):
                self._cpu_model_class = (
                    original_class_a  # Store a reference to the original class
                )
                _, self._gpuaccel = self._hyperparam_translator(**kwargs)
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
                    with open("filename2.pkl", "wb") as f:
                        pickle.dump(self._cpu_model_class, f)
                return (
                    reconstruct_proxy,
                    (self._cpu_model_class, self.__getstate__()),
                )
                # Version to only pickle sklearn
                # return (self._cpu_model_class, (), self.__getstate__())

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
            f"Created proxy estimator: ({module_b}, {self.class_name_b}, {ProxyEstimator})"
        )
        setattr(module_b, self.class_name_b, ProxyEstimator)


def reconstruct_proxy(orig_class, state):
    "Function needed to pickle since ProxyEstimator is"
    return ProxyEstimator.__setstate__(state)  # noqa: F821


def intercept(
    original_module: str,
    accelerated_module: str,
    original_class_name: str,
    accelerated_class_name: Optional[str] = None,
) -> None:

    if accelerated_class_name is None:
        accelerated_class_name = original_class_name

    interceptor = EstimatorInterceptor(
        original_module,
        accelerated_module,
        original_class_name,
        accelerated_class_name,
    )
    interceptor.load_and_replace()
