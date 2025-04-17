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
import inspect


def reconstruct_proxy(proxy_module, proxy_name, state):
    module = importlib.import_module(proxy_module)
    cls = getattr(module, proxy_name)
    obj = cls.__new__(cls)
    obj.__setstate__(state)
    return obj


class ProxyMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.import_cpu_model()
        cls._cpu_model_sig = inspect.signature(cls._cpu_model_class)
        cls._proxy_module = cls.__module__
        cls.__module__ = cls._cpu_model_class.__module__
        cls.__qualname__ = cls._cpu_model_class.__qualname__
        cls.__doc__ = cls._cpu_model_class.__doc__
        cls.__init__.__signature__ = cls._cpu_model_sig

    def __init__(self, *args, **kwargs):
        # The cuml signature may not align with the sklearn signature.
        # Additionally, some sklearn models support positional arguments.
        # To work around this, we
        # - Bind arguments to the sklearn signature
        # - Convert the arguments to named parameters
        # - Translate them to cuml equivalents
        # - Then forward them on to the cuml class
        bound = self._cpu_model_sig.bind_partial(*args, **kwargs)
        translated_kwargs, self._gpuaccel = self._hyperparam_translator(
            **bound.arguments
        )
        super().__init__(**translated_kwargs)
        self.build_cpu_model(**kwargs)

    def __repr__(self):
        return self._cpu_model.__repr__()

    def __str__(self):
        return self._cpu_model.__str__()

    def __getattr__(self, attr):
        # Don't dispatch __sklearn_clone__ so that cloning works as a
        # as a regular estimator without __sklearn_clone__
        may_dispatch = attr != "__sklearn_clone__"

        if may_dispatch and hasattr(self._cpu_model_class, attr):
            self.build_cpu_model()
            self.gpu_to_cpu()
            return getattr(self._cpu_model, attr)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {attr!r}"
        )

    def __reduce__(self):
        return (
            reconstruct_proxy,
            (
                self._proxy_module,
                type(self).__name__,
                self.__getstate__(),
            ),
        )

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self._cpu_model.get_params(deep=deep)

    def set_params(self, **params):
        """
        Set parameters for this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self : estimator instance
            The estimnator instance
        """
        self._cpu_model.set_params(**params)
        params, gpuaccel = self._hyperparam_translator(**params)
        params = {
            key: params[key]
            for key in self._get_param_names()
            if key in params
        }
        super().set_params(**params)
        return self
