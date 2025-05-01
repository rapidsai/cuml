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
from __future__ import annotations

import importlib
import inspect
from functools import wraps
from typing import Any

import sklearn
from sklearn.base import BaseEstimator

from cuml.internals.interop import UnsupportedOnGPU


def is_proxy(instance_or_class) -> bool:
    """Check if an instance or class is a proxy object created by the accelerator."""

    if isinstance(instance_or_class, type):
        cls = instance_or_class
    else:
        cls = type(instance_or_class)
    return issubclass(cls, (ProxyMixin, ProxyBase))


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
        # The cuml signature may not align with the CPU signature.
        # Additionally, some models support positional arguments.
        # To work around this, we
        # - Bind arguments to the CPU signature
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


class ProxyBase(BaseEstimator):
    """A base class for defining new Proxy estimators.

    Subclasses should define ``_gpu_class`` and ``_cpu_class``.

    Attributes and hyperparameters are always proxied through the CPU estimator.

    Methods _may_ be run on the GPU estimator if:

    - The GPU estimator supports all the hyperparameters
    - The GPU estimator has a method with the same name

    Additionally, GPU-specific behavior for a method may be overridden by defining a
    method on this class with the name ``_gpu_{method}`` (and the same signature).
    If defined, this will be called instead of ``self._gpu.{method}`` when dispatching
    to the GPU with the rules above. If this method raises a ``UnsupportedOnGPU`` error
    then the proxy will fallback to CPU.

    See the definitions in ``cuml.accel._wrappers.linear_model`` for examples.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The CPU estimator, always defined. This is the source of truth for any
        # hyperparameters. It also may have fit attributes on it - these are lazily
        # synced from the GPU estimator when accessed.
        self._cpu = self._cpu_class(*args, **kwargs)
        # The GPU estimator. This is only non-None if an estimator was successfully fit
        # on the GPU.
        self._gpu = None
        # Whether fit attributes on the CPU estimator are currently in sync with those
        # on the GPU estimator.
        self._synced = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Add proxy method definitions for all public methods on CPU class
        # that aren't already defined on the proxy class
        methods = [
            name
            for name in dir(cls._cpu_class)
            if not name.startswith("_")
            and callable(getattr(cls._cpu_class, name))
        ]

        def _make_method(name):
            cpu_method = getattr(cls._cpu_class, name)

            @wraps(cpu_method)
            def method(self, *args, **kwargs):
                return self._call_method(name, *args, **kwargs)

            return method

        for name in methods:
            if not hasattr(cls, name):
                setattr(cls, name, _make_method(name))

    def _sync_params_to_gpu(self) -> None:
        """Sync hyperparameters to GPU estimator.

        If the hyperparameters are unsupported, will cause proxy
        to fallback to CPU until refit."""
        if self._gpu is not None:
            try:
                params = self._gpu_class._params_from_cpu(self._cpu)
                self._gpu.set_params(**params)
            except UnsupportedOnGPU:
                self._sync_attrs_to_cpu()
                self._gpu = None

    def _sync_attrs_to_cpu(self) -> None:
        """Sync attributes to CPU estimator."""
        if not self._synced:
            if self._gpu is not None:
                self._gpu._sync_attrs_to_cpu(self._cpu)
            self._synced = True

    def _call_method(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the proxied estimators."""

        is_fit = method in ("fit", "fit_transform", "fit_predict")

        if is_fit:
            # Fitting a new estimator.
            # First clear all fit attributes on the CPU estimator.
            self._cpu = sklearn.clone(self._cpu)
            self._synced = False

            # Then attempt to create a new GPU estimator with the
            # current hyperparameters.
            try:
                self._gpu = self._gpu_class.from_sklearn(self._cpu)
            except UnsupportedOnGPU:
                # Unsupported, fallback to CPU
                self._gpu = None

        if self._gpu is not None:
            # The hyperparameters are supported, now check for a GPU method to run.
            gpu_func = getattr(self, f"_gpu_{method}", None)
            if gpu_func is None:
                gpu_func = getattr(self._gpu, method, None)

            if gpu_func is not None:
                try:
                    out = gpu_func(*args, **kwargs)
                except UnsupportedOnGPU:
                    # Unsupported. If it's a `fit` we need to clear
                    # the GPU state before falling back to CPU.
                    if is_fit:
                        self._gpu = None
                else:
                    # Ran successfully on GPU. Prep the results and return.
                    return self if out is self._gpu else out

        # Failed to run on GPU, fallback to CPU
        self._sync_attrs_to_cpu()
        out = getattr(self._cpu, method)(*args, **kwargs)
        return self if out is self._cpu else out

    ############################################################
    # Standard magic methods                                   #
    ############################################################

    def __str__(self) -> str:
        return self._cpu.__str__()

    def __repr__(self) -> str:
        return self._cpu.__repr__()

    def __dir__(self) -> list[str]:
        # Ensure attributes are synced so they show up
        self._sync_attrs_to_cpu()
        return dir(self._cpu)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            # Never proxy magic methods or private attributes
            raise AttributeError(name)

        if name.endswith("_"):
            # Fit attributes require syncing
            self._sync_attrs_to_cpu()
        return getattr(self._cpu, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_cpu", "_gpu", "_synced"):
            # Internal state
            object.__setattr__(self, name, value)
        elif name in self._cpu._get_param_names():
            # Hyperparameter, set on CPU
            setattr(self._cpu, name, value)
            self._sync_params_to_gpu()
        else:
            raise ValueError(
                f"Cannot set attribute {name!r} on {type(self).__name__!r} "
                "when running with `cuml.accel` enabled."
            )

    ############################################################
    # Public sklearn methods                                   #
    ############################################################

    @wraps(BaseEstimator.get_params)
    def get_params(self, deep=True):
        return self._cpu.get_params(deep=deep)

    @wraps(BaseEstimator.set_params)
    def set_params(self, **kwargs):
        self._cpu.set_params(**kwargs)
        self._sync_params_to_gpu()

    def __sklearn_tags__(self):
        return self._cpu.__sklearn_tags__()

    def __sklearn_clone__(self):
        cls = type(self)
        out = cls.__new__()
        # Clone only copies hyperparameters.
        out._cpu = sklearn.clone(self._cpu)
        out._gpu = None
        out._synced = False
        return out

    ############################################################
    # Methods on BaseEstimator used internally by sklearn      #
    ############################################################

    @property
    def _estimator_type(self):
        return self._cpu._estimator_type

    @classmethod
    def _get_param_names(cls):
        return cls._cpu._get_param_names()

    def _validate_params(self):
        self._cpu._validate_params()

    def _get_tags(self):
        return self._cpu._get_tags()

    def _more_tags(self):
        return self._cpu._more_tags()

    def _repr_mimebundle_(self, **kwargs):
        return self._cpu._repr_mimebundle_(**kwargs)

    @property
    def _repr_html_(self):
        return self._cpu._repr_html_
