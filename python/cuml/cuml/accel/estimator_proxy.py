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

import functools
from typing import Any

import sklearn
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin
from sklearn.utils._set_output import _wrap_data_with_container

from cuml.internals import logger
from cuml.internals.interop import UnsupportedOnGPU, is_fitted


def is_proxy(instance_or_class) -> bool:
    """Check if an instance or class is a proxy object created by the accelerator."""
    if isinstance(instance_or_class, type):
        cls = instance_or_class
    else:
        cls = type(instance_or_class)
    return issubclass(cls, ProxyBase)


class _ReconstructProxy:
    """A function for reconstructing serialized estimators.

    Defined this way so that it pickles by value, allowing unpickling in
    environments where cuml isn't installed."""

    @functools.cached_property
    def _reconstruct(self):
        # The reconstruct function is defined in a closure so that cloudpickle
        # will always serialize it by value rather than by reference. This allows
        # the saved model to be loaded in an environment without cuml installed,
        # where it will load as the CPU model directly.
        def reconstruct(cls_path, cpu_model):
            """Reconstruct a serialized estimator.

            Returns a proxy estimator if `cuml.accel` is installed, falling back
            to a CPU model otherwise.
            """
            import importlib

            path, _, name = cls_path.rpartition(".")
            module = importlib.import_module(path)
            try:
                # `cuml` is in the exclude list, so this _reconstruct method is as well.
                # To work around this, we directly access the accelerator override mapping
                # (if available), falling back to CPU if not.
                cls = module._accel_overrides[name]
            except (KeyError, AttributeError):
                # Either:
                # - cuml.accel is not installed
                # - The cuml version doesn't support this estimator type
                # Return the CPU estimator directly
                return cpu_model
            # `cuml.accel` is installed, reconstruct a proxy estimator
            return cls._reconstruct_from_cpu(cpu_model)

        return reconstruct

    def __reduce__(self):
        import pickle

        # Use cloudpickle bundled with joblib. Since joblib is a required dependency
        # of sklearn (and sklearn is a required dep of cuml & all accelerated modules),
        # this should always be installed.
        import joblib.externals.cloudpickle as cloudpickle

        return (pickle.loads, (cloudpickle.dumps(self._reconstruct),))

    def __call__(self, cls_path, cpu):
        return self._reconstruct(cls_path, cpu)


_reconstruct_proxy = _ReconstructProxy()


class ProxyBase(BaseEstimator):
    """A base class for defining new Proxy estimators.

    Subclasses should define ``_gpu_class``, which must be a subclass of
    ``InteropMixin``.

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

        # Store `_cpu_class` from `_gpu_class` for parity and ease-of-reference
        cls._cpu_class = cls._gpu_class._get_cpu_class()

        # Store whether sparse inputs are supported, unless overridden
        if not hasattr(cls, "_gpu_supports_sparse"):
            cls._gpu_supports_sparse = (
                "sparse" in cls._gpu_class._get_tags()["X_types_gpu"]
            )

        # Wrap __init__ to ensure signature compatibility.
        orig_init = cls.__init__

        @functools.wraps(cls._cpu_class.__init__)
        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)

        cls.__init__ = __init__

        # Forward some metadata to make the proxy class look like the CPU class
        cls.__qualname__ = cls._cpu_class.__qualname__
        cls.__doc__ = cls._cpu_class.__doc__
        # We use the `_cpu_class_path` (which matches the patched path) instead of
        # the original full module path so that the class (not an instance) can
        # be pickled properly.
        cls.__module__ = cls._gpu_class._cpu_class_path.rsplit(".", 1)[0]

        # Forward _estimator_type as a class attribute if available
        _estimator_type = getattr(cls._cpu_class, "_estimator_type", None)
        if isinstance(_estimator_type, str):
            cls._estimator_type = _estimator_type

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

            @functools.wraps(cpu_method)
            def method(self, *args, **kwargs):
                return self._call_method(name, *args, **kwargs)

            return method

        for name in methods:
            if not hasattr(cls, name):
                setattr(cls, name, _make_method(name))

    def _sync_params_to_gpu(self) -> None:
        """Sync hyperparameters to GPU estimator from CPU estimator.

        If the hyperparameters are unsupported, will cause proxy
        to fallback to CPU until refit."""
        if self._gpu is not None:
            try:
                params = self._gpu_class._params_from_cpu(self._cpu)
                self._gpu.set_params(**params)
                logger.debug(
                    f"[cuml.accel] Synced parameters from CPU to GPU "
                    f"for '{self._cpu_class.__qualname__}'"
                )
            except UnsupportedOnGPU as e:
                # Log the failure
                estimator_name = self._cpu_class.__qualname__
                exception_details = (
                    str(e)
                    or "specific input/parameter selection not supported on GPU"
                )
                logger.warn(
                    f"[cuml.accel] Parameter synchronization failed for "
                    f"'{estimator_name}': {exception_details} - falling back to CPU"
                )
                self._sync_attrs_to_cpu()
                self._gpu = None

    def _sync_attrs_to_cpu(self) -> None:
        """Sync fit attributes to CPU estimator from GPU estimator.

        This is a no-op if fit attributes are already in sync.
        """
        if self._gpu is not None and not self._synced:
            self._gpu._sync_attrs_to_cpu(self._cpu)
            self._synced = True
            # Log successful attribute sync
            estimator_name = self._cpu_class.__qualname__
            logger.debug(
                f"[cuml.accel] Synced fit attributes from GPU to CPU "
                f"for '{estimator_name}'"
            )

    @classmethod
    def _reconstruct_from_cpu(cls, cpu):
        """Reconstruct a proxy estimator from its CPU counterpart.

        Primarily used when unpickling serialized proxy estimators."""
        assert type(cpu) is cls._cpu_class
        self = cls.__new__(cls)
        self._cpu = cpu
        self._synced = False
        if is_fitted(self._cpu):
            # This is a fit estimator. Try to convert model back to GPU
            try:
                self._gpu = self._gpu_class.from_sklearn(self._cpu)
            except UnsupportedOnGPU:
                self._gpu = None
            else:
                # Supported on GPU, clear fit attributes from CPU to release host memory
                self._cpu = sklearn.clone(self._cpu)
        else:
            # Estimator is unfit, delay GPU init until needed
            self._gpu = None
        logger.debug(
            f"[cuml.accel] Reconstructed GPU estimator from CPU "
            f"for '{self._cpu_class.__qualname__}'"
        )
        return self

    def _call_gpu_method(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the wrapped GPU estimator."""
        from cuml.common.sparse_utils import is_sparse

        # Get estimator name for logging
        estimator_name = self._cpu_class.__qualname__
        method_call = f"{estimator_name}.{method}()"

        if args and is_sparse(args[0]) and not self._gpu_supports_sparse:
            # Sparse inputs not supported
            logger.warn(
                f"[cuml.accel] Unable to accelerate "
                f"'{method_call}' call: Sparse inputs are not supported"
            )
            raise UnsupportedOnGPU("Sparse inputs are not supported")

        # Determine the function to call. Check for an override on the proxy class,
        # falling back to the GPU class method if one exists.
        gpu_func = getattr(self, f"_gpu_{method}", None)
        if gpu_func is None:
            if (gpu_func := getattr(self._gpu, method, None)) is None:
                # Method is not implemented in cuml
                logger.warn(
                    f"[cuml.accel] Unable to accelerate "
                    f"'{method_call}' call: Method is not implemented in cuml"
                )
                raise UnsupportedOnGPU("Method is not implemented in cuml")

        # Log successful GPU acceleration
        logger.info(
            f"[cuml.accel] Successfully accelerated " f"'{method_call}' call"
        )

        out = gpu_func(*args, **kwargs)

        if method in ("transform", "fit_transform"):
            # Ensure transform result is properly wrapped for `set_output`
            out = _wrap_data_with_container("transform", out, args[0], self)

        return self if out is self._gpu else out

    def _call_method(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the proxied estimators."""

        if method.startswith("set_") and method.endswith("_request"):
            # This is a metadata request setter (like `set_{method}_request`),
            # always dispatch directly to CPU.
            getattr(self._cpu, method)(*args, **kwargs)
            return self

        is_fit = method in ("fit", "fit_transform", "fit_predict")

        # Get estimator name for logging
        estimator_name = self._cpu_class.__qualname__

        if is_fit:
            # Attempt to call CPU param validation to validate hyperparameters.
            # This ensures we match errors for invalid hyperparameters during fitting.
            self._validate_params()

            # Attempt to create a new GPU estimator with the current hyperparameters.
            try:
                self._gpu = self._gpu_class(
                    **self._gpu_class._params_from_cpu(self._cpu)
                )
            except UnsupportedOnGPU as e:
                # Unsupported, fallback to CPU
                exception_details = (
                    str(e)
                    or "specific input/parameter selection not supported on GPU"
                )
                logger.warn(
                    f"[cuml.accel] Failed to initialize '{estimator_name}' with GPU acceleration: "
                    f"{exception_details} - falling back to CPU"
                )
                self._gpu = None
            else:
                # Log successful GPU initialization
                logger.debug(
                    f"[cuml.accel] Initialized estimator '{estimator_name}' "
                    f"for GPU acceleration"
                )
                # New estimator successfully initialized on GPU, reset on CPU
                self._cpu = sklearn.clone(self._cpu)
                self._synced = False

        if self._gpu is not None:
            # The hyperparameters are supported, try calling the method
            try:
                return self._call_gpu_method(method, *args, **kwargs)
            except UnsupportedOnGPU as e:
                # GPU call failed, log the reason for failure
                exception_details = str(e) or "unknown error"
                method_call = f"{estimator_name}.{method}()"
                logger.warn(
                    f"[cuml.accel] Failed to accelerate '{method_call}': "
                    f"{exception_details} - falling back to CPU"
                )

                # Unsupported. If it's a `fit` we need to clear
                # the GPU state before falling back to CPU.
                if is_fit:
                    self._gpu = None

        # Failed to run on GPU, fallback to CPU
        method_call = f"{estimator_name}.{method}()"

        # Log general CPU execution (is not necessarily a fallback)
        logger.debug(f"[cuml.accel] Executing '{method_call}' on CPU")

        self._sync_attrs_to_cpu()
        out = getattr(self._cpu, method)(*args, **kwargs)
        return self if out is self._cpu else out

    ############################################################
    # set_output handling                                      #
    ############################################################

    def _gpu_set_output(self, *, transform=None):
        # `set_output` can always call the CPU model (where the output config state
        # is stored). It's defined as a `_gpu_*` method so it only shows up on the
        # proxy for models that define `set_output`, and can avoid unnecessary calls
        # to sync fit attributes to CPU
        self._cpu.set_output(transform=transform)
        return self._gpu

    def _gpu_get_feature_names_out(self, input_features=None):
        # In the common case `get_feature_names_out` doesn't require fitted attributes
        # on the CPU. Here we detect and special case a common mixin, falling back to
        # CPU when necessary. This helps avoid unnecessary device -> host transfers.
        cpu_method = self._cpu_class.get_feature_names_out
        if cpu_method is ClassNamePrefixFeaturesOutMixin.get_feature_names_out:
            # Can run cpu method directly on GPU instance, it only references `_n_features_out`
            return cpu_method(self._gpu, input_features=input_features)

        # Fallback to CPU
        raise UnsupportedOnGPU

    @property
    def _sklearn_output_config(self):
        # Used by sklearn to handle wrapping output type, just proxy through to the CPU model
        return self._cpu._sklearn_output_config

    @property
    def _sklearn_auto_wrap_output_keys(self):
        # Used by sklearn to handle wrapping output type, just proxy through to the CPU model
        return self._cpu._sklearn_auto_wrap_output_keys

    ############################################################
    # Standard magic methods                                   #
    ############################################################

    def __dir__(self) -> list[str]:
        # Ensure attributes are synced so they show up
        self._sync_attrs_to_cpu()
        return dir(self._cpu)

    def __reduce__(self):
        # We only use the CPU estimator for pickling, ensure its fully synced
        self._sync_attrs_to_cpu()
        return (
            _reconstruct_proxy,
            (self._gpu_class._cpu_class_path, self._cpu),
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            # Never proxy magic methods or private attributes
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )

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
            # Mutating non-hyperparameter (probably a fit attribute). This
            # is weird to do, and should never be done during normal sklearn
            # usage. This does happen sometimes in sklearn tests though.
            # The only sane thing to do is to fallback to CPU and forward
            # the mutation through.
            self._sync_attrs_to_cpu()
            self._gpu = None
            setattr(self._cpu, name, value)

    def __delattr__(self, name: str) -> None:
        if name in ("_cpu", "_gpu", "_synced"):
            # Internal state. We never call this, just here for parity.
            object.__delattr__(self, name)
        else:
            # No normal workflow deletes attributes (hyperparameters or otherwise)
            # from a sklearn estimator. The sklearn tests do sometimes though.
            # The only sane thing to do is to fallback to CPU and forward
            # the mutation through.
            self._sync_attrs_to_cpu()
            self._gpu = None
            delattr(self._cpu, name)

    ############################################################
    # Public sklearn methods                                   #
    ############################################################

    @functools.wraps(BaseEstimator.get_params)
    def get_params(self, deep=True):
        return self._cpu.get_params(deep=deep)

    @functools.wraps(BaseEstimator.set_params)
    def set_params(self, **kwargs):
        self._cpu.set_params(**kwargs)
        self._sync_params_to_gpu()
        return self

    def __sklearn_tags__(self):
        return self._cpu.__sklearn_tags__()

    def __sklearn_is_fitted__(self):
        model = self._cpu if self._gpu is None else self._gpu
        return is_fitted(model)

    def __sklearn_clone__(self):
        cls = type(self)
        out = cls.__new__(cls)
        # Clone only copies hyperparameters.
        out._cpu = sklearn.clone(self._cpu)
        out._gpu = None
        out._synced = False
        return out

    ############################################################
    # Metadata Routing Methods                                 #
    ############################################################

    @functools.wraps(BaseEstimator.get_metadata_routing)
    def get_metadata_routing(self):
        return self._cpu.get_metadata_routing()

    @functools.wraps(BaseEstimator._get_metadata_request)
    def _get_metadata_request(self):
        return self._cpu._get_metadata_request()

    @classmethod
    @functools.wraps(BaseEstimator._get_default_requests)
    def _get_default_requests(cls):
        return cls._cpu_class._get_default_requests()

    @property
    def _metadata_request(self):
        return self._cpu._metadata_request

    ############################################################
    # Methods on BaseEstimator used internally by sklearn      #
    ############################################################

    @property
    def _estimator_type(self):
        return self._cpu._estimator_type

    @property
    def _parameter_constraints(self):
        return self._cpu._parameter_constraints

    @classmethod
    def _get_param_names(cls):
        return cls._cpu_class._get_param_names()

    def _validate_params(self):
        if hasattr(self._cpu, "_validate_params") and hasattr(
            self._cpu, "_parameter_constraints"
        ):
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
