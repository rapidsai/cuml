#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import inspect
import os
import threading
import warnings

import pylibraft.common.handle

import cuml
import cuml.common
import cuml.internals
import cuml.internals.input_utils
import cuml.internals.logger as logger
import cuml.internals.nvtx as nvtx
from cuml.internals.input_utils import determine_array_type
from cuml.internals.mixins import TagsMixin
from cuml.internals.outputs import check_output_type

_THREAD_STATE = threading.local()


class DeprecatedHandleDescriptor:
    """A descriptor to ease deprecating the `handle` parameter."""

    def __set__(self, obj, value):
        # Only warn if set to non-None on *non-multiGPU classes*
        if value is not None and not type(obj).__name__.endswith("MG"):
            params = obj._get_param_names() if isinstance(obj, Base) else []
            if "n_streams" in params:
                suffix = (
                    " To configure the number of streams used, please use the "
                    "`n_streams` parameter instead."
                )
            elif "device_ids" in params:
                suffix = (
                    " To configure multi-device execution, please use the "
                    "`device_ids` parameter instead."
                )
            else:
                suffix = ""
            warnings.warn(
                f"The `handle` argument to `{type(obj).__name__}` was deprecated "
                f"in 26.02 and will be removed in 26.04. There is no need to "
                f"manually specify a `handle`, cuml now manages this resource "
                f"for you automatically.{suffix}",
                FutureWarning,
            )
        obj.__dict__["handle"] = value


def get_handle(*, handle=None, model=None, n_streams=0, device_ids=None):
    """Get a `pylibraft.common.Handle`.

    Parameters
    ----------
    handle : pylibraft.common.Handle or None, optional
        A `handle` argument to a function. Will raise a nice deprecation
        warning and return if provided. Will be removed once the deprecation of
        `handle` arguments is complete.
    model : cuml.Base or None, optional
        A model to extract the handle from (if one is configured). Will be removed
        once the deprecation of `handle` arguments is complete.
    n_streams : int, default=0
        The number of streams to use for a backing stream pool. If non-zero
        a temporary `Handle` with a pool that size will be created. Otherwise
        the default threadlocal `Handle` will be used.
    device_ids : list[int], "all", or None, default=None
        If non-None, will return a `pylibraft.common.DeviceResourcesSNMG`,
        enabling multi-device execution. May be a list of device IDs, or
        ``"all"`` to use all available devices.
    """
    if handle is not None:
        warnings.warn(
            (
                "The `handle` argument was deprecated in 26.02 and will be "
                "removed in 26.04. There is no need to manually specify a "
                "`handle`, cuml manages this resource for you automatically."
            ),
            FutureWarning,
        )
        return handle

    if model is not None and model.handle is not None:
        # Deprecation of model.handle is handled separately by the descriptor
        return model.handle

    if n_streams == 0 and device_ids is None:
        if not hasattr(_THREAD_STATE, "handle"):
            _THREAD_STATE.handle = pylibraft.common.handle.Handle()
        return _THREAD_STATE.handle
    elif device_ids is not None:
        if n_streams != 0:
            raise ValueError(
                "Cannot specify both `device_ids` and `n_streams`"
            )
        return pylibraft.common.handle.DeviceResourcesSNMG(
            device_ids=(None if device_ids == "all" else device_ids)
        )
    else:
        return pylibraft.common.handle.Handle(n_streams=n_streams)


class Base(TagsMixin):
    """Base class for cuml estimators.

    Subclasses should:

    - Define ``_get_param_names`` to extend the base implementation with
      any additional parameter names.

    - Decorate their ``fit`` method with ``cuml.internals.reflect(reset=True)``
      to store their fitted input type.

    - Decorate methods that return array likes with ``cuml.internals.reflect``
      to properly coerce outputs to the proper type.

    Parameters
    ----------
    handle : cuml.Handle or None, default=None

        .. deprecated:: 26.02
            The `handle` argument was deprecated in 26.02 and will be removed
            in 26.04. There's no need to pass in a handle, cuml now manages
            this resource automatically.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Examples
    --------

    .. code-block:: python

        import cupy as cp
        from cuml.internals import Base, reflect

        class MyAlgo(Base):
            def __init__(
                self,
                *,
                param=123,
                handle=None,
                verbose=False,
                output_type=None,
            ):
                super().__init__(handle=handle, verbose=verbose, output_type=output_type)
                self.param = param

            @classmethod
            def _get_param_names(cls):
                return [*super()._get_param_names(), "param"]

            @reflect(reset=True)
            def fit(self, X, y):
                # Training logic goes here...
                return self

            @reflect
            def predict(self, X):
                # Inference logic goes here...
                return cp.ones(len(X), dtype="int32")
    """

    handle = DeprecatedHandleDescriptor()

    def __init__(
        self,
        *,
        handle=None,
        verbose=False,
        output_type=None,
    ):
        self.handle = handle
        self.verbose = verbose
        if output_type is None:
            output_type = cuml.global_settings.output_type or "input"
            if output_type == "mirror":
                raise ValueError(
                    "Cannot pass output_type='mirror' to Base.__init__(). Did you forget "
                    "to pass `output_type=self.output_type` to a child estimator? "
                )
        else:
            output_type = check_output_type(output_type)
        self.output_type = output_type
        self._input_type = None

        nvtx_benchmark = os.getenv("NVTX_BENCHMARK")
        if nvtx_benchmark and nvtx_benchmark.lower() == "true":
            self.set_nvtx_annotations()

    def __repr__(self):
        """
        Pretty prints the arguments of a class using Scikit-learn standard :)
        """
        signature = inspect.getfullargspec(self.__init__).args
        if len(signature) > 0 and signature[0] == "self":
            del signature[0]
        state = self.__dict__
        string = self.__class__.__name__ + "("
        for key in signature:
            if key not in state:
                continue
            if type(state[key]) is str:
                string += "{}='{}', ".format(key, state[key])
            else:
                if hasattr(state[key], "__str__"):
                    string += "{}={}, ".format(key, state[key])
        string = string.rstrip(", ")
        output = string + ")"

        if hasattr(self, "sk_model_"):
            output += " <sk_model_ attribute used>"
        return output

    @property
    def _verbose_level(self):
        """The current `verbose` setting as a `logger.level_enum`"""
        return logger._verbose_to_level(self.verbose)

    @classmethod
    def _get_param_names(cls):
        """
        Returns a list of hyperparameter names owned by this class. It is
        expected that every child class overrides this method and appends its
        extra set of parameters that it in-turn owns. This is to simplify the
        implementation of `get_params` and `set_params` methods.
        """
        return ["handle", "verbose", "output_type"]

    def get_params(self, deep=True):
        """
        Returns a dict of all params owned by this class. If the child class
        has appropriately overridden the `_get_param_names` method and does not
        need anything other than what is there in this method, then it doesn't
        have to override this method
        """
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params):
        """
        Accepts a dict of params and updates the corresponding ones owned by
        this class. If the child class has appropriately overridden the
        `_get_param_names` method and does not need anything other than what is,
        there in this method, then it doesn't have to override this method
        """
        if not params:
            return self
        valid_params = self._get_param_names()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for `{type(self).__name__}`"
                )
            setattr(self, key, value)
        return self

    def _set_output_type(self, inp):
        self._input_type = determine_array_type(inp)

    def _get_output_type(self, inp=None):
        """
        Method to be called by predict/transform methods of inheriting classes.
        Returns the appropriate output type depending on the type of the input,
        class output type and global output type.
        """
        # Default to the global type
        output_type = cuml.global_settings.output_type

        # If not set to an explicit value, use the estimator's setting
        if output_type in (None, "input", "mirror"):
            output_type = self.output_type

        # If input, get the type from the input (if available)
        if output_type == "input":
            if inp is None:
                # No input value provided, use the estimator input type
                output_type = self._input_type
            else:
                # Determine the output from the input
                output_type = determine_array_type(inp)

        return output_type

    def _set_n_features_in(self, X):
        if isinstance(X, int):
            self.n_features_in_ = X
        else:
            shape = X.shape
            # dataframes can have only one dimension
            if len(shape) == 1:
                self.n_features_in_ = 1
            else:
                self.n_features_in_ = shape[1]

    def _more_tags(self):
        # 'preserves_dtype' tag's Scikit definition currently only applies to
        # transformers and whether the transform method conserves the dtype
        # (in that case returns an empty list, otherwise the dtype it
        # casts to).
        # By default, our transform methods convert to self.dtype, but
        # we need to check whether the tag has been defined already.
        if hasattr(self, "transform") and hasattr(self, "dtype"):
            return {"preserves_dtype": [self.dtype]}
        return {}

    def _repr_mimebundle_(self, **kwargs):
        """Prepare representations used by jupyter kernels to display estimator"""
        from sklearn.utils import estimator_html_repr

        output = {"text/plain": repr(self)}
        output["text/html"] = estimator_html_repr(self)
        return output

    def set_nvtx_annotations(self):
        for func_name in [
            "fit",
            "transform",
            "predict",
            "fit_transform",
            "fit_predict",
        ]:
            if hasattr(self, func_name):
                msg = "{class_name}.{func_name} [{addr}]"
                msg = msg.format(
                    class_name=self.__class__.__module__,
                    func_name=func_name,
                    addr=hex(id(self)),
                )
                msg = msg[5:]  # remove cuml.
                func = getattr(self, func_name)
                func = nvtx.annotate(message=msg, domain="cuml_python")(func)
                setattr(self, func_name, func)
