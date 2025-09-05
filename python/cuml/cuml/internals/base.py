#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import inspect
import os

import pylibraft.common.handle

import cuml
import cuml.common
import cuml.internals
import cuml.internals.input_utils
import cuml.internals.logger as logger
import cuml.internals.nvtx as nvtx
from cuml.internals import api_context_managers
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.input_utils import determine_array_type
from cuml.internals.mem_type import MemoryType
from cuml.internals.memory_utils import determine_array_memtype
from cuml.internals.mixins import TagsMixin
from cuml.internals.output_type import (
    INTERNAL_VALID_OUTPUT_TYPES,
    VALID_OUTPUT_TYPES,
)


class VerbosityDescriptor:
    """Descriptor for ensuring correct type is used for verbosity

    This descriptor ensures that when the 'verbose' attribute of a cuML
    estimator is accessed external to the cuML API, an integer is returned
    (consistent with Scikit-Learn's API for verbosity). Internal to the API, an
    enum is used. Scikit-Learn's numerical values for verbosity are the inverse
    of those used by spdlog, so the numerical value is also inverted internal
    to the cuML API. This ensures that cuML code treats verbosity values as
    expected for an spdlog-based codebase.
    """

    def __get__(self, obj, cls=None):
        if api_context_managers.in_internal_api():
            return logger._verbose_to_level(obj._verbose)
        else:
            return obj._verbose

    def __set__(self, obj, value):
        if api_context_managers.in_internal_api():
            assert isinstance(value, logger.level_enum), (
                "The log level should always be provided as a level_enum, "
                "not an integer"
            )
            obj._verbose = logger._verbose_from_level(value)
        else:
            if isinstance(value, logger.level_enum):
                raise ValueError(
                    "The log level should always be provided as an integer, "
                    "not using the enum"
                )
            obj._verbose = value


class Base(TagsMixin, metaclass=cuml.internals.BaseMetaClass):
    """
    Base class for all the ML algos. It handles some of the common operations
    across all algos. Every ML algo class exposed at cython level must inherit
    from this class.

    Typical estimator design using Base requires three main things:

    1. Call the base __init__ method explicitly from inheriting estimators in
        their __init__.

    2. Attributes that users will want to access, and are array-like should
        use cuml.internals.array, and have a preceding underscore `_` before
        the name the user expects. That way the __getattr__ of Base will
        convert it automatically to the appropriate output format for the
        user. For example, in DBSCAN the user expects to be able to access
        `model.labels_`, so the code actually has an attribute
        `model._labels_` that gets converted at the moment the user accesses
        `labels_` automatically. No need for extra code in inheriting classes
        as long as they follow that naming convention. It is recommended to
        create the attributes in the constructor assigned to None, and
        add a note for users that might look into the code to see what
        attributes the class might have. For example, in KMeans:

    .. code-block:: python

        def __init__(...)
            super(KMeans, self).__init__(handle, verbose, output_type)

            # initialize numeric variables

            # internal array attributes
            self._labels_ = None # accessed via estimator.labels_
            self._cluster_centers_ = None # accessed via estimator.cluster_centers_  # noqa

    3. To appropriately work for outputs mirroring the format of inputs of the
        user when appropriate, the code in the inheriting estimator must call
        the following methods, with input being the data sent by the user:

    - `self._set_output_type(input)` in `fit` methods that modify internal
        structures. This will allow users to receive the correct format when
        accessing internal attributes of the class (eg. labels_ in KMeans).:

    .. code-block:: python

        def fit(self, X):
            self._set_output_type(X)
            # rest of the fit code

    - `out_type = self._get_output_type(input)` in `predict`/`transform` style
        methods, that don't modify class attributes. out_type then can be used
        to return the correct format to the user. For example, in KMeans:

    .. code-block:: python

        def transform(self, X, convert_dtype=False):
            out_type = self._get_output_type(X)
            X_m, n_rows, n_cols, dtype = input_to_cuml_array(X ...)
            preds = CumlArray.zeros(...)

            # method code and call to C++ and whatever else is needed

            return preds.to_output(out_type)

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    output_mem_type : {'host', 'device'}, default=None
        Return results with memory of the indicated type and use the
        indicated memory type for estimator attributes. If None, the memory
        type set at the module level (`cuml.global_settings.memory_type`) will
        be used.

    Examples
    --------

    .. code-block:: python

        from cuml import Base

        # assuming this ML algo has separate 'fit' and 'predict' methods
        class MyAlgo(Base):
            def __init__(self, ...):
                super(MyAlgo, self).__init__(...)
                # other setup logic

            def fit(self, data, ...):
                # check output format
                self._check_output_type(data)
                # train logic goes here

            def predict(self, data, ...):
                # check output format
                self._check_output_type(data)
                # inference logic goes here

            @classmethod
            def _get_param_names(cls):
                # return a list of hyperparam names supported by this algo

        # stream and handle example:

        stream = pylibraft.common.Stream()
        handle = pylibraft.common.Handle(stream=stream)

        algo = MyAlgo(handle=handle)
        algo.fit(...)
        result = algo.predict(...)

        # final sync of all gpu-work launched inside this object
        # this is same as `pylibraft.common.Stream.sync()` call, but safer in case
        # the default stream inside the `raft::handle_t` is being used
        base.handle.sync()
        del base  # optional!
    """

    def __init__(
        self,
        *,
        handle=None,
        verbose=False,
        output_type=None,
        output_mem_type=None,
    ):
        """
        Constructor. All children must call init method of this base class.

        """
        self.handle = (
            pylibraft.common.handle.Handle() if handle is None else handle
        )

        # The following manipulation of the root_cm ensures that the verbose
        # descriptor sees any set or get of the verbose attribute as happening
        # internal to the cuML API. Currently, __init__ calls do not take place
        # within an api context manager, so setting "verbose" here would
        # otherwise appear to be external to the cuML API. This behavior will
        # be corrected with the update of cuML's API context manager
        # infrastructure in https://github.com/rapidsai/cuml/pull/6189.
        GlobalSettings().prev_root_cm = GlobalSettings().root_cm
        GlobalSettings().root_cm = True
        self.verbose = logger._verbose_to_level(verbose)
        # Please see above note on manipulation of the root_cm. This should be
        # rendered unnecessary with https://github.com/rapidsai/cuml/pull/6189.
        GlobalSettings().root_cm = GlobalSettings().prev_root_cm

        self.output_type = _check_output_type_str(
            cuml.global_settings.output_type
            if output_type is None
            else output_type
        )
        if output_mem_type is None:
            self.output_mem_type = cuml.global_settings.memory_type
        else:
            self.output_mem_type = MemoryType.from_str(output_mem_type)
        self._input_type = None
        self._input_mem_type = None
        self.target_dtype = None

        nvtx_benchmark = os.getenv("NVTX_BENCHMARK")
        if nvtx_benchmark and nvtx_benchmark.lower() == "true":
            self.set_nvtx_annotations()

    verbose = VerbosityDescriptor()

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
        params = dict()
        variables = self._get_param_names()
        for key in variables:
            var_value = getattr(self, key, None)
            # We are currently internal to the cuML API, but the value we
            # return will immediately be returned external to the API, so we
            # must perform the translation from enum to integer before
            # returning the value. Ordinarily, this is handled by
            # VerbosityDescriptor for direct access to the verbose
            # attribute.
            if key == "verbose":
                var_value = logger._verbose_from_level(var_value)
            params[key] = var_value
        return params

    def set_params(self, **params):
        """
        Accepts a dict of params and updates the corresponding ones owned by
        this class. If the child class has appropriately overridden the
        `_get_param_names` method and does not need anything other than what is,
        there in this method, then it doesn't have to override this method
        """
        if not params:
            return self
        variables = self._get_param_names()
        for key, value in params.items():
            if key not in variables:
                raise ValueError("Bad param '%s' passed to set_params" % key)
            else:
                # Switch verbose to enum since we are now internal to cuML API
                if key == "verbose":
                    value = logger._verbose_to_level(value)
                setattr(self, key, value)
        return self

    def _set_base_attributes(
        self, output_type=None, target_dtype=None, n_features=None
    ):
        """
        Method to set the base class attributes - output type,
        target dtype and n_features. It combines the three different
        function calls. It's called in fit function from estimators.

        Parameters
        --------
        output_type : DataFrame (default = None)
            Is output_type is passed, aets the output_type on the
            dataframe passed
        target_dtype : Target column (default = None)
            If target_dtype is passed, we call _set_target_dtype
            on it
        n_features: int or DataFrame (default=None)
            If an int is passed, we set it to the number passed
            If dataframe, we set it based on the passed df.

        Examples
        --------

        .. code-block:: python

                # To set output_type and n_features based on X
                self._set_base_attributes(output_type=X, n_features=X)

                # To set output_type on X and n_features to 10
                self._set_base_attributes(output_type=X, n_features=10)

                # To only set target_dtype
                self._set_base_attributes(output_type=X, target_dtype=y)
        """
        if output_type is not None:
            self._set_output_type(output_type)
            self._set_output_mem_type(output_type)
        if target_dtype is not None:
            self._set_target_dtype(target_dtype)
        if n_features is not None:
            self._set_n_features_in(n_features)

    def _set_output_type(self, inp):
        self._input_type = determine_array_type(inp)

    def _set_output_mem_type(self, inp):
        self._input_mem_type = determine_array_memtype(inp)

    def _get_output_type(self, inp=None):
        """
        Method to be called by predict/transform methods of inheriting classes.
        Returns the appropriate output type depending on the type of the input,
        class output type and global output type.
        """

        # Default to the global type
        output_type = cuml.global_settings.output_type

        # If its None, default to our type
        if output_type is None or output_type == "mirror":
            output_type = self.output_type

        # If we are input, get the type from the input (if available)
        if output_type == "input":
            if inp is None:
                # No input value provided, use the estimator input type
                output_type = self._input_type
            else:
                # Determine the output from the input
                output_type = determine_array_type(inp)

        return output_type

    def _get_output_mem_type(self, inp):
        """
        Method to be called by predict/transform methods of inheriting classes.
        Returns the appropriate memory type depending on the type of the input,
        class output type and global output type.
        """

        # Default to the global type
        mem_type = cuml.global_settings.memory_type

        # If we are input, get the type from the input
        if cuml.global_settings.output_type == "input":
            mem_type = determine_array_memtype(inp)

        return mem_type

    def _set_target_dtype(self, target):
        self.target_dtype = cuml.internals.input_utils.determine_array_dtype(
            target
        )

    def _get_target_dtype(self):
        """
        Method to be called by predict/transform methods of
        inheriting classifier classes. Returns the appropriate output
        dtype depending on the dtype of the target.
        """
        try:
            out_dtype = self.target_dtype
        except AttributeError:
            out_dtype = None
        return out_dtype

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


# Internal, non class owned helper functions
def _check_output_type_str(output_str):

    if output_str is None:
        return "input"

    assert output_str != "mirror", (
        "Cannot pass output_type='mirror' in Base.__init__(). Did you forget "
        "to pass `output_type=self.output_type` to a child estimator? "
        "Currently `cuml.global_settings.output_type==`{}`"
    ).format(cuml.global_settings.output_type)

    if isinstance(output_str, str):
        output_type = output_str.lower()
        # Check for valid output types + "input"
        if output_type in INTERNAL_VALID_OUTPUT_TYPES:
            # Return the original version if nothing has changed, otherwise
            # return the lowered. This is to try and keep references the same
            # to support sklearn.base.clone() where possible
            return output_str if output_type == output_str else output_type

    valid_output_types_str = ", ".join([f"'{x}'" for x in VALID_OUTPUT_TYPES])
    raise ValueError(
        f"output_type must be one of {valid_output_types_str}"
        f" Got: {output_str}"
    )
