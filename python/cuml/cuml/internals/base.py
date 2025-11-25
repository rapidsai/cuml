#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
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
from cuml.internals.input_utils import determine_array_type
from cuml.internals.mixins import TagsMixin
from cuml.internals.outputs import check_output_type


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
    ):
        self.handle = (
            pylibraft.common.handle.Handle() if handle is None else handle
        )
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
