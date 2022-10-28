#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

# distutils: language = c++

import os
import inspect
import typing
from importlib import import_module
import numpy as np
import cupy as cp
import nvtx

import cuml
import cuml.common
import cuml.common.cuda
import cuml.common.logger as logger
import cuml.internals
import pylibraft.common.handle
import cuml.common.input_utils
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.input_utils import input_to_host_array
from cuml.common.array import CumlArray

from cuml.common.doc_utils import generate_docstring
from cuml.common.mixins import TagsMixin
from cuml.common.device_selection import DeviceType, using_device_type
from cuml.common import logger


class Base(TagsMixin,
           metaclass=cuml.internals.BaseMetaClass):
    """
    Base class for all the ML algos. It handles some of the common operations
    across all algos. Every ML algo class exposed at cython level must inherit
    from this class.

    Typical estimator design using Base requires three main things:

    1. Call the base __init__ method explicitly from inheriting estimators in
        their __init__.

    2. Attributes that users will want to access, and are array-like should
        use cuml.common.Array, and have a preceding underscore `_` before
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
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.

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

            def get_param_names(self):
                # return a list of hyperparam names supported by this algo

        # stream and handle example:

        stream = cuml.cuda.Stream()
        handle = pylibraft.common.Handle(stream=stream)

        algo = MyAlgo(handle=handle)
        algo.fit(...)
        result = algo.predict(...)

        # final sync of all gpu-work launched inside this object
        # this is same as `cuml.cuda.Stream.sync()` call, but safer in case
        # the default stream inside the `raft::handle_t` is being used
        base.handle.sync()
        del base  # optional!
    """

    def __init__(self, *,
                 handle=None,
                 verbose=False,
                 output_type=None):
        """
        Constructor. All children must call init method of this base class.

        """
        self.handle = pylibraft.common.handle.Handle() if handle is None \
            else handle

        # Internally, self.verbose follows the spdlog/c++ standard of
        # 0 is most logging, and logging decreases from there.
        # So if the user passes an int value for logging, we convert it.
        if verbose is True:
            self.verbose = logger.level_debug
        elif verbose is False:
            self.verbose = logger.level_info
        else:
            self.verbose = verbose

        self.output_type = _check_output_type_str(
            cuml.global_settings.output_type
            if output_type is None else output_type)
        self._input_type = None
        self.target_dtype = None
        self.n_features_in_ = None

        nvtx_benchmark = os.getenv('NVTX_BENCHMARK')
        if nvtx_benchmark and nvtx_benchmark.lower() == 'true':
            self.set_nvtx_annotations()

    def __repr__(self):
        """
        Pretty prints the arguments of a class using Scikit-learn standard :)
        """
        cdef list signature = inspect.getfullargspec(self.__init__).args
        if len(signature) > 0 and signature[0] == 'self':
            del signature[0]
        cdef dict state = self.__dict__
        cdef str string = self.__class__.__name__ + '('
        cdef str key
        for key in signature:
            if key not in state:
                continue
            if type(state[key]) is str:
                string += "{}='{}', ".format(key, state[key])
            else:
                if hasattr(state[key], "__str__"):
                    string += "{}={}, ".format(key, state[key])
        string = string.rstrip(', ')
        output = string + ')'

        if hasattr(self, 'cpu_model_'):
            output += ' <cpu_model_ attribute used>'
        return output

    def dispatch_func(self, func_name, *args, **kwargs):
        """
        This function will dispatch calls to training and inference according
        to the global configuration. It should work for all estimators
        sufficiently close the scikit-learn implementation as it uses
        it for training and inferences on host.

        Parameters
        ----------
        func_name : string
            name of the function to be dispatched
        args : arguments
            arguments to be passed to the function for the call
        kwargs : keyword arguments
            keyword arguments to be passed to the function for the call
        """
        # look for current device_type
        device_type = cuml.global_settings.device_type
        if device_type == DeviceType.device:
            # call the original cuml method
            cuml_func_name = '_' + func_name
            if hasattr(self, cuml_func_name):
                cuml_func = getattr(self, cuml_func_name)
                return cuml_func(*args, **kwargs)
            else:
                raise ValueError('Function "{}" could not be found in'
                                 ' the cuML estimator'.format(cuml_func_name))

        elif device_type == DeviceType.host:
            # check if the sklean model already set as attribute of the cuml
            # estimator its presence should signify that CPU execution was
            # used previously
            if not hasattr(self, 'cpu_model_'):
                filtered_kwargs = {}
                for keyword, arg in self.full_kwargs.items():
                    if keyword in self.cpu_hyperparams:
                        filtered_kwargs[keyword] = arg
                    else:
                        logger.info("Unused keyword parameter: {} "
                                    "during CPU estimator "
                                    "initialization".format(keyword))

                # initialize model
                self.cpu_model_ = self.cpu_model_class(**filtered_kwargs)

                # transfer attributes trained with cuml
                for attr in self.get_attr_names():
                    # check presence of attribute
                    if hasattr(self, attr) or \
                       isinstance(getattr(type(self), attr, None), property):
                        # get the cuml attribute
                        if hasattr(self, attr):
                            cu_attr = getattr(self, attr)
                        else:
                            cu_attr = getattr(type(self), attr).fget(self)
                        # if the cuml attribute is a CumlArrayDescriptorMeta
                        if hasattr(cu_attr, 'get_input_value'):
                            # extract the actual value from the
                            # CumlArrayDescriptorMeta
                            cu_attr_value = cu_attr.get_input_value()
                            # check if descriptor is empty
                            if cu_attr_value is not None:
                                if cu_attr.input_type == 'cuml':
                                    # transform cumlArray to numpy and set it
                                    # as an attribute in the CPU estimator
                                    setattr(self.cpu_model_, attr,
                                            cu_attr_value.to_output('numpy'))
                                else:
                                    # transfer all other types of attributes
                                    # directly
                                    setattr(self.cpu_model_, attr,
                                            cu_attr_value)
                        elif isinstance(cu_attr, CumlArray):
                            # transform cumlArray to numpy and set it
                            # as an attribute in the CPU estimator
                            setattr(self.cpu_model_, attr,
                                    cu_attr.to_output('numpy'))
                        elif isinstance(cu_attr, cp.ndarray):
                            # transform cupy to numpy and set it
                            # as an attribute in the CPU estimator
                            setattr(self.cpu_model_, attr,
                                    cp.asnumpy(cu_attr))
                        else:
                            # transfer all other types of attributes directly
                            setattr(self.cpu_model_, attr, cu_attr)

            # converts all the args
            args = tuple(input_to_host_array(arg)[0] for arg in args)
            # converts all the kwarg
            for key, kwarg in kwargs.items():
                kwargs[key] = input_to_host_array(kwarg)[0]

            # call the method from the sklearn model
            cpu_func = getattr(self.cpu_model_, func_name)
            res = cpu_func(*args, **kwargs)

            if func_name == 'fit':
                # need to do this to mirror input type
                self._set_output_type(args[0])
                # always return the cuml estimator while training
                # mirror sk attributes to cuml after training
                for attr in self.get_attr_names():
                    # check presence of attribute
                    if hasattr(self.cpu_model_, attr) or \
                       isinstance(getattr(type(self.cpu_model_),
                                          attr, None), property):
                        # get the cpu attribute
                        if hasattr(self.cpu_model_, attr):
                            cpu_attr = getattr(self.cpu_model_, attr)
                        else:
                            cpu_attr = getattr(type(self.cpu_model_),
                                               attr).fget(self.cpu_model_)
                        # if the cpu attribute is an array
                        if isinstance(cpu_attr, np.ndarray):
                            # get data order wished for by CumlArrayDescriptor
                            if hasattr(self, attr + '_order'):
                                order = getattr(self, attr + '_order')
                            else:
                                order = 'K'
                            # transfer array to gpu and set it as a cuml
                            # attribute
                            cuml_array = input_to_cuml_array(cpu_attr,
                                                             order=order)[0]
                            setattr(self, attr, cuml_array)
                        else:
                            # transfer all other types of attributes directly
                            setattr(self, attr, cpu_attr)
                return self
            else:
                # return method result
                return res

    def fit(self, *args, **kwargs):
        return self.dispatch_func('fit', *args, **kwargs)

    def predict(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('predict', *args, **kwargs)

    def transform(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('transform', *args, **kwargs)

    def kneighbors(self, *args, **kwargs) \
            -> typing.Union[CumlArray, typing.Tuple[CumlArray, CumlArray]]:
        return self.dispatch_func('kneighbors', *args, **kwargs)

    def fit_transform(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('fit_transform', *args, **kwargs)

    def fit_predict(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('fit_predict', *args, **kwargs)

    def score(self, *args, **kwargs):
        return self.dispatch_func('score', *args, **kwargs)

    def decision_function(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('decision_function', *args, **kwargs)

    def predict_proba(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('predict_proba', *args, **kwargs)

    def predict_log_proba(self, *args, **kwargs) -> CumlArray:
        return self.dispatch_func('predict_log_proba', *args, **kwargs)

    def get_param_names(self):
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
        has appropriately overridden the `get_param_names` method and does not
        need anything other than what is there in this method, then it doesn't
        have to override this method
        """
        params = dict()
        variables = self.get_param_names()
        for key in variables:
            var_value = getattr(self, key, None)
            params[key] = var_value
        return params

    def set_params(self, **params):
        """
        Accepts a dict of params and updates the corresponding ones owned by
        this class. If the child class has appropriately overridden the
        `get_param_names` method and does not need anything other than what is,
        there in this method, then it doesn't have to override this method
        """
        if not params:
            return self
        variables = self.get_param_names()
        for key, value in params.items():
            if key not in variables:
                raise ValueError("Bad param '%s' passed to set_params" % key)
            else:
                setattr(self, key, value)
        return self

    def __getstate__(self):
        # getstate and setstate are needed to tell pickle to treat this
        # as regular python classes instead of triggering __getattr__
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getattr__(self, attr):
        # Attempts to redirect calls to CPU estimator
        if hasattr(self.cpu_model_class, attr):
            if callable(getattr(self.cpu_model_class, attr)):
                return self._cpu_redirection(attr)

        # Redirects to `solver_model` if the attribute exists.
        if attr == "solver_model":
            return self.__dict__['solver_model']
        if "solver_model" in self.__dict__.keys():
            return getattr(self.solver_model, attr)

        raise AttributeError(attr)

    def _cpu_redirection(self, method_name):
        def redirection(*args, **kwargs):
            with using_device_type('cpu'):
                res = self.dispatch_func(method_name, *args, **kwargs)
                if isinstance(res, self.cpu_model_class):
                    return self
                else:
                    return res
        return redirection

    def _set_base_attributes(self,
                             output_type=None,
                             target_dtype=None,
                             n_features=None):
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
        if target_dtype is not None:
            self._set_target_dtype(target_dtype)
        if n_features is not None:
            self._set_n_features_in(n_features)

    def _set_output_type(self, inp):
        self._input_type = cuml.common.input_utils.determine_array_type(inp)

    def _get_output_type(self, inp):
        """
        Method to be called by predict/transform methods of inheriting classes.
        Returns the appropriate output type depending on the type of the input,
        class output type and global output type.
        """

        # Default to the global type
        output_type = cuml.global_settings.output_type

        # If its None, default to our type
        if (output_type is None or output_type == "mirror"):
            output_type = self.output_type

        # If we are input, get the type from the input
        if output_type == 'input':
            output_type = cuml.common.input_utils.determine_array_type(inp)

        return output_type

    def _set_target_dtype(self, target):
        self.target_dtype = cuml.common.input_utils.determine_array_dtype(
            target)

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
            self.n_features_in_ = X.shape[1]

    def _more_tags(self):
        # 'preserves_dtype' tag's Scikit definition currently only appies to
        # transformers and whether the transform method conserves the dtype
        # (in that case returns an empty list, otherwise the dtype it
        # casts to).
        # By default, our transform methods convert to self.dtype, but
        # we need to check whether the tag has been defined already.
        if hasattr(self, 'transform') and hasattr(self, 'dtype'):
            return {'preserves_dtype': [self.dtype]}
        return {}

    def set_nvtx_annotations(self):
        for func_name in ['fit', 'transform', 'predict', 'fit_transform',
                          'fit_predict']:
            if hasattr(self, func_name):
                message = self.__class__.__module__ + '.' + func_name
                msg = '{class_name}.{func_name} [{addr}]'
                msg = msg.format(class_name=self.__class__.__module__,
                                 func_name=func_name,
                                 addr=hex(id(self)))
                msg = msg[5:]  # remove cuml.
                func = getattr(self, func_name)
                func = nvtx.annotate(message=msg, domain="cuml_python")(func)
                setattr(self, func_name, func)


# Internal, non class owned helper functions
def _check_output_type_str(output_str):

    if (output_str is None):
        return "input"

    assert output_str != "mirror", \
        ("Cannot pass output_type='mirror' in Base.__init__(). Did you forget "
         "to pass `output_type=self.output_type` to a child estimator? "
         "Currently `cuml.global_settings.output_type==`{}`"
         ).format(cuml.global_settings.output_type)

    if isinstance(output_str, str):
        output_type = output_str.lower()
        # Check for valid output types + "input"
        if output_type in ['numpy', 'cupy', 'cudf', 'numba', 'input']:
            # Return the original version if nothing has changed, otherwise
            # return the lowered. This is to try and keep references the same
            # to support sklearn.base.clone() where possible
            return output_str if output_type == output_str else output_type

    # Did not match any acceptable value
    raise ValueError("output_type must be one of " +
                     "'numpy', 'cupy', 'cudf' or 'numba'" +
                     "Got: {}".format(output_str))


def _determine_stateless_output_type(output_type, input_obj):
    """
    This function determines the output type using the same steps that are
    performed in `cuml.common.base.Base`. This can be used to mimic the
    functionality in `Base` for stateless functions or objects that do not
    derive from `Base`.
    """

    # Default to the global type if not specified, otherwise, check the
    # output_type string
    temp_output = cuml.global_settings.output_type if output_type is None \
        else _check_output_type_str(output_type)

    # If we are using 'input', determine the the type from the input object
    if temp_output == 'input':
        temp_output = cuml.common.input_utils.determine_array_type(input_obj)

    return temp_output
