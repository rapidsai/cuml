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

# distutils: language = c++

import copy
import os
import inspect
import numbers
from importlib import import_module
from cuml.internals.device_support import GPU_ENABLED
from cuml.internals.safe_imports import (
    cpu_only_import,
    gpu_only_import_from,
    null_decorator,
)
np = cpu_only_import('numpy')
nvtx_annotate = gpu_only_import_from("nvtx", "annotate", alt=null_decorator)

try:
    from sklearn.utils import estimator_html_repr
except ImportError:
    estimator_html_repr = None


import cuml
import cuml.common
from cuml.common.sparse_utils import is_sparse
import cuml.internals.logger as logger
import cuml.internals
from cuml.internals import api_context_managers
import cuml.internals.input_utils
from cuml.internals.available_devices import is_cuda_available
from cuml.internals.device_type import DeviceType
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.input_utils import (
    determine_array_type,
    input_to_cuml_array,
    input_to_host_array,
    input_to_host_array_with_sparse_support,
    is_array_like
)
from cuml.internals.memory_utils import determine_array_memtype
from cuml.internals.mem_type import MemoryType
from cuml.internals.memory_utils import using_memory_type
from cuml.internals.output_type import (
    INTERNAL_VALID_OUTPUT_TYPES,
    VALID_OUTPUT_TYPES
)
from cuml.internals.array import CumlArray
from cuml.internals.safe_imports import (
    gpu_only_import, gpu_only_import_from
)

from cuml.internals.mixins import TagsMixin

cp_ndarray = gpu_only_import_from('cupy', 'ndarray')
cp = gpu_only_import('cupy')


IF GPUBUILD == 1:
    import pylibraft.common.handle
    import cuml.common.cuda


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

        stream = cuml.common.cuda.Stream()
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

    _hyperparam_interop_translator = {}

    def __init__(self, *,
                 handle=None,
                 verbose=False,
                 output_type=None,
                 output_mem_type=None):
        """
        Constructor. All children must call init method of this base class.

        """
        IF GPUBUILD == 1:
            self.handle = pylibraft.common.handle.Handle() if handle is None \
                else handle
        ELSE:
            self.handle = None

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
            if output_type is None else output_type)
        if output_mem_type is None:
            self.output_mem_type = cuml.global_settings.memory_type
        else:
            self.output_mem_type = MemoryType.from_str(output_mem_type)
        self._input_type = None
        self._input_mem_type = None
        self.target_dtype = None
        self.n_features_in_ = None

        nvtx_benchmark = os.getenv('NVTX_BENCHMARK')
        if nvtx_benchmark and nvtx_benchmark.lower() == 'true':
            self.set_nvtx_annotations()

    verbose = VerbosityDescriptor()

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

        if hasattr(self, 'sk_model_'):
            output += ' <sk_model_ attribute used>'
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

    def __getstate__(self):
        # getstate and setstate are needed to tell pickle to treat this
        # as regular python classes instead of triggering __getattr__
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getattr__(self, attr):
        """
        Redirects to `solver_model` if the attribute exists.
        """
        if attr == "solver_model":
            return self.__dict__['solver_model']
        if "solver_model" in self.__dict__.keys():
            return getattr(self.solver_model, attr)
        else:
            raise AttributeError(attr)

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
            self._set_output_mem_type(output_type)
        if target_dtype is not None:
            self._set_target_dtype(target_dtype)
        if n_features is not None:
            self._set_n_features_in(n_features)

    def _set_output_type(self, inp):
        self._input_type = determine_array_type(inp)

    def _set_output_mem_type(self, inp):
        self._input_mem_type = determine_array_memtype(
            inp
        )

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
        if cuml.global_settings.output_type == 'input':
            mem_type = determine_array_memtype(inp)

        return mem_type

    def _set_target_dtype(self, target):
        self.target_dtype = cuml.internals.input_utils.determine_array_dtype(
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
        # 'preserves_dtype' tag's Scikit definition currently only applies to
        # transformers and whether the transform method conserves the dtype
        # (in that case returns an empty list, otherwise the dtype it
        # casts to).
        # By default, our transform methods convert to self.dtype, but
        # we need to check whether the tag has been defined already.
        if hasattr(self, 'transform') and hasattr(self, 'dtype'):
            return {'preserves_dtype': [self.dtype]}
        return {}

    def _repr_mimebundle_(self, **kwargs):
        """Prepare representations used by jupyter kernels to display estimator"""
        if estimator_html_repr is not None:
            output = {"text/plain": repr(self)}
            output["text/html"] = estimator_html_repr(self)
            return output

    def set_nvtx_annotations(self):
        for func_name in ['fit', 'transform', 'predict', 'fit_transform',
                          'fit_predict']:
            if hasattr(self, func_name):
                msg = '{class_name}.{func_name} [{addr}]'
                msg = msg.format(class_name=self.__class__.__module__,
                                 func_name=func_name,
                                 addr=hex(id(self)))
                msg = msg[5:]  # remove cuml.
                func = getattr(self, func_name)
                func = nvtx_annotate(message=msg, domain="cuml_python")(func)
                setattr(self, func_name, func)

    @classmethod
    def _hyperparam_translator(cls, **kwargs):
        """
        This method is meant to do checks and translations of hyperparameters
        at estimator creating time.
        Each children estimator can override the method, returning either
        modifier **kwargs with equivalent options, or setting gpuaccel to False
        for hyperaparameters not supported by cuML yet.
        """
        gpuaccel = True
        # Copy it so we can modify it
        # we need to explicitly use UniversalBase because not all estimator
        # have it as the first parent in their MRO/inheritance, like
        # linear_regression
        translations = dict(UniversalBase._hyperparam_interop_translator)
        # Allow the derived class to overwrite the base class
        translations.update(cls._hyperparam_interop_translator)
        for parameter_name, value in kwargs.items():
            if parameter_name in translations:
                try:
                    remapping = translations[parameter_name][value]
                    if remapping == "NotImplemented":
                        gpuaccel = False
                    else:
                        kwargs[parameter_name] = remapping
                except (KeyError, TypeError):
                    pass  # Parameter value not found in translation dictionary

        return kwargs, gpuaccel


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
        if output_type in INTERNAL_VALID_OUTPUT_TYPES:
            # Return the original version if nothing has changed, otherwise
            # return the lowered. This is to try and keep references the same
            # to support sklearn.base.clone() where possible
            return output_str if output_type == output_str else output_type

    valid_output_types_str = ', '.join(
        [f"'{x}'" for x in VALID_OUTPUT_TYPES]
    )
    raise ValueError(
        f'output_type must be one of {valid_output_types_str}'
        f' Got: {output_str}'
    )


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
        temp_output = determine_array_type(input_obj)

    return temp_output


class UniversalBase(Base):
    # variable to enable dispatching non-implemented methods to CPU
    # estimators, experimental.
    _experimental_dispatching = False

    def import_cpu_model(self):
        # skip if the CPU estimator has been imported already
        if hasattr(self, '_cpu_model_class'):
            return
        if hasattr(self, '_cpu_estimator_import_path'):
            # if import path differs from the one of sklearn
            # look for _cpu_estimator_import_path
            estimator_path = self._cpu_estimator_import_path.split('.')
            model_path = '.'.join(estimator_path[:-1])
            model_name = estimator_path[-1]
        else:
            # import from similar path to the current estimator
            # class
            model_path = 'sklearn' + self.__class__.__module__[4:]
            model_name = self.__class__.__name__
        self._cpu_model_class = getattr(import_module(model_path), model_name)

        # Save list of available CPU estimator hyperparameters
        self._cpu_hyperparams = list(
            inspect.signature(self._cpu_model_class.__init__).parameters.keys()
        )

    def build_cpu_model(self, **kwargs):
        if hasattr(self, '_cpu_model'):
            return
        if kwargs:
            filtered_kwargs = kwargs
        else:
            filtered_kwargs = {}
            for keyword, arg in self._full_kwargs.items():
                if keyword in self._cpu_hyperparams:
                    filtered_kwargs[keyword] = arg
                else:
                    logger.info("Unused keyword parameter: {} "
                                "during CPU estimator "
                                "initialization".format(keyword))

        # initialize model
        self._cpu_model = self._cpu_model_class(**filtered_kwargs)

    def gpu_to_cpu(self):
        # transfer attributes from GPU to CPU estimator
        for attr in self.get_attr_names():
            if hasattr(self, attr):
                cu_attr = getattr(self, attr)
                if isinstance(cu_attr, CumlArray):
                    # transform cumlArray to numpy and set it
                    # as an attribute in the CPU estimator
                    setattr(self._cpu_model, attr, cu_attr.to_output('numpy'))
                elif isinstance(cu_attr, cp_ndarray):
                    # transform cupy to numpy and set it
                    # as an attribute in the CPU estimator
                    setattr(self._cpu_model, attr, cp.asnumpy(cu_attr))
                else:
                    # transfer all other types of attributes directly
                    setattr(self._cpu_model, attr, cu_attr)

    def cpu_to_gpu(self):
        # transfer attributes from CPU to GPU estimator
        with using_memory_type(
            (MemoryType.host, MemoryType.device)[
                is_cuda_available()
            ]
        ):
            for attr in self.get_attr_names():
                if hasattr(self._cpu_model, attr):
                    cpu_attr = getattr(self._cpu_model, attr)
                    # if the cpu attribute is an array
                    if isinstance(cpu_attr, np.ndarray):
                        # get data order wished for by
                        # CumlArrayDescriptor
                        if hasattr(self, attr + '_order'):
                            order = getattr(self, attr + '_order')
                        else:
                            order = 'K'
                        # transfer array to gpu and set it as a cuml
                        # attribute
                        cuml_array = input_to_cuml_array(
                            cpu_attr,
                            order=order,
                            convert_to_mem_type=(
                                MemoryType.host,
                                MemoryType.device
                            )[is_cuda_available()]
                        )[0]
                        setattr(self, attr, cuml_array)
                    else:
                        # transfer all other types of attributes
                        # directly
                        setattr(self, attr, cpu_attr)

    def args_to_cpu(self, *args, **kwargs):
        # put all the args on host
        new_args = tuple(
            input_to_host_array_with_sparse_support(arg) for arg in args
        )

        # put all the kwargs on host
        new_kwargs = dict()
        for kw, arg in kwargs.items():
            # if array-like, ensure array-like is on the host
            if is_array_like(arg):
                new_kwargs[kw] = input_to_host_array_with_sparse_support(arg)
            # if Real or string, pass as is
            elif isinstance(arg, (numbers.Real, str)):
                new_kwargs[kw] = arg
            else:
                raise ValueError(f"Unable to process argument {kw}")

        new_kwargs.pop("convert_dtype", None)
        return new_args, new_kwargs

    def dispatch_func(self, func_name, gpu_func, *args, **kwargs):
        """
        This function will dispatch calls to training and inference according
        to the global configuration. It should work for all estimators
        sufficiently close the scikit-learn implementation as it uses
        it for training and inferences on host.

        Parameters
        ----------
        func_name : string
            name of the function to be dispatched
        gpu_func : function
            original cuML function
        args : arguments
            arguments to be passed to the function for the call
        kwargs : keyword arguments
            keyword arguments to be passed to the function for the call
        """
        # look for current device_type
        # device_type = cuml.global_settings.device_type
        device_type = self._dispatch_selector(func_name, *args, **kwargs)

        if device_type == DeviceType.device:
            # call the function from the GPU estimator
            if GlobalSettings().accelerator_active:
                logger.debug(f"cuML: Performing {func_name} in GPU")
            return gpu_func(self, *args, **kwargs)

        # CPU case
        elif device_type == DeviceType.host:
            # check if a CPU model already exists
            if not hasattr(self, '_cpu_model'):
                # import CPU estimator from library
                self.import_cpu_model()
                # create an instance of the estimator
                self.build_cpu_model()

                # new CPU model + CPU inference
                if func_name not in ['fit', 'fit_transform', 'fit_predict']:
                    # transfer trained attributes from GPU to CPU
                    self.gpu_to_cpu()

            # ensure args and kwargs are on the CPU
            args, kwargs = self.args_to_cpu(*args, **kwargs)

            # get the function from the CPU estimator
            cpu_func = getattr(self._cpu_model, func_name)
            # call the function from the CPU estimator
            logger.debug(f"cuML: Performing {func_name} in CPU")
            res = cpu_func(*args, **kwargs)

            # CPU training
            if func_name in ['fit', 'fit_transform', 'fit_predict']:
                # mirror input type
                self._set_output_type(args[0])
                self._set_output_mem_type(args[0])

                # transfer trained attributes from CPU to GPU
                self.cpu_to_gpu()

                # return the cuml estimator when training
                if func_name == 'fit':
                    return self

            # return function result
            return res

    def _dispatch_selector(self, func_name, *args, **kwargs):
        """
        """
        # check for sparse inputs and whether estimator supports them
        sparse_support = "sparse" in self._get_tags()["X_types_gpu"]

        if args and is_sparse(args[0]):
            if sparse_support:
                return DeviceType.device
            elif GlobalSettings().accelerator_active and not sparse_support:
                logger.info(
                    f"cuML: Estimator {self} does not support sparse inputs in GPU."
                )
                return DeviceType.host
            else:
                raise NotImplementedError(
                    "Estimator does not support sparse inputs currently"
                )

        # if not using accelerator, then return global device
        if not hasattr(self, "_gpuaccel"):
            return cuml.global_settings.device_type

        # if using accelerator and doing inference, always use GPU
        elif func_name not in ['fit', 'fit_transform', 'fit_predict']:
            device_type = DeviceType.device

        # otherwise we select CPU when _gpuaccel is off
        elif not self._gpuaccel:
            device_type = DeviceType.host
        else:
            if not self._should_dispatch_cpu(func_name, *args, **kwargs):
                device_type = DeviceType.device
            else:
                device_type = DeviceType.host

        return device_type

    def _should_dispatch_cpu(self, func_name, *args, **kwargs):
        """
        This method is meant to do checks of data sizes and other things
        at fit and other method call time, to decide where to disptach
        a function. For hyperparameters of the estimator,
        see the method _hyperparam_translator.
        Each estimator inheritting from UniversalBase can override this
        method to have custom rules of when to dispatch to CPU depending
        on the data passed to fit/predict...
        """

        return False

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError as ex:
            # When using cuml.experimental.accel or setting the
            # self._experimental_dispatching flag to True, we look for methods
            # that are not in the cuML estimator in the host estimator
            gs = GlobalSettings()
            if gs.accelerator_active or self._experimental_dispatching:
                # we don't want to special sklearn dispatch cloning function
                # so that cloning works with this class as a regular estimator
                # without __sklearn_clone__
                if attr == "__sklearn_clone__":
                    raise ex

                self.import_cpu_model()
                if hasattr(self._cpu_model_class, attr):
                    # we turn off and cache the dispatching variables off so that
                    # build_cpu_model and gpu_to_cpu don't recurse infinitely
                    orig_dispatching = self._experimental_dispatching
                    orig_accelerator_active = gs.accelerator_active

                    self._experimental_dispatching = False
                    gs.accelerator_active = False
                    try:
                        self.build_cpu_model()
                        self.gpu_to_cpu()
                    finally:
                        # Reset back to original values
                        self._experimental_dispatching = orig_dispatching
                        gs.accelerator_active = orig_accelerator_active

                    return getattr(self._cpu_model, attr)
            raise

    def as_sklearn(self, deepcopy=False):
        """
        Convert the current GPU-accelerated estimator into a scikit-learn estimator.

        This method imports and builds an equivalent CPU-backed scikit-learn model,
        transferring all necessary parameters from the GPU representation to the
        CPU model. After this conversion, the returned object should be a fully
        compatible scikit-learn estimator, allowing you to use it in standard
        scikit-learn pipelines and workflows.

        Parameters
        ----------
        deepcopy : boolean (default=False)
            Whether to return a deepcopy of the internal scikit-learn estimator of
            the cuML models. cuML models internally have CPU based estimators that
            could be updated. If you intend to use both the cuML and the scikit-learn
            estimators after using the method in parallel, it is recommended to set
            this to True to avoid one overwriting data of the other.

        Returns
        -------
        sklearn.base.BaseEstimator
            A scikit-learn compatible estimator instance that mirrors the trained
            state of the current GPU-accelerated estimator.

        """
        self.import_cpu_model()
        self.build_cpu_model()
        self.gpu_to_cpu()
        if deepcopy:
            return copy.deepcopy(self._cpu_model)
        else:
            return self._cpu_model

    @classmethod
    def from_sklearn(cls, model):
        """
        Create a GPU-accelerated estimator from a scikit-learn estimator.

        This class method takes an existing scikit-learn estimator and converts it
        into the corresponding GPU-backed estimator. It imports any required CPU
        model definitions, stores the given scikit-learn model internally, and then
        transfers the model parameters and state onto the GPU.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            A fitted scikit-learn estimator from which to create the GPU-accelerated
            version.

        Returns
        -------
        cls
            A new instance of the GPU-accelerated estimator class that mirrors the
            state of the input scikit-learn estimator.

        Notes
        -----
        - `output_type` of the estimator is set to "numpy"
            by default, as these cannot be inferred from training arguments. If
            something different is required, then please use cuML's output_type
            configuration utilities.
        """
        estimator = cls()
        estimator.import_cpu_model()
        estimator._cpu_model = model
        params, gpuaccel = cls._hyperparam_translator(**model.get_params())
        params = {key: params[key] for key in cls._get_param_names() if key in params}
        estimator.set_params(**params)
        estimator.cpu_to_gpu()

        # we need to set an output type here since
        # we cannot infer from training args.
        # Setting to numpy seems like a reasonable default for matching the
        # deserialized class by default.
        estimator.output_type = "numpy"
        estimator.output_mem_type = MemoryType.host

        return estimator

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
        if GlobalSettings().accelerator_active or self._experimental_dispatching:
            return self._cpu_model.get_params(deep=deep)
        else:
            return super().get_params(deep=deep)

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
        if GlobalSettings().accelerator_active or self._experimental_dispatching:
            self._cpu_model.set_params(**params)
            params, gpuaccel = self._hyperparam_translator(**params)
            params = {key: params[key] for key in self._get_param_names() if key in params}
        super().set_params(**params)
        return self
