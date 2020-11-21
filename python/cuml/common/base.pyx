#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import inspect

import cuml.common
import cuml.common.cuda
import cuml.common.logger as logger
import cuml.internals
import cuml.raft.common.handle
from cuml.common.doc_utils import generate_docstring
import cuml.common.input_utils


# tag system based on experimental tag system from Scikit-learn >=0.21
# https://scikit-learn.org/stable/developers/develop.html#estimator-tags
_default_tags = {
    # cuML specific tags
    'preferred_input_order': None,
    'X_types_gpu': ['2darray'],

    # Scikit-learn API standard tags
    'non_deterministic': False,
    'requires_positive_X': False,
    'requires_positive_y': False,
    'X_types': ['2darray'],
    'poor_score': False,
    'no_validation': False,
    'multioutput': False,
    'allow_nan': False,
    'stateless': False,
    'multilabel': False,
    '_skip_test': False,
    '_xfail_checks': False,
    'multioutput_only': False,
    'binary_only': False,
    'requires_fit': True,
    'requires_y': False,
    'pairwise': False,
}


class Base(metaclass=cuml.internals.BaseMetaClass):
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
        module level, `cuml.global_output_type`.
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
        handle = cuml.Handle()
        handle.setStream(stream)

        algo = MyAlgo(handle=handle)
        algo.fit(...)
        result = algo.predict(...)

        # final sync of all gpu-work launched inside this object
        # this is same as `cuml.cuda.Stream.sync()` call, but safer in case
        # the default stream inside the `cumlHandle` is being used
        base.handle.sync()
        del base  # optional!
    """
    def __init__(self,
                 handle=None,
                 verbose=False,
                 output_type=None):
        """
        Constructor. All children must call init method of this base class.

        """
        self.handle = cuml.raft.common.handle.Handle() if handle is None \
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
            cuml.global_output_type if output_type is None else output_type)
        self._input_type = None
        self.target_dtype = None
        self.n_features_in_ = None

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
        return string + ')'

    def enable_rmm_pool(self):
        self.handle.enable_rmm_pool()

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
        """
        Redirects to `solver_model` if the attribute exists.
        """
        if attr == "solver_model":
            return self.__dict__['solver_model']
        if "solver_model" in self.__dict__.keys():
            return getattr(self.solver_model, attr)
        else:
            raise AttributeError

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
        output_type = cuml.global_output_type

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

    def _get_tags(self):
        # method and code based on scikit-learn 0.21 _get_tags functionality:
        # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        collected_tags = _default_tags
        for cl in reversed(inspect.getmro(self.__class__)):
            if hasattr(cl, '_more_tags') and cl != Base:
                more_tags = cl._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags


class RegressorMixin:
    """Mixin class for regression estimators in cuML"""

    _estimator_type = "regressor"

    @generate_docstring(
        return_values={
            'name': 'score',
            'type': 'float',
            'description': 'R^2 of self.predict(X) '
                           'wrt. y.'
        })
    @cuml.internals.api_base_return_any_skipall
    def score(self, X, y, **kwargs):
        """
        Scoring function for regression estimators

        Returns the coefficient of determination R^2 of the prediction.

        """
        from cuml.metrics.regression import r2_score

        if hasattr(self, 'handle'):
            handle = self.handle
        else:
            handle = None

        preds = self.predict(X, **kwargs)
        return r2_score(y, preds, handle=handle)

    def _more_tags(self):
        return {
            'requires_y': True
        }


class ClassifierMixin:
    """Mixin class for classifier estimators in cuML"""

    _estimator_type = "classifier"

    @generate_docstring(
        return_values={
            'name':
                'score',
            'type':
                'float',
            'description': ('Accuracy of self.predict(X) wrt. y '
                            '(fraction where y == pred_y)')
        })
    @cuml.internals.api_base_return_any_skipall
    def score(self, X, y, **kwargs):
        """
        Scoring function for classifier estimators based on mean accuracy.

        """
        from cuml.metrics.accuracy import accuracy_score

        if hasattr(self, 'handle'):
            handle = self.handle
        else:
            handle = None

        preds = self.predict(X, **kwargs)
        return accuracy_score(y, preds, handle=handle)

    def _more_tags(self):
        return {
            'requires_y': True
        }


# Internal, non class owned helper functions
def _check_output_type_str(output_str):

    if (output_str is None):
        return "input"

    assert output_str != "mirror", \
        ("Cannot pass output_type='mirror' in Base.__init__(). Did you forget "
         "to pass `output_type=self.output_type` to a child estimator? "
         "Currently `cuml.global_output_type==`{}`"
         ).format(cuml.global_output_type)

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
    temp_output = cuml.global_output_type if output_type is None \
        else _check_output_type_str(output_type)

    # If we are using 'input', determine the the type from the input object
    if temp_output == 'input':
        temp_output = cuml.common.input_utils.determine_array_type(input_obj)

    return temp_output
