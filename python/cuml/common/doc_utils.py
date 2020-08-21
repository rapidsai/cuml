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

"""
Decorators to generate common docstrings in the codebase.
Dense datatypes are currently the default, if you're a developer that landed
here, the docstrings apply to every parameter to which the decorators
are applied. The docstrings are generated at import time.

There are 2 decorators:
- generate_docstring: Meant to be used by fit/predict/et.al methods that have
    the typical signatures (i.e. fit(x,y) or predict(x)). It detects the
    parameters and default values and generates the appropriate docstring,
    with some configurability for shapes and formats.
- insert_into_docstring: More flexible but less automatic method, meant to be
    used by functions that use our common dense or sparse datatypes, but have
    many more custom parameters that are particular to the class(es) as opposed
    to being common in the codebase. Allows to keep our documentation up to
    date and correct with minimal changes by keeping our common datatypes
    concentrated here. NearestNeigbors is a good example of this use case.

More data types can be added as we need them.

cuml.dask datatype version of the docstrings will come in a future update.

"""

from inspect import signature


_parameters_docstrings = {
    'dense':
    '{name} : array-like (device or host) shape = {shape}\n'
    '    Dense matrix containing floats or doubles.\n'
    '    Acceptable formats: CUDA array interface compliant objects like\n'
    '    CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas\n'
    '    DataFrame/Series.',

    'dense_anydtype':
    '{name} : array-like (device or host) shape = {shape}\n'
    '    Dense matrix of any dtype.\n'
    '    Acceptable formats: CUDA array interface compliant objects like\n'
    '    CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas\n'
    '    DataFrame/Series.',

    'dense_intdtype':
    '{name} : array-like (device or host) shape = {shape}\n'
    '    Dense matrix of type np.int32.\n'
    '    Acceptable formats: CUDA array interface compliant objects like\n'
    '    CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas\n'
    '    DataFrame/Series.',

    'sparse':
    '{name} : sparse array-like (device) shape = {shape}\n'
    '    Dense matrix containing floats or doubles.\n'
    '    Acceptable formats: cupy.sparse',

    'dense_sparse':
    '{name} : array-like (device or host) shape = {shape}\n'
    '    Dense or sparse matrix containing floats or doubles.\n'
    '    Acceptable dense formats: CUDA array interface compliant objects like\n'  # noqa
    '    CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas\n'
    '    DataFrame/Series.',

    'convert_dtype_fit':
    'convert_dtype : bool, optional (default = {default})\n'
    '    When set to True, the train method will, when necessary, convert\n'
    '    y to be the same data type as X if they differ. This\n'
    '    will increase memory used for the method.',

    'convert_dtype_other':
    'convert_dtype : bool, optional (default = {default})\n'
    '    When set to True, the {func_name} method will, when necessary,\n'
    '    convert the input to the data type which was used to train the\n'
    '    model. This will increase memory used for the method.',

    'convert_dtype_single':
    'convert_dtype : bool, optional (default = {default})\n'
    '    When set to True, the method will automatically\n'
    '    convert the inputs to {dtype}.',

    'sample_weight':
    'sample_weight : array-like (device or host) shape = (n_samples,), default={default}\n'  # noqa
    '    The weights for each observation in X. If None, all observations\n'
    '    are assigned equal weight.\n'
    '    Acceptable dense formats: CUDA array interface compliant objects like\n'  # noqa
    '    CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas\n'
    '    DataFrame/Series.',  # noqa
    'return_sparse':
    'return_sparse : bool, optional (default = {default})\n'
    '    Ignored when the model is not fit on a sparse matrix\n'
    '    If True, the method will convert the result to a\n'
    '    cupy.sparse.csr_matrix object.\n'
    '    NOTE: Currently, there is a loss of information when converting\n'
    '    to csr matrix (cusolver bug). Default will be switched to True\n'
    '    once this is solved.',

    'sparse_tol':
    'sparse_tol : float, optional (default = {default})\n'
    '    Ignored when return_sparse=False.\n'
    '    If True, values in the inverse transform below this parameter\n'
    '    are clipped to 0.'
}

_parameter_possible_values = ['name',
                              'type',
                              'shape',
                              'default',
                              'description',
                              'accepted']

_return_values_docstrings = {
    'dense':
    '{name} : cuDF, CuPy or NumPy object depending on cuML\'s output type configuration, shape = {shape}\n'  # noqa
    '    {description}\n\n    For more information on how to configure cuML\'s output type,\n'  # noqa
    '    refer to: `Output Data Type Configuration`_.',  # noqa

    'dense_sparse':
    '{name} : cuDF, CuPy or NumPy object depending on cuML\'s output type configuration, cupy.sparse for sparse output, shape = {shape}\n'  # noqa
    '    {description}\n\n    For more information on how to configure cuML\'s dense output type,\n'  # noqa
    '    refer to: `Output Data Type Configuration`_.',  # noqa

    'dense_datatype':
    'cuDF, CuPy or NumPy object depending on cuML\'s output type'
    'configuration, shape ={shape}',

    'dense_sparse_datatype':
    'cuDF, CuPy or NumPy object depending on cuML\'s output type'
    'configuration, shape ={shape}',

    'custom_type':
    '{name} : {type}\n'
    '    {description}'
}

_return_values_possible_values = ['name',
                                  'type',
                                  'shape',
                                  'description']

_simple_params = ['return_sparse',
                  'sparse_tol',
                  'sample_weight']


def generate_docstring(X='dense',
                       X_shape='(n_samples, n_features)',
                       y='dense',
                       y_shape='(n_samples, 1)',
                       convert_dtype_cast=False,
                       skip_parameters=[],
                       skip_parameters_heading=False,
                       prepend_parameters=True,
                       parameters=False,
                       return_values=False):
    """
    Decorator to generate dostrings of common functions in the codebase.
    It will auto detect what parameters and default values the function has.
    Unfortunately due to using cython, we cannot (cheaply) do detection of
    return values.

    Currently auto detected variables include:
    - X
    - y
    - convert_dtype
    - sample_weights
    - return_sparse
    - sparse_tol

    Typical usage scenarios:

    Examples
    --------

    # for a function that passes all dense parameters, no need to specify
    # anything, and the decorator auto detects the parameters and defaults

    @generate_docstring()
    def fit(self, X, y, convert_dtype=True):

    # for a function that takes X as dense or sparse

    @generate_docstring(X='dense_sparse')
    def fit(self, X, y, sample_weight=None):

    # to specify return values

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})


    Parameters
    -----------
    X : str (default = 'dense')
        Data type of variable X. Currently accepted types are: dense,
        dense_anydtype, dense_intdtype, sparse, dense_sparse
    X_shape : str (default = '(n_samples, n_features)')
        Shape of variable X
    y : str (default = 'dense')
        Data type of variable y. Currently accepted types are: dense,
        dense_anydtype, dense_intdtype, sparse, dense_sparse
    y_shape : str (default = '(n_samples, 1)')
        Shape of variable y
    convert_dtype_cast : Boolean or str (default = False)
        If not false, use it to specify when convert_dtype is used to convert
        to a single specific dtype (as opposed to converting the dtype of one
        variable to the dtype of another for example). Example of this is how
        NearestNeighbors and UMAP use convert_dtype to convert inputs to
        np.float32.
    skip_parameters : list of str (default = [])
        Use if you want the decorator to skip generating a docstring entry
        for a specific parameter
    skip_parameters_heading : boolean (default = False)
        Set to True to not generate the Parameters section heading
    prepend_parameters : boolean (default = True)
        Use when setting skip_parameters_heading to True, so that the
        parameters inserted by the decorator are inserted before the
        parameters you already have in your docstring.
    return_values : dict or list of dicts (default = False)
        Use to generate docstrings of return values. One dictionary per
        return value, this is the format:
            {'name': 'name_of_variable',
             'type': 'data type of returned value',
             'description': 'Description of variable',
             'shape': 'shape of returned variable'}

        If type is one of dense or dense_sparse then the type is generated
        from the corresponding entry in _return_values_docstrings. Otherwise
        the type is used as specified.
    """

    def deco(func):
        params = signature(func).parameters

        # Add parameter section header if needed, can be skipped
        if(('X' in params or 'y' in params or parameters) and not
                skip_parameters_heading):

            func.__doc__ += \
                '\nParameters\n----------\n'

        # Check if we want to prepend the parameters
        if skip_parameters_heading and prepend_parameters:
            loc_pars = func.__doc__.find("----------") + 11
            current_params_in_docstring = \
                func.__doc__[loc_pars:]

            func.__doc__ = func.__doc__[:loc_pars]

        # Process each parameter
        for par, value in params.items():
            if par == 'self':
                pass

            # X and y are the most common
            elif par in ['X', 'y'] and par not in skip_parameters:
                func.__doc__ += \
                    _parameters_docstrings[X].format(name=par,
                                                     shape=X_shape)

            # convert_dtype requires some magic to distinguish
            # whether we use the fit version or the version
            # for the other methods.
            elif par == 'convert_dtype' and par not in skip_parameters:
                if not convert_dtype_cast:
                    if func.__name__ == 'fit':
                        k = 'convert_dtype_fit'
                    else:
                        k = 'convert_dtype_other'

                    func.__doc__ += \
                        _parameters_docstrings[k].format(
                            default=params['convert_dtype'].default,
                            func_name=func.__name__
                        )

                else:
                    func.__doc__ += \
                        _parameters_docstrings['convert_dtype_single'].format(
                            default=params['convert_dtype'].default,
                            dtype=convert_dtype_cast
                        )

            # All other parameters only take a default (for now).
            else:
                if par in _simple_params:
                    func.__doc__ += \
                        _parameters_docstrings[par].format(
                            default=params[par].default
                        )
            func.__doc__ += '\n\n'

        if skip_parameters_heading and prepend_parameters:
            # indexing at 8 to match indentation of inserted parameters
            # this can be replaced with indentation detection
            # https://github.com/rapidsai/cuml/issues/2714
            func.__doc__ += current_params_in_docstring[8:]

        # Add return section header if needed, no option to skip currently.
        if(return_values):
            func.__doc__ += \
                '\nReturns\n----------\n'

            # convenience call to allow users to pass a single return
            # value as a dictionary instead of a list of dictionaries
            rets = [return_values] if not isinstance(return_values, list) \
                else return_values

            # process each entry in the return_values
            # auto naming of predicted variable names will be a
            # future improvement
            for ret in rets:
                if ret['type'] in _return_values_docstrings:
                    key = ret['type']
                    # non custom types don't take the type parameter
                    del ret['type']
                else:
                    key = 'custom_type'

                # ret is already a dictionary, we just use it for the named
                # parameters
                func.__doc__ += \
                    _return_values_docstrings[key].format(
                        **ret
                    )
                func.__doc__ += '\n\n'

        return func
    return deco


def insert_into_docstring(parameters=False,
                          return_values=False):
    """
    Decorator to insert a single entry into an existing docstring. Use
    standard {} format parameters in your docstring, and then use this
    decorator to insert the standard type information for that variable.

    Examples
    --------

    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)')],
                           return_values=[('dense', '(n_samples, n_features)'),
                                          ('dense',
                                           '(n_samples, n_features)')])
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True,
                   convert_dtype=True):
        \"""
        Query the GPU index for the k nearest neighbors of column vectors in X.

        Parameters
        ----------
        X : {}

        n_neighbors : Integer
            Number of neighbors to search. If not provided, the n_neighbors
            from the model instance is used (default=10)

        return_distance: Boolean
            If False, distances will not be returned

        convert_dtype : bool, optional (default = True)
            When set to True, the kneighbors method will automatically
            convert the inputs to np.float32.

        Returns
        -------
        distances : {}
            The distances of the k-nearest neighbors for each column vector
            in X

        indices : {}
            The indices of the k-nearest neighbors for each column vector in X
        \"""

    Parameters
    ----------
    parameters : list of tuples
        List of tuples, each tuple containing: (type, shape) for the type
        and shape of each parameter to be inserted. Current accepted values
        are `dense` and `dense_sparse`.
    return_values : list of tuples
        List of tuples, each tuple containing: (type, shape) for the type
        and shape of each parameter to be inserted. Current accepted values
        are `dense` and `dense_sparse`.

    """

    def deco(func):
        # List of parameters to use in `format` call of the docstring
        to_add = []

        # See if we need to add parameter data types
        if parameters:
            for par in parameters:
                to_add.append(
                    _parameters_docstrings[par[0]][9:].format(shape=par[1])
                )

        # See if we need to add return value data types
        if return_values:
            for ret in return_values:
                to_add.append(
                    _return_values_docstrings[ret[0] + '_datatype'].format(
                        shape=ret[1]
                    )
                )

        if(len(to_add) > 0):
            func.__doc__ = str(func.__doc__).format(*to_add)

        func.__doc__ += '\n\n'

        return func
    return deco
