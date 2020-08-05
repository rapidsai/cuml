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

from functools import wraps
from inspect import signature

_parameters_docstrings = {
    'dense':
    '{} : array-like (device or host) shape = {} \n \
        Dense matrix containing floats or doubles. \
        Acceptable formats: CUDA array interface compliant objects like \
        CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas \
        DataFrame/Series.',

    'sparse':
    '{} : sparse array-like (device) shape = {} \n \
        Dense matrix containing floats or doubles. \
        Acceptable formats: cupy.sparse',

    'dense_sparse':
    '{} : array-like (device or host) shape = {} \n \
        Dense or sparse matrix containing floats or doubles. \
        Acceptable dense formats: CUDA array interface compliant objects like \
        CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas \
        DataFrame/Series.',

    'convert_dtype_fit':
    'convert_dtype : bool, optional (default = {})\n \
        When set to True, the train method will, when necessary, convert \
        y to be the same data type as X if they differ. This \
        will increase memory used for the method.',

    'convert_dtype_other':
    'convert_dtype : bool, optional (default = {})\n \
        When set to True, the {} method will, when necessary, convert \
        the input to the data type which was used to train the model. This \
        will increase memory used for the method.',

    'convert_dtype_single':
    'convert_dtype : bool, optional (default = {})\n \
            When set to True, the method will automatically \
            convert the inputs to {}.',

    'sample_weight':
    'sample_weight : array-like (device or host) shape = (n_samples,), default={} \n\
                The weights for each observation in X. If None, all observations  \
                are assigned equal weight. \
                Acceptable dense formats: CUDA array interface compliant objects like \
                CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas \
                DataFrame/Series.',  # noqa

    'return_sparse':
    'return_sparse : bool, optional (default = {}) \n \
            Ignored when the model is not fit on a sparse matrix \
            If True, the method will convert the result to a \
            cupy.sparse.csr_matrix object. \n \
            NOTE: Currently, there is a loss of information when converting \
            to csr matrix (cusolver bug). Default will be switched to True \
            once this is solved.',

    'sparse_tol':
    'sparse_tol : float, optional (default = {}) \n\
            Ignored when return_sparse=False. \
            If True, values in the inverse transform below this parameter \
            are clipped to 0.',

    '_custom_docstring_default':
    '{}: {} (default = {}) \n \
        {}',

    '_custom_docstring_no_default':
    '{}: {} \n \
        {}'
}

_parameter_possible_values = ['name',
                              'type',
                              'shape',
                              'default',
                              'description',
                              'accepted']

_return_values_docstrings = {
    'dense':
    '{} : cuDF, CuPy or NumPy object depending on cuML\'s output type configuration, shape = {}\n \
        {} For more information on how to configure cuML\'s output type, \
        refer to: `Output Data Type Configuration`_.',  # noqa

    'dense_sparse':
    '{} : cuDF, CuPy or NumPy object depending on cuML\'s output type configuration, cupy.sparse for sparse output, shape = {}\n \
        {} \n For more information on how to configure cuML\'s dense output type, \
        refer to: `Output Data Type Configuration`_.',  # noqa

    'dense_datatype':
    'cuDF, CuPy or NumPy object depending on cuML\'s output type \
    configuration, shape ={}',

    'dense_sparse_datatype':
    'cuDF, CuPy or NumPy object depending on cuML\'s output type \
    configuration, shape ={}',

    'custom_type':
    '{} : {} \n \
        {}'

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
                       parameters=False,
                       return_values=False):
    def deco(func):
        @wraps(func)
        def docstring_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        docstring_wrapper.__doc__ = str(func.__doc__)

        params = signature(func).parameters
        print(params)

        if(('X' in params or 'y' in params or parameters) and not
                skip_parameters_heading):
            docstring_wrapper.__doc__ += \
                '\nParameters \n ---------- \n'

        for par, value in params.items():
            if par == 'self':
                pass
            elif par == 'X' and par not in skip_parameters:
                docstring_wrapper.__doc__ += \
                    _parameters_docstrings[X].format('X', X_shape)

            elif par == 'y' and par not in skip_parameters:
                docstring_wrapper.__doc__ += \
                    _parameters_docstrings[y].format('y', y_shape)

            elif par == 'convert_dtype' and par not in skip_parameters:
                if not convert_dtype_cast:
                    if func.__name__ == 'fit':
                        k = 'convert_dtype_fit'
                    else:
                        k = 'convert_dtype_other'

                    docstring_wrapper.__doc__ += \
                        _parameters_docstrings[k].format(
                            params['convert_dtype'].default, func.__name__
                        )

                else:
                    docstring_wrapper.__doc__ += \
                        _parameters_docstrings['convert_dtype_single'].format(
                            params['convert_dtype'].default, convert_dtype_cast
                        )

            else:
                if par in _simple_params:
                    docstring_wrapper.__doc__ += \
                        _parameters_docstrings[par].format(
                            params[par].default
                        )
            docstring_wrapper.__doc__ += '\n\n'

        if(return_values):
            docstring_wrapper.__doc__ += \
                '\nReturns \n ---------- \n'

            rets = [return_values] if not isinstance(return_values, list) \
                else return_values

            for ret in rets:
                if ret['type'] in _return_values_docstrings:
                    key = ret['type']
                    del ret['type']
                else:
                    key = 'custom_type'
                res_values = (ret[b] for b in _return_values_possible_values
                              if b in ret.keys())

                docstring_wrapper.__doc__ += \
                    _return_values_docstrings[key].format(
                        *res_values
                    )
                docstring_wrapper.__doc__ += '\n\n'

        return docstring_wrapper
    return deco


def insert_into_docstring(parameters=False,
                          return_values=False):
    def deco(func):
        @wraps(func)
        def docstring_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        to_add = []

        if parameters:
            for par in parameters:
                to_add.append(
                    _parameters_docstrings[par[0]][5:].format(par[1])
                )

        if return_values:
            for ret in return_values:
                to_add.append(
                    _return_values_docstrings[ret[0] + '_datatype'].format(ret[1])
                )
            docstring_wrapper.__doc__ = str(func.__doc__).format(*to_add)

        docstring_wrapper.__doc__ += '\n\n'

        return docstring_wrapper
    return deco
