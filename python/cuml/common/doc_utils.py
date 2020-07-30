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


def generate_docstring(X=True,
                       X_shape=None,
                       y=False,
                       y_shape=None,
                       convert_dtype_fit=False,
                       convert_dtype_predict=False,
                       out_dtype=False,
                       out_dtype_values=None,
                       return_value=False,
                       return_value_shape=None,
                       return_value_description=None):
    def deco(func):
        @wraps(func)
        def docstring_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        docstring_wrapper.__doc__ = str(func.__doc__) + \
    'Parameters \n \
    ---------- \n \
    '
        if(X):
            docstring_wrapper.__doc__ += \
    'X : array-like (device or host) shape = (n_samples, n_features) \n \
           Dense matrix (floats or doubles) of shape (n_samples, n_features). \n \
           Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device \n \
           ndarray, cuda array interface compliant array like CuPy. \n\n \
           '

        if(y):
            docstring_wrapper.__doc__ += \
    'y : array-like (device or host) shape = (n_samples, 1) \n \
                Dense vector (floats or doubles) of shape (n_samples, 1). \n \
                Acceptable formats: cuDF Series, NumPy ndarray, Numba device \n \
                ndarray, cuda array interface compliant array like CuPy \n\n \
    '

        if(convert_dtype_fit):

            docstring_wrapper.__doc__ += \
    'convert_dtype : bool, optional (default = True) \n \
            When set to True, the fit method will, when necessary, convert \n \
            y to be the same data type as X if they differ. This \n \
            will increase memory used for the method. \n\n \
    '

        if(convert_dtype_fit):

            docstring_wrapper.__doc__ += \
    'convert_dtype : bool, optional (default = False) \n \
            When set to True, the predict method will, when necessary, convert \n \
            the input to the data type which was used to train the model. This \n \
            will increase memory used for the method. \n\n \
    '

        if(convert_dtype_predict):

            docstring_wrapper.__doc__ += \
    'convert_dtype : bool, optional (default = False) \n \
            When set to True, the predict method will, when necessary, convert \n \
            the input to the data type which was used to train the model. This \n \
            will increase memory used for the method. \n\n \
    '

        if(out_dtype):
            docstring_wrapper.__doc__ += \
    'out_dtype: dtype (default = "int32") \n \
                Determines the precision of the output labels array.\n \
                valid values are { "int32", np.int32, \n \
                "int64", np.int64}. \n\n \
    '

        if(return_value):
            docstring_wrapper.__doc__ += \
    '\nReturns \n \
    ------- \n \
    '

            docstring_wrapper.__doc__ += return_value

            docstring_wrapper.__doc__ += \
    ' : cuDF, CuPy or NumPy object depending on cuML\'s output type configuration \n \
    '

            if(return_value_description is not None):
                docstring_wrapper.__doc__ += return_value_description

        return docstring_wrapper
    return deco
