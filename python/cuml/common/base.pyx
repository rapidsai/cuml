#
# Copyright (c) 2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


import cuml.common.handle
import cuml.common.cuda

import cudf


class Base:
    """
    Base class for all the ML algos. It handles some of the common operations
    across all algos. Every ML algo class exposed at cython level must inherit
    from this class.

    Examples
    --------

    .. code-block:: python

        import cuml

        # assuming this ML algo has separate 'fit' and 'predict' methods
        class MyAlgo(cuml.Base):
            def __init__(self, ...):
                super(MyAlgo, self).__init__(...)
                # other setup logic

            def fit(self, ...):
                # train logic goes here

            def predict(self, ...):
                # inference logic goes here

            def get_param_names(self):
                # return a list of hyperparam names supported by this algo

        stream = cuml.cuda.Stream()
        handle = cuml.Handle()
        handle.setStream(stream)
        handle.enableRMM()   # Enable RMM as the device-side allocator
        algo = MyAlgo(handle=handle)
        algo.fit(...)
        result = algo.predict(...)
        # final sync of all gpu-work launched inside this object
        # this is same as `cuml.cuda.Stream.sync()` call, but safer in case
        # the default stream inside the `cumlHandle` is being used
        base.handle.sync()
        del base  # optional!
    """

    def __init__(self, handle=None, verbose=False):
        """
        Constructor. All children must call init method of this base class.

        Parameters
        ----------
        handle : cuml.Handle
               If it is None, a new one is created just for this class
        verbose : bool
                Whether to print debug spews
        """
        self.handle = cuml.common.handle.Handle() if handle is None else handle
        self.verbose = verbose


    def get_param_names(self):
        """
        Returns a list of hyperparameter names owned by this class. It is
        expected that every child class overrides this method and appends its
        extra set of parameters that it in-turn owns. This is to simplify the
        implementation of `get_params` and `set_params` methods.
        """
        return []


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

    def _get_dev_array_ptr(self, obj):
        """
        Get ctype pointer of a numba style device array
        """
        return obj.device_ctypes_pointer.value

    def _get_cudf_column_ptr(self, col):
        """
        Get ctype pointer of a cudf column
        """
        return cudf.bindings.cudf_cpp.get_column_data_ptr(col._column)
