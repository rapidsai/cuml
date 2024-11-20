#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

import numpy as np

from cuml.internals.array import CumlArray
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.mixins import FMajorInputTagMixin
from cuml.internals.base import UniversalBase, DynamicDescriptor


def io_fit(func):
    def wrapper(self, *args, **kwargs):
        # increase global counter to detect we are internal
        GlobalSettings().increase_arc()

        # check input type of first arg and fit estimator
        self._set_output_type(args[0])
        result = func(self, *args, **kwargs) 
        self._is_fit = True

        # decrease counter after exiting function
        GlobalSettings().decrease_arc()

        return result

    return wrapper


def io_predict_transform_array(func):
    def wrapper(self, *args, **kwargs):
        # increase global counter to detect we are internal
        GlobalSettings().increase_arc()

        result = func(self, *args, **kwargs) 

        # decrease counter after exiting function
        GlobalSettings().decrease_arc()

        if GlobalSettings().is_internal:
            return result

        else:
            # need to add logic to check globalsettings and mirror output_type
            return result.to_output(self._input_type) 

        return result

    return wrapper


class Estimator(UniversalBase,
                FMajorInputTagMixin):
    coef_ = DynamicDescriptor("coef_")
    intercept_ = DynamicDescriptor("intercept_")

    def __init__(self, 
                 *, 
                 awesome=True,
                 output_type=None,
                 handle=None,
                 verbose=None):

        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        self.awesome = awesome
        self._is_fit = False  # this goes in base

    @io_fit
    def fit(self,
            X,
            y,
            convert_dtype=True):

        input_X = CumlArray.from_input(
            X,
            order="C",
            convert_dtype=convert_dtype,
            target_dtype=np.float32,
            check_dtype=[np.float32, np.float64],
        )
        self.n_features_in_ = input_X.n_cols
        self.dtype = input_X.dtype

        input_y = CumlArray.from_input(
            y,
            order="C",
            convert_dtype=convert_dtype,
            target_dtype=self.dtype,
            check_dtype=[np.float32, np.float64],
        )

        self.coef_ = CumlArray.zeros(self.n_features_in_, 
                                     dtype=self.dtype)

        self.intercept_ = CumlArray.zeros(self.n_features_in_, 
                                     dtype=self.dtype)

        # do awesome C++ fitting here :) 

        return self

    @io_predict_transform_array
    def predict(self, 
                X,
                convert_dtype=True):
        input_X = CumlArray.from_input(
            X,
            order="C",
            convert_dtype=convert_dtype,
            target_dtype=self.dtype,
            check_dtype=[np.float32, np.float64],
        )
        n_rows = input_X.shape[0]

        preds = CumlArray.zeros(n_rows, 
                                dtype=self.dtype, 
                                index=input_X.index)

        # more awesome C++ 

        return preds