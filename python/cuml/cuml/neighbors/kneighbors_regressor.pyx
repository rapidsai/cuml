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

import numpy as np

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.api_decorators import api_base_return_array
from cuml.internals.array import CumlArray
from cuml.internals.interop import UnsupportedOnGPU, to_cpu, to_gpu
from cuml.internals.mixins import FMajorInputTagMixin, RegressorMixin
from cuml.neighbors.nearest_neighbors import NearestNeighbors

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uintptr_t
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/neighbors/knn.hpp" namespace "ML" nogil:

    void knn_regress(
        handle_t &handle,
        float *out,
        int64_t *knn_indices,
        vector[float *] &y,
        size_t n_rows,
        size_t n_samples,
        int k,
    ) except +


class KNeighborsRegressor(RegressorMixin,
                          FMajorInputTagMixin,
                          NearestNeighbors):
    """

    K-Nearest Neighbors Regressor is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.

    The K-Nearest Neighbors Regressor will compute the average of the
    labels for the k closest neighbors and use it as the label.

    Parameters
    ----------
    n_neighbors : int (default=5)
        Default number of neighbors to query
    algorithm : string (default='auto')
        The query algorithm to use. Valid options are:

        - ``'auto'``: to automatically select brute-force or
          random ball cover based on data shape and metric
        - ``'rbc'``: for the random ball algorithm, which partitions
          the data space and uses the triangle inequality to lower the
          number of potential distances. Currently, this algorithm
          supports 2d Euclidean and Haversine.
        - ``'brute'``: for brute-force, slow but produces exact results
        - ``'ivfflat'``: for inverted file, divide the dataset in partitions
          and perform search on relevant partitions only
        - ``'ivfpq'``: for inverted file and product quantization,
          same as inverted list, in addition the vectors are broken
          in n_features/M sub-vectors that will be encoded thanks
          to intermediary k-means clusterings. This encoding provide
          partial information allowing faster distances calculations
    metric : string (default='euclidean').
        Distance metric to use.
    weights : string (default='uniform')
        Sample weights to use. Currently, only the uniform strategy is
        supported.
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

        >>> from cuml.neighbors import KNeighborsRegressor
        >>> from cuml.datasets import make_regression
        >>> from cuml.model_selection import train_test_split

        >>> X, y = make_regression(n_samples=100, n_features=10,
        ...                        random_state=5)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...   X, y, train_size=0.80, random_state=5)

        >>> knn = KNeighborsRegressor(n_neighbors=10)
        >>> knn.fit(X_train, y_train)
        KNeighborsRegressor()
        >>> knn.predict(X_test) # doctest: +SKIP
        array([ 14.770798  ,  51.8834    ,  66.15657   ,  46.978275  ,
            21.589611  , -14.519918  , -60.25534   , -20.856869  ,
            29.869623  , -34.83317   ,   0.45447388, 120.39675   ,
            109.94834   ,  63.57794   , -17.956171  ,  78.77663   ,
            30.412262  ,  32.575233  ,  74.72834   , 122.276855  ],
        dtype=float32)

    Notes
    -----

    For additional docs, see `scikitlearn's KNeighborsClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_.
    """

    y = CumlArrayDescriptor(order="F")

    _cpu_class_path = "sklearn.neighbors.KNeighborsRegressor"

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "weights"]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.weights != "uniform":
            raise UnsupportedOnGPU

        return {
            "weights": "uniform",
            **super()._params_from_cpu(model),
        }

    def _params_to_cpu(self):
        return {
            "weights": self.weights,
            **super()._params_to_cpu(),
        }

    def _attrs_from_cpu(self, model):
        return {
            "y": to_gpu(model._y, order="F", dtype=np.float32),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "_y": to_cpu(self.y),
            **super()._attrs_to_cpu(model),
        }

    def __init__(self, *, weights="uniform", handle=None, verbose=False,
                 output_type=None, **kwargs):
        super().__init__(
            handle=handle,
            verbose=verbose,
            output_type=output_type,
            **kwargs)
        self.y = None
        self.weights = weights

    @generate_docstring(convert_dtype_cast='np.float32')
    def fit(self, X, y, *, convert_dtype=True) -> "KNeighborsRegressor":
        """
        Fit a GPU index for k-nearest neighbors regression model.

        """
        if self.weights != "uniform":
            raise ValueError("Only uniform weighting strategy "
                             "is supported currently.")
        self._set_target_dtype(y)

        super(KNeighborsRegressor, self).fit(X, convert_dtype=convert_dtype)
        self.y, _, _, _ = \
            input_to_cuml_array(y, order='F', check_dtype=np.float32,
                                convert_to_dtype=(np.float32
                                                  if convert_dtype
                                                  else None))
        return self

    @generate_docstring(convert_dtype_cast='np.float32',
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, n_features)'})
    @api_base_return_array(get_output_dtype=True)
    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Use the trained k-nearest neighbors regression model to
        predict the labels for X

        """

        knn_indices = self.kneighbors(X, return_distance=False,
                                      convert_dtype=convert_dtype)

        inds, n_rows, _n_cols, _dtype = \
            input_to_cuml_array(knn_indices, order='C', check_dtype=np.int64,
                                convert_to_dtype=(np.int64
                                                  if convert_dtype
                                                  else None))
        cdef uintptr_t inds_ctype = inds.ptr

        res_cols = 1 if len(self.y.shape) == 1 else self.y.shape[1]
        res_shape = n_rows if res_cols == 1 else (n_rows, res_cols)
        results = CumlArray.zeros(res_shape, dtype=np.float32,
                                  order="C",
                                  index=inds.index)

        cdef uintptr_t results_ptr = results.ptr
        cdef uintptr_t y_ptr
        cdef vector[float*] *y_vec = new vector[float*]()

        for col_num in range(res_cols):
            col = self.y if res_cols == 1 else self.y[:, col_num]
            y_ptr = col.ptr
            y_vec.push_back(<float*>y_ptr)

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        knn_regress(
            handle_[0],
            <float*>results_ptr,
            <int64_t*>inds_ctype,
            deref(y_vec),
            <size_t>self.n_samples_fit_,
            <size_t>n_rows,
            <int>self.n_neighbors
        )

        self.handle.sync()
        return results
