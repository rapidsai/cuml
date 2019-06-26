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

import cudf
import cuml
import ctypes
import numpy as np
import pandas as pd

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.memory cimport shared_ptr

cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "umap/umapparams.h" namespace "ML::UMAPParams":

    enum MetricType:
        EUCLIDEAN = 0,
        CATEGORICAL = 1

cdef extern from "umap/umapparams.h" namespace "ML":

    cdef cppclass UMAPParams:
        int n_neighbors,
        int n_components,
        int n_epochs,
        float learning_rate,
        float min_dist,
        float spread,
        int init,
        float set_op_mix_ratio,
        float local_connectivity,
        float repulsion_strength,
        int negative_sample_rate,
        float transform_queue_size,
        bool verbose,
        float a,
        float b,
        int target_n_neighbors,
        float target_weights,
        MetricType target_metric


cdef extern from "umap/umap.hpp" namespace "ML":
    void fit(cumlHandle & handle,
             float * X,
             int n,
             int d,
             UMAPParams * params,
             float * embeddings) except +

    void fit(cumlHandle & handle,
             float * X,
             float * y,
             int n,
             int d,
             UMAPParams * params,
             float * embeddings) except +

    void transform(cumlHandle & handle,
                   float * X,
                   int n,
                   int d,
                   float * orig_X,
                   int orig_n,
                   float * embedding,
                   int embedding_n,
                   UMAPParams * params,
                   float * out) except +


class UMAP(Base):
    """Uniform Manifold Approximation and Projection
    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.

    Adapted from https://github.com/lmcinnes/umap/blob/master/umap/umap_.py

    Parameters
    ----------
    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.
    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    learning_rate: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.
    init: string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.
    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with
        ``min_dist`` this determines how clustered/clumped the embedded
        points are.
    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    repulsion_strength: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    transform_queue_size: float (optional, default 4.0)
        For transform operations (embedding new points using a trained model_
        this will control how aggressively to search for nearest neighbors.
        Larger values will result in slower performance but more accurate
        nearest neighbor evaluation.
    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    verbose: bool (optional, default False)
        Controls verbosity of logging.

    Notes
    -----
    This module is heavily based on Leland McInnes' reference UMAP package.
    However, there are a number of differences and features that are not yet
    implemented in cuml.umap:
      * Specifying the random seed
      * Using a non-euclidean distance metric (support for a fixed set
        of non-euclidean metrics is planned for an upcoming release).
      * Using a pre-computed pairwise distance matrix (under consideration
        for future releases)
      * Manual initialization of initial embedding positions

    In addition to these missing features, you should expect to see
    the final embeddings differing between cuml.umap and the reference
    UMAP. In particular, the reference UMAP uses an approximate kNN
    algorithm for large data sizes while cuml.umap always uses exact
    kNN.

    References
    ----------
    * Leland McInnes, John Healy, James Melville
      UMAP: Uniform Manifold Approximation and Projection for Dimension
      Reduction
      https://arxiv.org/abs/1802.03426

    """

    def __init__(self,
                 n_neighbors=15,
                 n_components=2,
                 n_epochs=500,
                 learning_rate=1.0,
                 min_dist=0.1,
                 spread=1.0,
                 set_op_mix_ratio=1.0,
                 local_connectivity=1.0,
                 repulsion_strength=1.0,
                 negative_sample_rate=5,
                 transform_queue_size=4.0,
                 init="spectral",
                 verbose=False,
                 a=None,
                 b=None,
                 target_n_neighbors=-1,
                 target_weights=0.5,
                 target_metric="euclidean",
                 should_downcast=True,
                 handle=None):

        super(UMAP, self).__init__(handle, verbose)

        cdef UMAPParams * umap_params = new UMAPParams()

        self.n_neighbors = n_neighbors
        umap_params.n_neighbors = n_neighbors

        umap_params.n_components = <int > n_components
        umap_params.n_epochs = <int > n_epochs
        umap_params.verbose = <bool > verbose

        if(init == "spectral"):
            umap_params.init = <int > 1
        elif(init == "random"):
            umap_params.init = <int > 0
        else:
            raise Exception("Initialization strategy not supported: %d" % init)

        if a is not None:
            umap_params.a = <float > a

        if b is not None:
            umap_params.b = <float > b

        umap_params.learning_rate = <float > learning_rate
        umap_params.min_dist = <float > min_dist
        umap_params.spread = <float > spread
        umap_params.set_op_mix_ratio = <float > set_op_mix_ratio
        umap_params.local_connectivity = <float > local_connectivity
        umap_params.repulsion_strength = <float > repulsion_strength
        umap_params.negative_sample_rate = <int > negative_sample_rate
        umap_params.transform_queue_size = <int > transform_queue_size

        umap_params.target_n_neighbors = target_n_neighbors
        umap_params.target_weights = target_weights

        if target_metric == "euclidean":
            umap_params.target_metric = MetricType.EUCLIDEAN
        elif target_metric == "categorical":
            umap_params.target_metric = MetricType.CATEGORICAL
        else:
            raise Exception("Invalid target metric: {}" % target_metric)

        self._should_downcast = should_downcast

        self.umap_params = <size_t > umap_params

    def __getstate__(self):
        state = self.__dict__.copy()

        del state['handle']

        cdef size_t params_t = <size_t>self.umap_params
        cdef UMAPParams* umap_params = <UMAPParams*>params_t

        state['X_m'] = cudf.DataFrame.from_gpu_matrix(self.X_m)
        state['arr_embed'] = cudf.DataFrame.from_gpu_matrix(self.arr_embed)
        state["n_neighbors"] = umap_params.n_neighbors
        state["n_components"] = umap_params.n_components
        state["n_epochs"] = umap_params.n_epochs
        state["learning_rate"] = umap_params.learning_rate
        state["min_dist"] = umap_params.min_dist
        state["spread"] = umap_params.spread
        state["set_op_mix_ratio"] = umap_params.set_op_mix_ratio
        state["local_connectivity"] = umap_params.local_connectivity
        state["repulsion_strength"] = umap_params.repulsion_strength
        state["negative_sample_rate"] = umap_params.negative_sample_rate
        state["transform_queue_size"] = umap_params.transform_queue_size
        state["init"] = umap_params.init
        state["a"] = umap_params.a
        state["b"] = umap_params.b
        state["target_n_neighbors"] = umap_params.target_n_neighbors
        state["target_weights"] = umap_params.target_weights
        state["target_metric"] = umap_params.target_metric

        del state["umap_params"]

        return state

    def __del__(self):
        cdef UMAPParams* umap_params = <UMAPParams*><size_t>self.umap_params
        free(umap_params)

    def __setstate__(self, state):
        super(UMAP, self).__init__(handle=None, verbose=state['verbose'])

        state['X_m'] = state['X_m'].as_gpu_matrix(order="C")
        state["arr_embed"] = state["arr_embed"].as_gpu_matrix(order="C")

        cdef UMAPParams *umap_params = new UMAPParams()

        self.X_m = None
        umap_params.n_neighbors = state["n_neighbors"]
        umap_params.n_components = state["n_components"]
        umap_params.n_epochs = state["n_epochs"]
        umap_params.learning_rate = state["learning_rate"]
        umap_params.min_dist = state["min_dist"]
        umap_params.spread = state["spread"]
        umap_params.set_op_mix_ratio = state["set_op_mix_ratio"]
        umap_params.local_connectivity = state["local_connectivity"]
        umap_params.repulsion_strength = state["repulsion_strength"]
        umap_params.negative_sample_rate = state["negative_sample_rate"]
        umap_params.transform_queue_size = state["transform_queue_size"]
        umap_params.init = state["init"]
        umap_params.a = state["a"]
        umap_params.b = state["b"]
        umap_params.target_n_neighbors = state["target_n_neighbors"]
        umap_params.target_weights = state["target_weights"]
        umap_params.target_metric = state["target_metric"]

        state["umap_params"] = <size_t>umap_params

        self.__dict__.update(state)

    def fit(self, X, y=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            X contains a sample per row.
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        y : array-like (device or host) shape = (n_samples, 1)
            y contains a label per row.
            Acceptable formats: cuDF Series, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        """
        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        if self._should_downcast:
            self.X_m, X_ctype, n_rows, n_cols, dtype = \
                input_to_dev_array(X, order='C', convert_to_dtype=np.float32)
        else:
            self.X_m, X_ctype, n_rows, n_cols, dtype = \
                input_to_dev_array(X, order='C', check_dtype=np.float32)

        if n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample to "
                             "build nearest the neighbors graph")

        cdef UMAPParams * umap_params = \
            <UMAPParams*> < size_t > self.umap_params
        umap_params.n_neighbors = min(n_rows, umap_params.n_neighbors)
        self.n_dims = n_cols
        self.raw_data = X_ctype
        self.raw_data_rows = n_rows

        self.arr_embed = cuda.to_device(zeros((self.X_m.shape[0],
                                               umap_params.n_components),
                                              order="C", dtype=np.float32))
        self.embeddings = \
            self.arr_embed.device_ctypes_pointer.value

        cdef cumlHandle * handle_ = \
            <cumlHandle*> < size_t > self.handle.getHandle()

        cdef uintptr_t y_raw
        cdef uintptr_t x_raw = X_ctype

        cdef uintptr_t embed_raw = self.embeddings

        if y is not None:
            y_m, y_raw, _, _, _ = \
                input_to_dev_array(y)
            fit(handle_[0],
                < float*> x_raw,
                < float*> y_raw,
                < int > self.X_m.shape[0],
                < int > self.X_m.shape[1],
                < UMAPParams*>umap_params,
                < float*>embed_raw)

        else:

            fit(handle_[0],
                < float*> x_raw,
                < int > self.X_m.shape[0],
                < int > self.X_m.shape[1],
                < UMAPParams*>umap_params,
                < float*>embed_raw)

        return self

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            X contains a sample per row.
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self.fit(X, y)
        if isinstance(X, cudf.DataFrame):
            ret = cudf.DataFrame()
            for i in range(0, self.arr_embed.shape[1]):
                ret[str(i)] = self.arr_embed[:, i]
        elif isinstance(X, np.ndarray):
            ret = np.asarray(self.arr_embed)

        return ret

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.

        Please refer to the reference UMAP implementation for information
        on the differences between fit_transform() and running fit()
        transform().

        Specifically, the transform() function is stochastic:
        https://github.com/lmcinnes/umap/issues/158

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            New data to be transformed.
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        cdef uintptr_t x_ptr
        if self._should_downcast:
            X_m, x_ptr, n_rows, n_cols, dtype = \
                input_to_dev_array(X, order='C', convert_to_dtype=np.float32)
        else:
            X_m, x_ptr, n_rows, n_cols, dtype = \
                input_to_dev_array(X, order='C', check_dtype=np.float32)

        if n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample to "
                             "build nearest the neighbors graph")

        if n_cols != self.n_dims:
            raise ValueError("n_features of X must match n_features of "
                             "training data")

        cdef UMAPParams * umap_params = \
            <UMAPParams*> < size_t > self.umap_params
        embedding = cuda.to_device(zeros((X_m.shape[0],
                                          umap_params.n_components),
                                         order="C", dtype=np.float32))
        cdef uintptr_t xformed_ptr = embedding.device_ctypes_pointer.value

        cdef cumlHandle * handle_ = \
            <cumlHandle*> < size_t > self.handle.getHandle()

        cdef uintptr_t orig_x_raw = self.raw_data

        cdef uintptr_t embed_ptr = self.embeddings

        transform(handle_[0],
                  < float*>x_ptr,
                  < int > X_m.shape[0],
                  < int > X_m.shape[1],
                  < float*>orig_x_raw,
                  < int > self.raw_data_rows,
                  < float*> embed_ptr,
                  < int > self.arr_embed.shape[0],
                  < UMAPParams*> umap_params,
                  < float*> xformed_ptr)

        if isinstance(X, cudf.DataFrame):
            ret = cudf.DataFrame()
            for i in range(0, embedding.shape[1]):
                ret[str(i)] = embedding[:, i]
        elif isinstance(X, np.ndarray):
            ret = np.asarray(embedding)

        del X_m

        return ret
