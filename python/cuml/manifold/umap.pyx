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
    input_to_array

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


cdef extern from "umap/umap.h" namespace "ML":
    cdef cppclass UMAP_API:

        UMAP_API(UMAPParams *p) except +

        void fit(cumlHandle &handle,
                 float *X,
                 int n,
                 int d,
                 float *embeddings)

        void fit(cumlHandle &handle,
                 float *X,
                 float *y,
                 int n,
                 int d,
                 float *embeddings)

        void transform(cumlHandle &handle,
                       float *X,
                       int n,
                       int d,
                       float *embedding,
                       int embedding_n,
                       float *out)


cdef class UMAPImpl:
    cpdef UMAPParams *umap_params
    cpdef UMAP_API *umap
    cdef uintptr_t embeddings
    cdef uintptr_t raw_data

    cpdef object n_neighbors
    cpdef object arr_embed
    cpdef object n_dims

    cdef bool _should_downcast

    cdef object handle

    def __cinit__(self,
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

        self.handle = handle

        self.umap_params = new UMAPParams()

        self.n_neighbors = n_neighbors
        self.umap_params.n_neighbors = n_neighbors

        self.umap_params.n_components = <int>n_components
        self.umap_params.n_epochs = <int>n_epochs
        self.umap_params.verbose = <bool>verbose

        if(init == "spectral"):
            self.umap_params.init = <int>1
        elif(init == "random"):
            self.umap_params.init = <int>0
        else:
            raise Exception("Initialization strategy not supported: %d" % init)

        if a is not None:
            self.umap_params.a = <float>a

        if b is not None:
            self.umap_params.b = <float>b

        self.umap_params.learning_rate = <float>learning_rate
        self.umap_params.min_dist = <float>min_dist
        self.umap_params.spread = <float>spread
        self.umap_params.set_op_mix_ratio = <float>set_op_mix_ratio
        self.umap_params.local_connectivity = <float>local_connectivity
        self.umap_params.repulsion_strength = <float>repulsion_strength
        self.umap_params.negative_sample_rate = <int>negative_sample_rate
        self.umap_params.transform_queue_size = <int>transform_queue_size

        self.umap_params.target_n_neighbors = target_n_neighbors
        self.umap_params.target_weights = target_weights

        if target_metric == "euclidean":
            self.umap_params.target_metric = MetricType.EUCLIDEAN
        elif target_metric == "categorical":
            self.umap_params.target_metric = MetricType.CATEGORICAL
        else:
            raise Exception("Invalid target metric: {}" % target_metric)

        self._should_downcast = should_downcast

        self.umap = new UMAP_API(self.umap_params)

    def __dealloc__(self):
        del self.umap_params
        del self.umap

    def fit(self, X, y=None):

        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        if self._should_downcast:
            X_m, X_ctype, n_rows, n_cols, dtype = \
                input_to_array(X, order='C', convert_to_dtype=np.float32)
        else:
            X_m, X_ctype, n_rows, n_cols, dtype = \
                input_to_array(X, order='C', check_dtype=np.float32)

        if n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample to "
                             "build nearest the neighbors graph")

        self.umap_params.n_neighbors = min(n_rows,
                                           self.umap_params.n_neighbors)
        self.n_dims = n_cols
        self.raw_data = X_ctype

        self.arr_embed = cuda.to_device(np.zeros((X_m.shape[0],
                                        self.umap_params.n_components),
                                        order="C", dtype=np.float32))
        self.embeddings = self.arr_embed.device_ctypes_pointer.value

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t y_raw
        if y is not None:
            y_m, y_raw, _, _, _ = \
                input_to_array(y)
            self.umap.fit(
                handle_[0],
                <float*> self.raw_data,
                <float*> y_raw,
                <int> X_m.shape[0],
                <int> X_m.shape[1],
                <float*>self.embeddings
            )

        else:

            self.umap.fit(
                handle_[0],
                <float*> self.raw_data,
                <int> X_m.shape[0],
                <int> X_m.shape[1],
                <float*>self.embeddings
            )

        del X_m

    def fit_transform(self, X, y=None):
        self.fit(X, y)

        if isinstance(X, cudf.DataFrame):
            ret = cudf.DataFrame()
            for i in range(0, self.arr_embed.shape[1]):
                ret[str(i)] = self.arr_embed[:, i]
        elif isinstance(X, np.ndarray):
            ret = np.asarray(self.arr_embed)

        return ret

    def transform(self, X):
        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        cdef uintptr_t x_ptr
        if self._should_downcast:
            X_m, x_ptr, n_rows, n_cols, dtype = \
                input_to_array(X, order='C', convert_to_dtype=np.float32)
        else:
            X_m, x_ptr, n_rows, n_cols, dtype = \
                input_to_array(X, order='C', check_dtype=np.float32)

        if n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample to "
                             "build nearest the neighbors graph")

        if n_cols != self.n_dims:
            raise ValueError("n_features of X must match n_features of "
                             "training data")

        embedding = cuda.to_device(np.zeros((X_m.shape[0],
                                             self.umap_params.n_components),
                                            order="C", dtype=np.float32))
        cdef uintptr_t embed_ptr = embedding.device_ctypes_pointer.value

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        self.umap.transform(handle_[0],
                            <float*>x_ptr,
                            <int>X_m.shape[0],
                            <int>X_m.shape[1],
                            <float*> self.embeddings,
                            <int> self.arr_embed.shape[0],
                            <float*> embed_ptr)

        if isinstance(X, cudf.DataFrame):
            ret = cudf.DataFrame()
            for i in range(0, embedding.shape[1]):
                ret[str(i)] = embedding[:, i]
        elif isinstance(X, np.ndarray):
            ret = np.asarray(embedding)

        del X_m

        return ret


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

        self._impl = UMAPImpl(n_neighbors,
                              n_components,
                              n_epochs,
                              learning_rate,
                              min_dist,
                              spread,
                              set_op_mix_ratio,
                              local_connectivity,
                              repulsion_strength,
                              negative_sample_rate,
                              transform_queue_size,
                              init,
                              verbose,
                              a, b,
                              target_n_neighbors,
                              target_weights,
                              target_metric,
                              should_downcast,
                              self.handle)

    def fit(self, X, y=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            X contains a sample per row.
        y : array, shape (n_samples)
            y contains a label per row.
        """

        return self._impl.fit(X, y)

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
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """

        return self._impl.transform(X)

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            X contains a sample per row.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """

        return self._impl.fit_transform(X, y)
