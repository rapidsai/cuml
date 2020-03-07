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
import warnings

import joblib

import cupy

import numba.cuda as cuda

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros, row_matrix

import rmm

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdint cimport uint64_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.memory cimport shared_ptr

cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "cuml/manifold/umapparams.h" namespace "ML::UMAPParams":

    enum MetricType:
        EUCLIDEAN = 0,
        CATEGORICAL = 1

cdef extern from "internals/internals.h" namespace "ML::Internals":

    cdef cppclass GraphBasedDimRedCallback

cdef extern from "cuml/manifold/umapparams.h" namespace "ML":

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
        MetricType target_metric,
        uint64_t random_state,
        bool multicore_implem,
        GraphBasedDimRedCallback* callback


cdef extern from "cuml/manifold/umap.hpp" namespace "ML":
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

    void find_ab(cumlHandle &handle,
                 UMAPParams *params) except +


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
    hash_input: UMAP can hash the training input so that exact embeddings
                are returned when transform is called on the same data upon
                which the model was trained. This enables consistent
                behavior between calling model.fit_transform(X) and
                calling model.fit(X).transform(X). Not that the CPU-based
                UMAP reference implementation does this by default. This
                feature is made optional in the GPU version due to the
                significant overhead in copying memory to the host for
                computing the hash. (default = False)
    random_state : int, RandomState instance or None, optional (default=None)
        random_state is the seed used by the random number generator during
        embedding initialization and during sampling used by the optimizer.
        Note: Unfortunately, achieving a high amount of parallelism during
        the optimization stage often comes at the expense of determinism,
        since many floating-point additions are being made in parallel
        without a deterministic ordering. This causes slightly different
        results across training sessions, even when the same seed is used
        for random number generation. Setting a random_state will enable
        consistency of trained embeddings, allowing for reproducible results
        to 3 digits of precision, but will do so at the expense of potentially
        slower training and increased memory usage.
    callback: An instance of GraphBasedDimRedCallback class to intercept
              the internal state of embeddings while they are being trained.
              Example of callback usage:
                  from cuml.internals import GraphBasedDimRedCallback
                  class CustomCallback(GraphBasedDimRedCallback):
                    def on_preprocess_end(self, embeddings):
                        print(embeddings.copy_to_host())

                    def on_epoch_end(self, embeddings):
                        print(embeddings.copy_to_host())

                    def on_train_end(self, embeddings):
                        print(embeddings.copy_to_host())
    verbose: bool (optional, default False)
        Controls verbosity of logging.

    Notes
    -----
    This module is heavily based on Leland McInnes' reference UMAP package.
    However, there are a number of differences and features that are not yet
    implemented in cuml.umap:
      * Specifying the random seed
      * Using a non-Euclidean distance metric (support for a fixed set
        of non-Euclidean metrics is planned for an upcoming release).
      * Using a pre-computed pairwise distance matrix (under consideration
        for future releases)
      * Manual initialization of initial embedding positions

    In addition to these missing features, you should expect to see
    the final embeddings differing between cuml.umap and the reference
    UMAP. In particular, the reference UMAP uses an approximate kNN
    algorithm for large data sizes while cuml.umap always uses exact
    kNN.

    Known issue: If a UMAP model has not yet been fit, it cannot be pickled.
    However, after fitting, a UMAP mode.

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
                 n_epochs=0,
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
                 target_metric="categorical",
                 handle=None,
                 hash_input=False,
                 random_state=None,
                 callback=None):

        super(UMAP, self).__init__(handle, verbose)

        cdef UMAPParams * umap_params = new UMAPParams()

        self.hash_input = hash_input

        self.n_neighbors = n_neighbors
        umap_params.n_neighbors = n_neighbors

        umap_params.n_components = <int> n_components
        umap_params.n_epochs = <int> n_epochs
        umap_params.verbose = <bool> verbose

        if(init == "spectral"):
            umap_params.init = <int> 1
        elif(init == "random"):
            umap_params.init = <int> 0
        else:
            raise Exception("Initialization strategy not supported: %d" % init)

        if a is not None:
            umap_params.a = <float> a
        if b is not None:
            umap_params.b = <float> b

        umap_params.learning_rate = <float> learning_rate
        umap_params.min_dist = <float> min_dist
        umap_params.spread = <float> spread
        umap_params.set_op_mix_ratio = <float> set_op_mix_ratio
        umap_params.local_connectivity = <float> local_connectivity
        umap_params.repulsion_strength = <float> repulsion_strength
        umap_params.negative_sample_rate = <int> negative_sample_rate
        umap_params.transform_queue_size = <int> transform_queue_size

        umap_params.target_n_neighbors = target_n_neighbors
        umap_params.target_weights = target_weights

        umap_params.multicore_implem = random_state is None
        if isinstance(random_state, np.random.RandomState):
            rs = random_state
        else:
            rs = np.random.RandomState(random_state)
        umap_params.random_state = long(rs.randint(low=0,
                                        high=np.iinfo(np.uint64).max,
                                        dtype=np.uint64))

        if target_metric == "euclidean":
            umap_params.target_metric = MetricType.EUCLIDEAN
        elif target_metric == "categorical":
            umap_params.target_metric = MetricType.CATEGORICAL
        else:
            raise Exception("Invalid target metric: {}" % target_metric)

        cdef uintptr_t callback_ptr = 0
        if callback:
            callback_ptr = callback.get_native_callback()
            umap_params.callback = <GraphBasedDimRedCallback*>callback_ptr

        cdef cumlHandle * handle_ = \
            <cumlHandle*> <size_t> self.handle.getHandle()

        if a is None or b is None:
            find_ab(handle_[0], umap_params)

        self.umap_params = <size_t> umap_params

        self.callback = callback  # prevent callback destruction
        self.X_m = None
        self.embedding_ = None

    def __getstate__(self):
        state = self.__dict__.copy()

        del state['handle']

        cdef size_t params_t = <size_t>self.umap_params
        cdef UMAPParams* umap_params = <UMAPParams*>params_t

        if hasattr(self, "X_m") and self.X_m is not None:
            # fit has not yet been called
            state['X_m'] = cudf.DataFrame.from_gpu_matrix(self.X_m)
            state['embedding_'] = \
                cudf.DataFrame.from_gpu_matrix(self.embedding_)

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
        cdef UMAPParams * umap_params
        if hasattr(self, 'umap_params'):
            umap_params = <UMAPParams*><size_t>self.umap_params
            free(umap_params)

    def __setstate__(self, state):
        super(UMAP, self).__init__(handle=None, verbose=state['verbose'])

        if "X_m" in state and state["X_m"] is not None:
            # fit has not yet been called
            state["X_m"] = row_matrix(state["X_m"])
            state["embedding_"] = row_matrix(state["embedding_"])

        cdef UMAPParams *umap_params = new UMAPParams()

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

    @staticmethod
    def _prep_output(X, embedding):
        if isinstance(X, cudf.DataFrame):
            return cudf.DataFrame.from_gpu_matrix(embedding)
        elif isinstance(X, np.ndarray):
            return np.asarray(embedding)
        elif isinstance(X, cuda.is_cuda_array(X)):
            return embedding
        elif isinstance(X, cupy.ndarray):
            return cupy.array(embedding)

    def fit(self, X, y=None, convert_dtype=True):
        """
        Fit X into an embedded space.

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

        self.X_m, X_ctype, self.n_rows, self.n_dims, dtype = \
            input_to_dev_array(X, order='C', check_dtype=np.float32,
                               convert_to_dtype=(np.float32 if convert_dtype
                                                 else None))

        if self.n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample to "
                             "build nearest the neighbors graph")

        cdef UMAPParams * umap_params = \
            <UMAPParams*> <size_t> self.umap_params
        umap_params.n_neighbors = min(self.n_rows, umap_params.n_neighbors)

        self.embedding_ = rmm.to_device(zeros((self.n_rows,
                                              umap_params.n_components),
                                              order="C", dtype=np.float32))

        if self.hash_input:
            self.input_hash = joblib.hash(self.X_m.copy_to_host())

        embeddings = \
            self.embedding_.device_ctypes_pointer.value

        cdef cumlHandle * handle_ = \
            <cumlHandle*> <size_t> self.handle.getHandle()

        cdef uintptr_t y_raw
        cdef uintptr_t x_raw = X_ctype

        cdef uintptr_t embed_raw = embeddings

        if y is not None:
            y_m, y_raw, _, _, _ = \
                input_to_dev_array(y, check_dtype=np.float32,
                                   convert_to_dtype=(np.float32
                                                     if convert_dtype
                                                     else None))
            fit(handle_[0],
                <float*> x_raw,
                <float*> y_raw,
                <int> self.n_rows,
                <int> self.n_dims,
                <UMAPParams*>umap_params,
                <float*>embed_raw)

        else:

            fit(handle_[0],
                <float*> x_raw,
                <int> self.n_rows,
                <int> self.n_dims,
                <UMAPParams*>umap_params,
                <float*>embed_raw)

        self.handle.sync()

        return self

    def fit_transform(self, X, y=None, convert_dtype=True):
        """
        Fit X into an embedded space and return that transformed
        output.

        There is a subtle difference between calling fit_transform(X)
        and calling fit().transform(). Calling fit_transform(X) will
        train the embeddings on X and return the embeddings. Calling
        fit(X).transform(X) will train the embeddings on X and then
        run a second optimization
        return the embedding after it is trained while calling

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
        self.fit(X, y, convert_dtype=convert_dtype)
        return UMAP._prep_output(X, self.embedding_)

    def transform(self, X, convert_dtype=True):
        """
        Transform X into the existing embedded space and return that
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
        X_m, x_ptr, n_rows, n_cols, dtype = \
            input_to_dev_array(X, order='C', check_dtype=np.float32,
                               convert_to_dtype=(np.float32 if convert_dtype
                                                 else None))

        if n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample to "
                             "build nearest the neighbors graph")

        if n_cols != self.n_dims:
            raise ValueError("n_features of X must match n_features of "
                             "training data")

        if self.hash_input and joblib.hash(X_m.copy_to_host()) == \
                self.input_hash:
            ret = UMAP._prep_output(X, self.embedding_)
            del X_m
            return ret

        cdef UMAPParams * umap_params = \
            <UMAPParams*> <size_t> self.umap_params
        embedding = rmm.to_device(zeros((X_m.shape[0],
                                         umap_params.n_components),
                                        order="C", dtype=np.float32))
        cdef uintptr_t xformed_ptr = embedding.device_ctypes_pointer.value

        cdef cumlHandle *handle_ = \
            <cumlHandle*> <size_t> self.handle.getHandle()

        cdef uintptr_t orig_x_raw = self.X_m.device_ctypes_pointer.value

        cdef uintptr_t embed_ptr = self.embedding_.device_ctypes_pointer.value

        transform(handle_[0],
                  <float*>x_ptr,
                  <int> X_m.shape[0],
                  <int> X_m.shape[1],
                  <float*>orig_x_raw,
                  <int> self.n_rows,
                  <float*> embed_ptr,
                  <int> self.n_rows,
                  <UMAPParams*> umap_params,
                  <float*> xformed_ptr)
        self.handle.sync()

        ret = UMAP._prep_output(X, embedding)
        del X_m
        return ret
