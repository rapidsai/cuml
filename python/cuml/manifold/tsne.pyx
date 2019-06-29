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

from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, input_to_dev_array, zeros
from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.memory cimport shared_ptr

cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "tsne/tsne.h" namespace "ML":
    void TSNE_fit(const cumlHandle &handle,
                const float *X, float *Y,
                const int n, const int p, const int dim, int n_neighbors,
                const float theta, const float epssq,
                float perplexity, const int perplexity_max_iter,
                const float perplexity_tol,
                const float early_exaggeration,
                const int exaggeration_iter, const float min_gain,
                const float pre_learning_rate, const float post_learning_rate,
                const int max_iter, const float min_grad_norm,
                const float pre_momentum, const float post_momentum,
                const long long random_state, const bool verbose,
                const bool intialize_embeddings, bool barnes_hut) except +


class TSNE:
    """
    TSNE (T-Distributed Stochastic Neighbor Embedding) is an extremely
    powerful dimensionality reduction technique that aims to maintain
    local distances between data points. It is extremely robust to whatever
    dataset you give it, and is used in many areas including cancer research,
    music analysis and neural network weight visualizations.

    cuML's TSNE implementation handles any # of n_components although specifying
    n_components = 2 will yield a somewhat extra speedup.

    Currently, TSNE only has a fit_transform method. For embedding new data, we
    are currently working on using weighted nearest neighborhood methods. This
    can also reduce the time complexity of TSNE's Naive O(n^2) to O(p * log(n)).

    A FFT based approach (pseudo-O(n)) is also in the works! We are also working on
    a (pseudo-O(p * log(n))) version using the Nystroem method to approximate the
    repulsion forces.

    Parameters
    ----------
    n_neighbors : float (optional, default 15)
        The number of datapoints you want to use in the
        attractive forces. Smaller values are better for preserving
        local structure, whilst larger values can improve global structure
        preservation.
    n_components: int (optional, default 2)
        The output dimensionality size. Can be any number, but with
        n_components = 2 TSNE can run faster.
    method : str (optional, default Fast)
        Options are [Fast, Naive]. Fast uses a parallel-O(n) algorithm.
        Naive is a pure O(n^2) algorithm where a n*n matrix is constructed
        for the repulsive forces. Fast uses O(N + NNZ) memory whilst Naive
        uses O(N^2 + NNZ) memory.
    epochs : int (optional, default 300)
        The more epochs, the more stable/accruate the final embedding.
    perplexity : (float, default 30)
        Larger datasets require a larger value. Consider choosing different
        perplexity values from 5 to 50 and see the output differences.
    perplexity_epochs : (int, default 100)
        The number of epochs the best guassian bands are found for.
    perplexity_tol : (float, default 1e-5)
        The tolerance during the best guassian band search.
    early_exaggeration : (int, default 12)
        Controls the space between clusters. Not critical to tune this.
    exaggeration_iter : (int, default 150)
        To promote the growth of clusters, set this higher.
    min_gain : (float, default 0.01)
        If the gradient updates are below this number, it is thresholded.
    learning_rate : (float, default 500)
        The learning rate usually between (10, 1000). If this is too high,
        TSNE could look like a cloud / ball of points.
    pre_momentum : (float, default 0.8)
        During the exaggeration iteration, more forcefully apply gradients.
    post_momentum : (float, default 0.5)
        During the late phases, less forcefully apply gradients.
    random_state : (int, default None)
        Setting this can allow future runs of TSNE to look the same.
    verbose : (bool, default False)
        Whether to print help messages.

    References
    ----------
    *   van der Maaten, L.J.P.
        t-Distributed Stochastic Neighbor Embedding
        https://lvdmaaten.github.io/tsne/

    *   van der Maaten, L.J.P.; Hinton, G.E. 
        Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
    
    *   George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, Yuval Kluger
        Efficient Algorithms for t-distributed Stochastic Neighborhood Embedding
    """

    def __init__(self,
                n_components = 2,
                perplexity = 30.0,
                early_exaggeration = 12.0,
                learning_rate = 200.0,
                n_iter = 1000,
                n_iter_without_progress = 300,
                min_grad_norm = 1e-07,
                metric = 'euclidean',
                init = 'random',
                verbose = 0,
                random_state = None,
                method = 'barnes_hut',
                angle = 0.5,

                n_neighbors = 90,
                perplexity_max_iter = 100,
                exaggeration_iter = 250,
                pre_momentum = 0.5,
                post_momentum = 0.8,
                should_downcast = True,
                handle = None):

        self.handle = handle

        if n_components < 0:
            print("[Error] n_components = {} should be more than 0.".format(n_components))
            n_components = 2
        if n_components != 2 and method == 'barnes_hut':
            print("[Warn] Barnes Hut only works when n_components == 2. Switching to exact.")
            method = 'exact'
        if perplexity < 0:
            print("[Error] perplexity = {} should be more than 0.".format(perplexity))
            perplexity = 30
        if early_exaggeration < 0:
            print("[Error] early_exaggeration = {} should be more than 0.".format(early_exaggeration))
            early_exaggeration = 12
        if learning_rate < 0:
            print("[Error] learning_rate = {} should be more than 0.".format(learning_rate))
            learning_rate = 200
        if n_iter < 0:
            print("[Error] n_iter = {} should be more than 0.".format(n_iter))
            n_iter = 1000
        if n_iter <= 100:
            print("[Warn] n_iter = {} might cause TSNE to output wrong results. Set it higher.".format(n_iter))
        if metric.lower() != 'euclidean':
            print("[Warn] TSNE does not support {} but only Euclidean. Will do in the near future.".format(metric))
            metric = 'euclidean'
        if init.lower() != 'random':
            print("[Warn] TSNE does not support {} but only random intialization. Will do in the near future.".format(init))
            init = 'random'
        if verbose != 0:
            verbose = 1
        if random_state is None:
            random_state = -1
        if angle < 0 or angle > 1:
            print("[Error] angle = {} should be more than 0 and less than 1.".format(angle))
            angle = 0.5
        if n_neighbors < 0:
            print("[Error] n_neighbors = {} should be more than 0.".format(n_neighbors))
            n_neighbors = perplexity * 3
        if n_neighbors > 1023:
            print("[Error] n_neighbors = {} should be less than 1023, as FAISS doesn't support more".format(n_neighbors))
            n_neighbors = 1023
        if perplexity_max_iter < 0:
            print("[Error] perplexity_max_iter = {} should be more than 0.".format(perplexity_max_iter))
            perplexity_max_iter = 100
        if exaggeration_iter < 0:
            print("[Error] exaggeration_iter = {} should be more than 0.".format(exaggeration_iter))
            exaggeration_iter = 250
        if exaggeration_iter > n_iter:
            print("[Error] exaggeration_iter = {} should be more less than n_iter = {}.".format(exaggeration_iter, n_iter))
            exaggeration_iter = max(int(n_iter * 0.25) , 1)
        if pre_momentum < 0 or pre_momentum > 1:
            print("[Error] pre_momentum = {} should be more than 0 and less than 1.".format(pre_momentum))
            pre_momentum = 0.5
        if post_momentum < 0 or post_momentum > 1:
            print("[Error] post_momentum = {} should be more than 0 and less than 1.".format(post_momentum))
            post_momentum = 0.8
        if pre_momentum > post_momentum:
            print("[Error] post_momentum = {} should be more than pre_momentum = {}".format(post_momentum, pre_momentum))
            pre_momentum = post_momentum * 0.75


        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric,
        self.init = init,
        self.verbose = verbose,
        self.random_state = random_state,
        self.method = 1 if method == 'barnes_hut' else 0
        self.angle = angle
        self.n_neighbors = n_neighbors
        self.perplexity_max_iter = perplexity_max_iter
        self.exaggeration_iter = exaggeration_iter
        self.pre_momentum = pre_momentum
        self.post_momentum = post_momentum

        self.epssq = 0.0025
        self.perplexity_tol = 1e-5
        self.min_gain = 0.01
        self.pre_learning_rate = learning_rate
        self.post_learning_rate = learning_rate * 2

        self._should_downcast = should_downcast
        return


    def fit(self, X):
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
        cdef int n, p

        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        if self._should_downcast:
            X_m, X_ctype, n, p, dtype = input_to_dev_array(X, order = 'C', convert_to_dtype = np.float32)
        else:
            X_m, X_ctype, n, p, dtype = input_to_dev_array(X, order = 'C', check_dtype = np.float32)

        if n <= 1:
            raise ValueError("There needs to be more than 1 sample to build nearest the neighbors graph")

        self.n_neighbors = min(n, self.n_neighbors)
        if self.perplexity > n:
            print("[Warn] Perplexity = {} should be less than the # of datapoints = {}.".format(self.perplexity, n))
            self.perplexity = n


        self.arr_embed = cuda.to_device( zeros((n, self.n_components), order = "F", dtype = np.float32) )
        self.embeddings = self.arr_embed.device_ctypes_pointer.value

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t X_ptr = X_ctype
        cdef uintptr_t embed_ptr = self.embeddings
        cdef uintptr_t y_raw

        TSNE_fit(handle_[0],
                <float*> X_ptr, <float*> embed_ptr,
                <int> n, <int> p, <int> self.n_components, <int> self.n_neighbors,
                <float> self.angle, <float> self.epssq,
                <float> self.perplexity, <int> self.perplexity_max_iter,
                <float> self.perplexity_tol,
                <float> self.early_exaggeration,
                <int> self.exaggeration_iter, <float> self.min_gain,
                <float> self.pre_learning_rate, <float> self.post_learning_rate,
                <int> self.n_iter, <float> self.min_grad_norm,
                <float> self.pre_momentum, <float> self.post_momentum,
                <long long> self.random_state, <bool> self.verbose,
                <bool> True, <bool> self.barnes_hut)
        del X_m
        return self


    def fit_transform(self, X):
        """Fit X into an embedded space and return that transformed output.
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
        self.fit(X)

        if isinstance(X, cudf.DataFrame):
            ret = cudf.DataFrame()
            for i in range(0, self.arr_embed.shape[1]):
                ret[str(i)] = self.arr_embed[:, i]
        elif isinstance(X, np.ndarray):
            ret = np.asarray(self.arr_embed)

        return ret
