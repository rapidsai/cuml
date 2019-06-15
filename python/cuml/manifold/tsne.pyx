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

cdef extern from "tsne/tsne.h" namespace "ML":
    void TSNE_fit(cumlHandle &handle,
                    float *X,
                    float *Y,
                    int n,
                    int p,
                    int n_components,
                    int n_neighbors,
                    float perplexity,
                    int perplexity_epochs,
                    int perplexity_tol,
                    float early_exaggeration,
                    int exaggeration_iter,
                    float min_gain,
                    float eta,
                    int epochs,
                    float pre_momentum,
                    float post_momentum,
                    long long seed,
                    bool initialize_embeddings,
                    bool verbose,
                    char *method) except +


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

    def __cinit__(self,
                n_neighbors=30,
                n_components=2,
                method = "Fast",
                epochs=300,
                perplexity=30.0,
                perplexity_epochs=100,
                perplexity_tol=1e-5,
                early_exaggeration=12.0,
                exaggeration_iter=150,
                min_gain=0.01,
                learning_rate=500.0,
                pre_momentum=0.8,
                post_momentum=0.5,
                random_state=-1,
                verbose=False,
                should_downcast=True,
                handle=None):

        self.handle = handle

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.method = 0 if method == "Naive" else 1
        self.epochs = epochs
        self.perplexity = perplexity
        self.perplexity_epochs = perplexity_epochs
        self.perplexity_tol = perplexity_tol
        self.early_exaggeration = early_exaggeration
        self.exaggeration_iter = exaggeration_iter
        self.min_gain = min_gain
        self.eta = learning_rate
        self.pre_momentum = pre_momentum
        self.post_momentum = post_momentum

        if random_state is None:
            self.seed = -1
        elif type(random_state) is int:
            self.seed = random_state
        else:
            self.seed = -1
        
        self.verbose = verbose

        self._should_downcast = should_downcast


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

            if len(X.shape) != 2:
                raise ValueError("data should be two dimensional")

            if self._should_downcast:
                X_m, X_ctype, n_rows, n_cols, dtype = \
                        input_to_dev_array(X, order='C', convert_to_dtype=np.float32)
            else:
                X_m, X_ctype, n_rows, n_cols, dtype = \
                        input_to_dev_array(X, order='C', check_dtype=np.float32)

            if n_rows <= 1:
                raise ValueError("There needs to be more than 1 sample to "
                                    "build nearest the neighbors graph")

            self.n_neighbors = min(n_rows, self.n_neighbors)
            self.n_dims = n_cols
            self.raw_data = X_ctype

            self.arr_embed = cuda.to_device(zeros((X_m.shape[0],
                                             self.n_components),
                                             order="C", dtype=np.float32))
            self.embeddings = self.arr_embed.device_ctypes_pointer.value

            cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

            cdef uintptr_t X_ptr = self.raw_data
            cdef uintptr_t embed_ptr = self.embeddings

            cdef uintptr_t y_raw
            TSNE(handle_[0],
                     <float*> X_ptr,
                     <float*>embed_ptr,
                     <int> X_m.shape[0],
                     <int> X_m.shape[1],
                     <int>self.n_components,
                     <int>self.n_neighbors,
                     <float>self.perplexity,
                     <int> self.perplexity_epochs,
                     <int> self.perplexity_tol,
                     <float>self.early_exaggeration,
                     <int> self.exaggeration_iter,
                     <float> self.min_gain,
                     <float> self.eta,
                     <int> self.epochs,
                     <float> self.pre_momentum,
                     <float> self.post_momentum,
                     <long long> self.seed,
                     <bool>True,
                     <bool>self.verbose,
                     <int>self.method)
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
