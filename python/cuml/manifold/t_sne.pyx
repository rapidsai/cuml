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

# cython: profile = False
# distutils: language = c++
# distutils: extra_compile_args = -Ofast
# cython: embedsignature = True, language_level = 3
# cython: boundscheck = False, wraparound = False

import cudf
import cuml
import ctypes
import numpy as np
import inspect
import pandas as pd
import warnings

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
import cuml.common.logger as logger

from cuml.common.array import CumlArray
from cuml.common.doc_utils import generate_docstring
from cuml.common import input_to_cuml_array
import rmm

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libcpp.memory cimport shared_ptr

cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "cuml/manifold/tsne.h" namespace "ML" nogil:
    cdef void TSNE_fit(
        const cumlHandle &handle,
        const float *X,
        float *Y,
        const int n,
        const int p,
        const int dim,
        int n_neighbors,
        const float theta,
        const float epssq,
        float perplexity,
        const int perplexity_max_iter,
        const float perplexity_tol,
        const float early_exaggeration,
        const int exaggeration_iter,
        const float min_gain,
        const float pre_learning_rate,
        const float post_learning_rate,
        const int max_iter,
        const float min_grad_norm,
        const float pre_momentum,
        const float post_momentum,
        const long long random_state,
        int verbosity,
        const bool intialize_embeddings,
        bool barnes_hut) except +


class TSNE(Base):
    """
    TSNE (T-Distributed Stochastic Neighbor Embedding) is an extremely
    powerful dimensionality reduction technique that aims to maintain
    local distances between data points. It is extremely robust to whatever
    dataset you give it, and is used in many areas including cancer research,
    music analysis and neural network weight visualizations.

    Currently, cuML's TSNE supports the fast Barnes Hut O(NlogN) TSNE
    approximation (derived from CannyLabs' BH open source CUDA code). This
    allows TSNE to produce extremely fast embeddings when n_components = 2.
    cuML defaults to this algorithm. A slower but more accurate Exact
    algorithm is also provided.

    Parameters
    -----------
    n_components : int (default 2)
        The output dimensionality size. Currently only size=2 is tested, but
        the 'exact' algorithm will support greater dimensionality in future.
    perplexity : float (default 30.0)
        Larger datasets require a larger value. Consider choosing different
        perplexity values from 5 to 50 and see the output differences.
    early_exaggeration : float (default 12.0)
        Controls the space between clusters. Not critical to tune this.
    learning_rate : float (default 200.0)
        The learning rate usually between (10, 1000). If this is too high,
        TSNE could look like a cloud / ball of points.
    n_iter : int (default 1000)
        The more epochs, the more stable/accurate the final embedding.
    n_iter_without_progress : int (default 300)
        Currently unused. When the KL Divergence becomes too small after some
        iterations, terminate TSNE early.
    min_grad_norm : float (default 1e-07)
        The minimum gradient norm for when TSNE will terminate early.
    metric : str 'euclidean' only (default 'euclidean')
        Currently only supports euclidean distance. Will support cosine in
        a future release.
    init : str 'random' (default 'random')
        Currently supports random intialization.
    verbose : int or boolean (default = False) (default logger.level_info)
        Level of verbosity.
        Most messages will be printed inside the Python Console.
    random_state : int (default None)
        Setting this can allow future runs of TSNE to look mostly the same.
        It is known that TSNE tends to have vastly different outputs on
        many runs. Try using PCA intialization (upcoming with change #1098)
        to possibly counteract this problem.
        It is known that small perturbations can directly
        change the result of the embedding for parallel TSNE implementations.
    method : str 'barnes_hut' or 'exact' (default 'barnes_hut')
        Options are either barnes_hut or exact. It is recommended that you use
        the barnes hut approximation for superior O(nlogn) complexity.
    angle : float (default 0.5)
        Tradeoff between accuracy and speed. Choose between (0,2 0.8) where
        closer to one indicates full accuracy but slower speeds.
    learning_rate_method : str 'adaptive', 'none' or None (default 'adaptive')
        Either adaptive or None. Uses a special adpative method that tunes
        the learning rate, early exaggeration and perplexity automatically
        based on input size.
    n_neighbors : int (default 90)
        The number of datapoints you want to use in the
        attractive forces. Smaller values are better for preserving
        local structure, whilst larger values can improve global structure
        preservation. Default is 3 * 30 (perplexity)
    perplexity_max_iter : int (default 100)
        The number of epochs the best gaussian bands are found for.
    exaggeration_iter : int (default 250)
        To promote the growth of clusters, set this higher.
    pre_momentum : float (default 0.5)
        During the exaggeration iteration, more forcefully apply gradients.
    post_momentum : float (default 0.8)
        During the late phases, less forcefully apply gradients.
    handle : (cuML Handle, default None)
        You can pass in a past handle that was initialized, or we will create
        one for you anew!

    References
    -----------
    .. [1] `van der Maaten, L.J.P.
       t-Distributed Stochastic Neighbor Embedding
       <https://lvdmaaten.github.io/tsne/>`_

    .. [2] van der Maaten, L.J.P.; Hinton, G.E.
       Visualizing High-Dimensional Data
       Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

    .. [3] George C. Linderman, Manas Rachh, Jeremy G. Hoskins,
        Stefan Steinerberger, Yuval Kluger Efficient Algorithms for
        t-distributed Stochastic Neighborhood Embedding

    .. tip::
        Maaten and Linderman showcased how TSNE can be very sensitive to both
        the starting conditions (ie random initialization), and how parallel
        versions of TSNE can generate vastly different results. It has been
        suggested that you run TSNE a few times to settle on the best
        configuration. Notice specifying random_state and fixing it across runs
        can help, but TSNE does not guarantee similar results each time.

        As suggested, PCA (upcoming with change #1098) can also help to
        alleviate this issue.

    .. note::
        The CUDA implementation is derived from the excellent CannyLabs open
        source implementation here: https://github.com/CannyLab/tsne-cuda/. The
        CannyLabs code is licensed according to the conditions in
        cuml/cpp/src/tsne/ cannylabs_tsne_license.txt. A full description of
        their approach is available in their article t-SNE-CUDA:
        GPU-Accelerated t-SNE and its Applications to Modern Data
        (https://arxiv.org/abs/1807.11824).

    """
    def __init__(self,
                 int n_components=2,
                 float perplexity=30.0,
                 float early_exaggeration=12.0,
                 float learning_rate=200.0,
                 int n_iter=1000,
                 int n_iter_without_progress=300,
                 float min_grad_norm=1e-07,
                 str metric='euclidean',
                 str init='random',
                 int verbose=False,
                 random_state=None,
                 str method='barnes_hut',
                 float angle=0.5,
                 learning_rate_method='adaptive',
                 int n_neighbors=90,
                 int perplexity_max_iter=100,
                 int exaggeration_iter=250,
                 float pre_momentum=0.5,
                 float post_momentum=0.8,
                 handle=None):

        super(TSNE, self).__init__(handle=handle, verbose=verbose)

        if n_components < 0:
            raise ValueError("n_components = {} should be more "
                             "than 0.".format(n_components))
        if n_components != 2 and method == 'barnes_hut':
            warnings.warn("Barnes Hut only works when n_components == 2. "
                          "Switching to exact.")
            method = 'exact'
        if n_components != 2:
            warnings.warn("Currently TSNE supports n_components = 2.")
            n_components = 2
        if perplexity < 0:
            raise ValueError("perplexity = {} should be more than 0.".format(
                             perplexity))
        if early_exaggeration < 0:
            raise ValueError("early_exaggeration = {} should be more "
                             "than 0.".format(early_exaggeration))
        if learning_rate < 0:
            raise ValueError("learning_rate = {} should be more "
                             "than 0.".format(learning_rate))
        if n_iter < 0:
            raise ValueError("n_iter = {} should be more than 0.".format(
                             n_iter))
        if n_iter <= 100:
            warnings.warn("n_iter = {} might cause TSNE to output wrong "
                          "results. Set it higher.".format(n_iter))
        if metric.lower() != 'euclidean':
            warnings.warn("TSNE does not support {} but only Euclidean. "
                          "Will do in the near future.".format(metric))
            metric = 'euclidean'
        if init.lower() != 'random':
            warnings.warn("TSNE does not support {} but only random "
                          "intialization. Will do in the near "
                          "future.".format(init))
            init = 'random'
        if angle < 0 or angle > 1:
            raise ValueError("angle = {} should be > 0 and less "
                             "than 1.".format(angle))
        if n_neighbors < 0:
            raise ValueError("n_neighbors = {} should be more "
                             "than 0.".format(n_neighbors))
        if n_neighbors > 1023:
            warnings.warn("n_neighbors = {} should be less than 1024")
            n_neighbors = 1023
        if perplexity_max_iter < 0:
            raise ValueError("perplexity_max_iter = {} should be more "
                             "than 0.".format(perplexity_max_iter))
        if exaggeration_iter < 0:
            raise ValueError("exaggeration_iter = {} should be more "
                             "than 0.".format(exaggeration_iter))
        if exaggeration_iter > n_iter:
            raise ValueError("exaggeration_iter = {} should be more less "
                             "than n_iter = {}.".format(exaggeration_iter,
                                                        n_iter))
        if pre_momentum < 0 or pre_momentum > 1:
            raise ValueError("pre_momentum = {} should be more than 0 "
                             "and less than 1.".format(pre_momentum))
        if post_momentum < 0 or post_momentum > 1:
            raise ValueError("post_momentum = {} should be more than 0 "
                             "and less than 1.".format(post_momentum))
        if pre_momentum > post_momentum:
            raise ValueError("post_momentum = {} should be more than "
                             "pre_momentum = {}".format(post_momentum,
                                                        pre_momentum))

        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_neighbors = n_neighbors
        self.perplexity_max_iter = perplexity_max_iter
        self.exaggeration_iter = exaggeration_iter
        self.pre_momentum = pre_momentum
        self.post_momentum = post_momentum
        if learning_rate_method is None:
            self.learning_rate_method = 'none'
        else:
            self.learning_rate_method = learning_rate_method.lower()
        self.epssq = 0.0025
        self.perplexity_tol = 1e-5
        self.min_gain = 0.01
        self.pre_learning_rate = learning_rate
        self.post_learning_rate = learning_rate * 2

    @generate_docstring(convert_dtype_cast='np.float32')
    def fit(self, X, convert_dtype=True):
        """
        Fit X into an embedded space.

        """
        self._set_n_features_in(X)
        cdef int n, p
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        if handle_ == NULL:
            raise ValueError("cuML Handle is Null! Terminating TSNE.")

        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        cdef uintptr_t X_ptr
        X_m, n, p, dtype = \
            input_to_cuml_array(X, order='C', check_dtype=np.float32,
                                convert_to_dtype=(np.float32 if convert_dtype
                                                  else None))
        X_ptr = X_m.ptr

        if n <= 1:
            raise ValueError("There needs to be more than 1 sample to build "
                             "nearest the neighbors graph")

        self.n_neighbors = min(n, self.n_neighbors)
        if self.perplexity > n:
            warnings.warn("Perplexity = {} should be less than the "
                          "# of datapoints = {}.".format(self.perplexity, n))
            self.perplexity = n

        # Prepare output embeddings
        Y = CumlArray.zeros(
            (n, self.n_components),
            order="F",
            dtype=np.float32)

        cdef uintptr_t embed_ptr = Y.ptr

        # Find best params if learning rate method is adaptive
        if self.learning_rate_method=='adaptive' and self.method=="barnes_hut":
            logger.debug("Learning rate is adaptive. In TSNE paper, "
                         "it has been shown that as n->inf, "
                         "Barnes Hut works well if n_neighbors->30, "
                         "learning_rate->20000, early_exaggeration->24.")
            logger.debug("cuML uses an adpative method."
                         "n_neighbors decreases to 30 as n->inf. "
                         "Likewise for the other params.")
            if n <= 2000:
                self.n_neighbors = min(max(self.n_neighbors, 90), n)
            else:
                # A linear trend from (n=2000, neigh=100) to (n=60000,neigh=30)
                self.n_neighbors = max(int(102 - 0.0012 * n), 30)
            self.pre_learning_rate = max(n / 3.0, 1)
            self.post_learning_rate = self.pre_learning_rate
            self.early_exaggeration = 24.0 if n > 10000 else 12.0
            if logger.should_log_for(logger.level_debug):
                logger.debug("New n_neighbors = {}, learning_rate = {}, "
                             "exaggeration = {}"
                             .format(self.n_neighbors, self.pre_learning_rate,
                                     self.early_exaggeration))

        cdef long long seed = -1
        if self.random_state is not None:
            seed = self.random_state

        TSNE_fit(handle_[0],
                 <float*> X_ptr,
                 <float*> embed_ptr,
                 <int> n,
                 <int> p,
                 <int> self.n_components,
                 <int> self.n_neighbors,
                 <float> self.angle,
                 <float> self.epssq,
                 <float> self.perplexity,
                 <int> self.perplexity_max_iter,
                 <float> self.perplexity_tol,
                 <float> self.early_exaggeration,
                 <int> self.exaggeration_iter,
                 <float> self.min_gain,
                 <float> self.pre_learning_rate,
                 <float> self.post_learning_rate,
                 <int> self.n_iter,
                 <float> self.min_grad_norm,
                 <float> self.pre_momentum,
                 <float> self.post_momentum,
                 <long long> seed,
                 <int> self.verbose,
                 <bool> True,
                 <bool> (self.method == 'barnes_hut'))

        # Clean up memory
        self._embedding_ = Y
        return self

    def __del__(self):
        if hasattr(self, '_embedding_'):
            del self._embedding_
            self._embedding_ = None

    @generate_docstring(convert_dtype_cast='np.float32',
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Embedding of the \
                                                       training data in \
                                                       low-dimensional space.',
                                       'shape': '(n_samples, n_components)'})
    def fit_transform(self, X, convert_dtype=True):
        """
        Fit X into an embedded space and return that transformed output.


        """
        self.fit(X, convert_dtype=convert_dtype)
        out_type = self._get_output_type(X)

        data = self._embedding_.to_output(out_type)
        del self._embedding_

        return data

    def __getstate__(self):
        state = self.__dict__.copy()
        if "handle" in state:
            del state["handle"]
        return state

    def __setstate__(self, state):
        super(TSNE, self).__init__(handle=None,
                                   verbose=state['verbose'])
        self.__dict__.update(state)
        return state
