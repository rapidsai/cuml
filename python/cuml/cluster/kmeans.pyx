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

import ctypes
import cudf
import numpy as np
import warnings

from numba import cuda
from cuml import numba_utils

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free


class KMeans:

    """
    KMeans is a basic but powerful clustering method which is optimized via Expectation Maximization.
    It randomnly selects K data points in X, and computes which samples are close to these points.
    For every cluster of points, a mean is computed (hence the name), and this becomes the new
    centroid.

    cuML's KMeans expects a cuDF DataFrame, and supports the fast KMeans++ intialization method. This
    method is more stable than randomnly selecting K points.

    Examples
    --------

    .. code-block:: python

        # Both import methods supported
        from cuml import KMeans
        from cuml.cluster import KMeans

        import cudf
        import numpy as np
        import pandas as pd

        def np2cudf(df):
            # convert numpy array to cuDF dataframe
            df = pd.DataFrame({'fea%d'%i:df[:,i] for i in range(df.shape[1])})
            pdf = cudf.DataFrame()
            for c,column in enumerate(df):
              pdf[str(c)] = df[column]
            return pdf


        a = np.asarray([[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]],dtype=np.float32)
        b = np2cudf(a)
        print("input:")
        print(b)

        print("Calling fit")
        kmeans_float = KMeans(n_clusters=2, n_gpu=-1)
        kmeans_float.fit(b)

        print("labels:")
        print(kmeans_float.labels_)
        print("cluster_centers:")
        print(kmeans_float.cluster_centers_)


    Output:

    .. code-block:: python

          input:

               0    1
           0  1.0  1.0
           1  1.0  2.0
           2  3.0  2.0
           3  4.0  3.0

          Calling fit

          labels:

             0    1
             1    1
             2    0
             3    0

          cluster_centers:

             0    1
          0  3.5  2.5
          1  1.0  1.5

    Parameters
    ----------
    n_clusters : int (default = 8)
        The number of centroids or clusters you want.
    max_iter : int (default = 300)
        The more iterations of EM, the more accurate, but slower.
    tol : float (default = 1e-4)
        Stopping criterion when centroid means do not change much.
    verbose : boolean (default = 0)
        If True, prints diagnositc information.
    random_state : int (default = 1)
        If you want results to be the same when you restart Python, select a state.
    precompute_distances : boolean (default = 'auto')
        Not supported yet.
    init : 'kmeans++'
        Uses fast and stable kmeans++ intialization. More options will be supported later.
    n_init : int (default = 1)
        Number of times intialization is run. More is slower, but can be better.
    algorithm : "auto"
        Currently uses full EM, but will support others later.
    n_gpu : int (default = 1)
        Number of GPUs to use. If -1 is used, uses all GPUs. More usage is faster.
    gpu_id : int (default = 0)
        Which GPU to use if n_gpu == 1.


    Attributes
    ----------
    cluster_centers_ : array
        The coordinates of the final clusters. This represents of "mean" of each data cluster.
    labels_ : array
        Which cluster each datapoint belongs to.

    Notes
    ------
    KMeans requires n_clusters to be specified. This means one needs to approximately guess or know
    how many clusters a dataset has. If one is not sure, one can start with a small number of clusters, and
    visualize the resulting clusters with PCA, UMAP or T-SNE, and verify that they look appropriate.

    **Applications of KMeans**

        The biggest advantage of KMeans is its speed and simplicity. That is why KMeans is many practitioner's
        first choice of a clustering algorithm. KMeans has been extensively used when the number of clusters is
        approximately known, such as in big data clustering tasks, image segmentation and medical clustering.


    For additional docs, see `scikitlearn's Kmeans <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, verbose=0, random_state=1, precompute_distances='auto', init='kmeans++', n_init=1, algorithm='auto', n_gpu=1, gpu_id=0):
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.random_state = random_state
        self.precompute_distances = precompute_distances
        self.init = init
        self.n_init = n_init
        self.copy_x = None
        self.n_jobs = None
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.cluster_centers_ = None
        self.n_gpu = n_gpu
        self.gpu_id = gpu_id

        raise NotImplementedError("Old KMeans is deprecated and has been "
                                  "removed. Please use cuML 0.6 stable release"
                                  "for old KMeans. We will be adding the new "
                                  "KMeans soon.")

    def _get_ctype_ptr(self, obj):
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def fit(self, X):
        """
        Compute k-means clustering with X.

        Parameters
        ----------
        X : cuDF DataFrame
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        """

        cdef uintptr_t input_ptr
        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = numba_utils.row_matrix(X)
            self.n_rows = len(X)
            self.n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(X)
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        input_ptr = self._get_ctype_ptr(X_m)

        del(X_m)

        return self

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : cuDF DataFrame
                    Dense matrix (floats or doubles) of shape (n_samples, n_features)

        """
        return self.fit(X).labels_

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : cuDF DataFrame
                    Dense matrix (floats or doubles) of shape (n_samples, n_features)

        """

        cdef uintptr_t input_ptr
        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = numba_utils.row_matrix(X)
            self.n_rows = len(X)
            self.n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(X)
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        input_ptr = self._get_ctype_ptr(X_m)

        del(X_m)
        return self.labels_

    def transform(self, X):
        """
        Transform X to a cluster-distance space.

        Parameters
        ----------
        X : cuDF DataFrame
                    Dense matrix (floats or doubles) of shape (n_samples, n_features)

        """

        cdef uintptr_t input_ptr
        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = numba_utils.row_matrix(X)
            self.n_rows = len(X)
            self.n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(X)
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        input_ptr = self._get_ctype_ptr(X_m)



        del(X_m)
        return None

    def fit_transform(self, X):
        """
        Compute clustering and transform input_gdf to cluster-distance space.

        Parameters
        ----------
        X : cuDF DataFrame
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        """
        return self.fit(X).transform(X)

    def get_params(self, deep=True):
        """
        Sklearn style return parameter state

        Parameters
        -----------
        deep : boolean (default = True)
        """
        params = dict()
        variables = ['algorithm', 'copy_x', 'init', 'max_iter', 'n_clusters',
                     'n_init', 'n_jobs', 'precompute_distances',
                     'random_state', 'tol', 'verbose']
        for key in variables:
            var_value = getattr(self, key, None)
            params[key] = var_value
        return params

    def set_params(self, **params):
        """
        Sklearn style set parameter state to dictionary of params.

        Parameters
        -----------
        params : dict of new params
        """
        if not params:
            return self
        current_params = {"algorithm":self.algorithm,'copy_x':self.copy_x,'init':self.init,"max_iter":self.max_iter,
            "n_clusters":self.n_clusters,"n_init":self.n_init,"n_jobs":self.n_jobs, "precompute_distances":self.precompute_distances,
            "random_state":self.random_state,"tol":self.tol, "verbose":self.verbose}
        for key, value in params.items():
            if key not in current_params:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)
                current_params[key] = value
        return self
