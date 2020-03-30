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
import rmm
import warnings

from libcpp cimport bool
from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import input_to_cuml_array

cdef extern from "cuml/cluster/kmeans.hpp" namespace \
        "ML::kmeans::KMeansParams":
    enum InitMethod:
        KMeansPlusPlus, Random, Array

cdef extern from "cuml/cluster/kmeans.hpp" namespace "ML::kmeans":

    cdef struct KMeansParams:
        int n_clusters,
        InitMethod init
        int max_iter,
        double tol,
        int verbose,
        int seed,
        int metric,
        int n_init,
        double oversampling_factor,
        int batch_samples,
        int batch_centroids,
        bool inertia_check

    cdef void fit_predict(cumlHandle& handle,
                          KMeansParams& params,
                          const float *X,
                          int n_samples,
                          int n_features,
                          float *centroids,
                          int *labels,
                          float &inertia,
                          int &n_iter) except +

    cdef void fit_predict(cumlHandle& handle,
                          KMeansParams& params,
                          const double *X,
                          int n_samples,
                          int n_features,
                          double *centroids,
                          int *labels,
                          double &inertia,
                          int &n_iter) except +

    cdef void predict(cumlHandle& handle,
                      KMeansParams& params,
                      const float *centroids,
                      const float *X,
                      int n_samples,
                      int n_features,
                      int *labels,
                      float &inertia) except +

    cdef void predict(cumlHandle& handle,
                      KMeansParams& params,
                      double *centroids,
                      const double *X,
                      int n_samples,
                      int n_features,
                      int *labels,
                      double &inertia) except +

    cdef void transform(cumlHandle& handle,
                        KMeansParams& params,
                        const float *centroids,
                        const float *X,
                        int n_samples,
                        int n_features,
                        int metric,
                        float *X_new) except +

    cdef void transform(cumlHandle& handle,
                        KMeansParams& params,
                        const double *centroids,
                        const double *X,
                        int n_samples,
                        int n_features,
                        int metric,
                        double *X_new) except +


class KMeans(Base):

    """
    KMeans is a basic but powerful clustering method which is optimized via
    Expectation Maximization. It randomly selects K data points in X, and
    computes which samples are close to these points.
    For every cluster of points, a mean is computed (hence the name), and this
    becomes the new centroid.

    cuML's KMeans expects an array-like object or cuDF DataFrame, and supports
    the scalable KMeans++ initialization method. This method is more stable
    than randomly selecting K points.

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


        a = np.asarray([[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]],
                       dtype=np.float32)
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

             0    0
             1    0
             2    1
             3    1

          cluster_centers:

             0    1
          0  1.0  1.5
          1  3.5  2.5

    Parameters
    ----------
    handle : cuml.Handle
        If it is None, a new one is created just for this class.
    n_clusters : int (default = 8)
        The number of centroids or clusters you want.
    max_iter : int (default = 300)
        The more iterations of EM, the more accurate, but slower.
    tol : float64 (default = 1e-4)
        Stopping criterion when centroid means do not change much.
    verbose : boolean (default = 0)
        If True, prints diagnostic information.
    random_state : int (default = 1)
        If you want results to be the same when you restart Python, select a
        state.
    init : 'scalable-kmeans++', 'k-means||' , 'random' or an ndarray (default = 'scalable-k-means++')  # noqa
        'scalable-k-means++' or 'k-means||': Uses fast and stable scalable
        kmeans++ initialization.
        'random': Choose 'n_cluster' observations (rows) at random from data
        for the initial centroids. If an ndarray is passed, it should be of
        shape (n_clusters, n_features) and gives the initial centers.
    n_init: int (default = 1)
        Number of instances the k-means algorithm will be called with different seeds.
        The final results will be from the instance that produces lowest inertia out
        of n_init instances.
    oversampling_factor : float64
        scalable k-means|| oversampling factor
    max_samples_per_batch : int (default=1<<15)
        maximum number of samples to use for each batch
        of the pairwise distance computation.
    oversampling_factor : int (default = 2)
        The amount of points to sample
        in scalable k-means++ initialization for potential centroids.
        Increasing this value can lead to better initial centroids at the
        cost of memory. The total number of centroids sampled in scalable
        k-means++ is oversampling_factor * n_clusters * 8.
    max_samples_per_batch : int (default = 32768)
        The number of data
        samples to use for batches of the pairwise distance computation.
        This computation is done throughout both fit predict. The default
        should suit most cases. The total number of elements in the batched
        pairwise distance computation is max_samples_per_batch * n_clusters.
        It might become necessary to lower this number when n_clusters
        becomes prohibitively large.

    Attributes
    ----------
    cluster_centers_ : array
        The coordinates of the final clusters. This represents of "mean" of
        each data cluster.
    labels_ : array
        Which cluster each datapoint belongs to.

    Notes
    ------
    KMeans requires n_clusters to be specified. This means one needs to
    approximately guess or know how many clusters a dataset has. If one is not
    sure, one can start with a small number of clusters, and visualize the
    resulting clusters with PCA, UMAP or T-SNE, and verify that they look
    appropriate.

    **Applications of KMeans**

        The biggest advantage of KMeans is its speed and simplicity. That is
        why KMeans is many practitioner's first choice of a clustering
        algorithm. KMeans has been extensively used when the number of clusters
        is approximately known, such as in big data clustering tasks,
        image segmentation and medical clustering.


    For additional docs, see `scikitlearn's Kmeans
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.
    """

    def __init__(self, handle=None, n_clusters=8, max_iter=300, tol=1e-4,
                 verbose=0, random_state=1, init='scalable-k-means++',
                 n_init=1, oversampling_factor=2.0,
                 max_samples_per_batch=1<<15, output_type=None):
        super(KMeans, self).__init__(handle, verbose, output_type)
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.inertia_ = 0
        self.n_iter_ = 0
        self.oversampling_factor=oversampling_factor
        self.max_samples_per_batch=int(max_samples_per_batch)

        # internal array attributes
        self._labels_ = None  # accessed via estimator.labels_
        self._cluster_centers_ = None  # accessed via estimator.cluster_centers_  # noqa

        cdef KMeansParams params
        params.n_clusters = <int>self.n_clusters

        # K-means++ is the constrained case of k-means||
        # w/ oversampling factor = 0
        if (init == 'k-means++'):
            init = 'k-means||'
            self.oversampling_factor = 0

        if (init in ['scalable-k-means++', 'k-means||']):
            self.init = init
            params.init = KMeansPlusPlus

        elif (init == 'random'):
            self.init = init
            params.init = Random

        else:
            self.init = 'preset'
            params.init = Array
            self._cluster_centers_, n_rows, self.n_cols, self.dtype = \
                input_to_cuml_array(init, order='C',
                                    check_dtype=[np.float32, np.float64])

        params.max_iter = <int>self.max_iter
        params.tol = <double>self.tol
        params.verbose = <int>self.verbose
        params.seed = <int>self.random_state
        params.metric = 0   # distance metric as squared L2: @todo - support other metrics # noqa: E501
        params.batch_samples=<int>self.max_samples_per_batch
        params.oversampling_factor=<double>self.oversampling_factor
        params.n_init = <int>self.n_init
        self._params = params

    def fit(self, X):
        """
        Compute k-means clustering with X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        """

        self._set_output_type(X)

        if self.init == 'preset':
            check_cols = self.n_cols
            check_dtype = self.dtype
        else:
            check_cols = False
            check_dtype = [np.float32, np.float64]

        X_m, n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, order='C',
                                check_cols=check_cols,
                                check_dtype=check_dtype)

        cdef uintptr_t input_ptr = X_m.ptr

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        self._labels_ = CumlArray.zeros(shape=n_rows, dtype=np.int32)
        cdef uintptr_t labels_ptr = self._labels_.ptr

        if (self.init in ['scalable-k-means++', 'k-means||', 'random']):
            self._cluster_centers_ = \
                CumlArray.zeros(shape=(self.n_clusters, self.n_cols),
                                dtype=self.dtype, order='C')

        cdef uintptr_t cluster_centers_ptr = self._cluster_centers_.ptr

        cdef float inertiaf = 0
        cdef double inertiad = 0

        cdef KMeansParams params = self._params
        cdef int n_iter = 0

        if self.dtype == np.float32:
            fit_predict(
                handle_[0],
                <KMeansParams> params,
                <const float*> input_ptr,
                <size_t> n_rows,
                <size_t> self.n_cols,
                <float*> cluster_centers_ptr,
                <int*> labels_ptr,
                inertiaf,
                n_iter)
            self.handle.sync()
            self.inertia_ = inertiaf
            self.n_iter_ = n_iter
        elif self.dtype == np.float64:
            fit_predict(
                handle_[0],
                <KMeansParams> params,
                <const double*> input_ptr,
                <size_t> n_rows,
                <size_t> self.n_cols,
                <double*> cluster_centers_ptr,
                <int*> labels_ptr,
                inertiad,
                n_iter)
            self.handle.sync()
            self.inertia_ = inertiad
            self.n_iter_ = n_iter
        else:
            raise TypeError('KMeans supports only float32 and float64 input,'
                            'but input type ' + str(self.dtype) +
                            ' passed.')

        self.handle.sync()

        del(X_m)

        return self

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        """
        return self.fit(X).labels_

    def _predict_labels_inertia(self, X, convert_dtype=False):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = False)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.

        Returns
        -------
        labels : array
        Which cluster each datapoint belongs to.

        inertia : float/double
        Sum of squared distances of samples to their closest cluster center.
        """

        out_type = self._get_output_type(X)

        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C', check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)

        cdef uintptr_t input_ptr = X_m.ptr

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t cluster_centers_ptr = self._cluster_centers_.ptr

        self._labels_ = CumlArray.zeros(shape=n_rows, dtype=np.int32)
        cdef uintptr_t labels_ptr = self._labels_.ptr

        # Sum of squared distances of samples to their closest cluster center.
        cdef float inertiaf = 0
        cdef double inertiad = 0

        if self.dtype == np.float32:
            predict(
                handle_[0],
                <KMeansParams> self._params,
                <float*> cluster_centers_ptr,
                <float*> input_ptr,
                <size_t> n_rows,
                <size_t> self.n_cols,
                <int*> labels_ptr,
                inertiaf)
            self.handle.sync()
            inertia = inertiaf
        elif self.dtype == np.float64:
            predict(
                handle_[0],
                <KMeansParams> self._params,
                <double*> cluster_centers_ptr,
                <double*> input_ptr,
                <size_t> n_rows,
                <size_t> self.n_cols,
                <int*> labels_ptr,
                inertiad)
            self.handle.sync()
            inertia = inertiad
        else:
            raise TypeError('KMeans supports only float32 and float64 input,'
                            'but input type ' + str(self.dtype) +
                            ' passed.')

        self.handle.sync()
        del(X_m)

        return self._labels_.to_output(out_type), inertia

    def predict(self, X, convert_dtype=False):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        Returns
        -------
        labels : array
        Which cluster each datapoint belongs to.
        """

        labels, _ = self._predict_labels_inertia(X,
                                                 convert_dtype=convert_dtype)
        return labels

    def transform(self, X, convert_dtype=False):
        """
        Transform X to a cluster-distance space.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = False)
            When set to True, the transform method will, when necessary,
            convert the input to the data type which was used to train the
            model. This will increase memory used for the method.


        """

        out_type = self._get_output_type(X)

        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C', check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)

        cdef uintptr_t input_ptr = X_m.ptr

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t cluster_centers_ptr = self._cluster_centers_.ptr

        preds = CumlArray.zeros(shape=(n_rows, self.n_clusters),
                                dtype=self.dtype,
                                order='C')

        cdef uintptr_t preds_ptr = preds.ptr

        # distance metric as L2-norm/euclidean distance: @todo - support other metrics # noqa: E501
        distance_metric = 1

        if self.dtype == np.float32:
            transform(
                handle_[0],
                <KMeansParams> self._params,
                <float*> cluster_centers_ptr,
                <float*> input_ptr,
                <size_t> n_rows,
                <size_t> self.n_cols,
                <int> distance_metric,
                <float*> preds_ptr)
        elif self.dtype == np.float64:
            transform(
                handle_[0],
                <KMeansParams> self._params,
                <double*> cluster_centers_ptr,
                <double*> input_ptr,
                <size_t> n_rows,
                <size_t> self.n_cols,
                <int> distance_metric,
                <double*> preds_ptr)
        else:
            raise TypeError('KMeans supports only float32 and float64 input,'
                            'but input type ' + str(self.dtype) +
                            ' passed.')

        self.handle.sync()

        del(X_m)
        return preds.to_output(out_type)

    def score(self, X):
        """
        Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        Returns
        -------
        score: float
                 Opposite of the value of X on the K-means objective.
        """

        return -1 * self._predict_labels_inertia(X)[1]

    def fit_transform(self, X, convert_dtype=False):
        """
        Compute clustering and transform X to cluster-distance space.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = False)
            When set to True, the fit_transform method will automatically
            convert the input to the data type which was used to train the
            model. This will increase memory used for the method.

        """
        return self.fit(X).transform(X, convert_dtype=convert_dtype)

    def get_param_names(self):
        return ['n_init', 'oversampling_factor', 'max_samples_per_batch',
                'init', 'max_iter', 'n_clusters', 'random_state',
                'tol', 'verbose']
