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
import typing

import numpy as np

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import ClusterMixin, CMajorInputTagMixin
from cuml.internals.utils import check_random_seed

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

cimport cuml.cluster.cpp.kmeans as lib
from cuml.internals.logger cimport level_enum
from cuml.metrics.distance_type cimport DistanceType


cdef _kmeans_init_params(kmeans, lib.KMeansParams& params):
    """Initialize a passed KMeansParams instance from a KMeans instance."""
    params.n_clusters = kmeans.n_clusters
    params.max_iter = kmeans.max_iter
    params.tol = kmeans.tol
    params.rng_state.seed = check_random_seed(kmeans.random_state)
    params.verbosity = <level_enum>(<int>kmeans.verbose)
    params.metric = DistanceType.L2Expanded
    params.batch_samples = int(kmeans.max_samples_per_batch)
    params.oversampling_factor = kmeans.oversampling_factor

    if isinstance(kmeans.init, str):
        if kmeans.init == "k-means++":
            # K-means++ is the constrained case of k-means|| w/
            # oversampling factor = 0
            params.oversampling_factor = 0
            params.init = lib.InitMethod.KMeansPlusPlus
        elif kmeans.init in ('scalable-k-means++', 'k-means||'):
            params.init = lib.InitMethod.KMeansPlusPlus
        elif kmeans.init == "random":
            params.init = lib.InitMethod.Random
    else:
        params.init = lib.InitMethod.Array

    if kmeans.n_init == "auto":
        if isinstance(kmeans.init, str) and params.init == lib.InitMethod.KMeansPlusPlus:
            params.n_init = 1
        else:
            params.n_init = 10
    else:
        params.n_init = kmeans.n_init


cdef _kmeans_fit(
    handle_t& handle,
    lib.KMeansParams& params,
    X,
    sample_weight,
    centers,
):
    """Fit the kmeans centers and return `n_iter`"""
    cdef int64_t n_rows = X.shape[0]
    cdef int64_t n_cols = X.shape[1]

    cdef bool values_f32 = X.dtype == np.float32
    cdef bool indices_i32 = (n_rows * n_cols) < (2**31 - 1)

    cdef uintptr_t X_ptr = X.ptr
    cdef uintptr_t centers_ptr = centers.ptr
    cdef uintptr_t sample_weight_ptr = sample_weight.ptr

    cdef int n_iter_32 = 0
    cdef int64_t n_iter_64 = 0
    cdef float inertia_32 = 0
    cdef double inertia_64 = 0

    with nogil:
        if values_f32:
            if indices_i32:
                lib.fit(
                    handle,
                    params,
                    <float *>X_ptr,
                    <int>n_rows,
                    <int>n_cols,
                    <float *>sample_weight_ptr,
                    <float *>centers_ptr,
                    inertia_32,
                    n_iter_32,
                )
            else:
                lib.fit(
                    handle,
                    params,
                    <float *>X_ptr,
                    n_rows,
                    n_cols,
                    <float *>sample_weight_ptr,
                    <float *>centers_ptr,
                    inertia_32,
                    n_iter_64,
                )
        else:
            if indices_i32:
                lib.fit(
                    handle,
                    params,
                    <double *>X_ptr,
                    <int>n_rows,
                    <int>n_cols,
                    <double *>sample_weight_ptr,
                    <double *>centers_ptr,
                    inertia_64,
                    n_iter_32,
                )
            else:
                lib.fit(
                    handle,
                    params,
                    <double *>X_ptr,
                    n_rows,
                    n_cols,
                    <double *>sample_weight_ptr,
                    <double *>centers_ptr,
                    inertia_64,
                    n_iter_64,
                )
    return n_iter_32 if indices_i32 else n_iter_64


cdef _kmeans_predict(
    handle_t& handle,
    lib.KMeansParams &params,
    X,
    sample_weight,
    centers,
):
    """Predict labels & inertia from a fit `KMeans`.

    Split out to be shared between `KMeans.fit` and `KMeans.predict`
    """
    cdef int64_t n_rows = X.shape[0]
    cdef int64_t n_cols = X.shape[1]

    labels = CumlArray.zeros(
        shape=n_rows,
        dtype=(np.int32 if n_rows * n_cols < 2**31 - 1 else np.int64),
    )

    cdef uintptr_t X_ptr = X.ptr
    cdef uintptr_t centers_ptr = centers.ptr
    cdef uintptr_t sample_weight_ptr = sample_weight.ptr
    cdef uintptr_t labels_ptr = labels.ptr

    cdef bool values_f32 = X.dtype == np.float32
    cdef bool indices_i32 = labels.dtype == np.int32

    cdef float inertia_f32 = 0
    cdef double inertia_f64 = 0

    with nogil:
        if values_f32:
            if indices_i32:
                lib.predict(
                    handle,
                    params,
                    <float*>centers_ptr,
                    <float*>X_ptr,
                    <int>n_rows,
                    <int>n_cols,
                    <float*>sample_weight_ptr,
                    True,
                    <int*>labels_ptr,
                    inertia_f32,
                )
            else:
                lib.predict(
                    handle,
                    params,
                    <float*>centers_ptr,
                    <float*>X_ptr,
                    <int64_t>n_rows,
                    <int64_t>n_cols,
                    <float*>sample_weight_ptr,
                    True,
                    <int64_t*>labels_ptr,
                    inertia_f32,
                )
        else:
            if indices_i32:
                lib.predict(
                    handle,
                    params,
                    <double*>centers_ptr,
                    <double*>X_ptr,
                    <int>n_rows,
                    <int>n_cols,
                    <double*>sample_weight_ptr,
                    True,
                    <int*>labels_ptr,
                    inertia_f64,
                )
            else:
                lib.predict(
                    handle,
                    params,
                    <double*>centers_ptr,
                    <double*>X_ptr,
                    <int64_t>n_rows,
                    <int64_t>n_cols,
                    <double*>sample_weight_ptr,
                    True,
                    <int64_t*>labels_ptr,
                    inertia_f64,
                )

    inertia = inertia_f32 if values_f32 else inertia_f64

    return labels, inertia


class KMeans(Base,
             InteropMixin,
             ClusterMixin,
             CMajorInputTagMixin):
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

        >>> # Both import methods supported
        >>> from cuml import KMeans
        >>> from cuml.cluster import KMeans
        >>> import cudf
        >>> import numpy as np
        >>> import pandas as pd
        >>>
        >>> a = np.asarray([[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]],
        ...                dtype=np.float32)
        >>> b = cudf.DataFrame(a)
        >>> # Input:
        >>> b
            0    1
        0  1.0  1.0
        1  1.0  2.0
        2  3.0  2.0
        3  4.0  3.0
        >>>
        >>> # Calling fit
        >>> kmeans_float = KMeans(n_clusters=2, n_init="auto", random_state=1)
        >>> kmeans_float.fit(b)
        KMeans()
        >>>
        >>> # Labels:
        >>> kmeans_float.labels_
        0    0
        1    0
        2    1
        3    1
        dtype: int32
        >>> # cluster_centers:
        >>> kmeans_float.cluster_centers_
            0    1
        0  1.0  1.5
        1  3.5  2.5

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    n_clusters : int (default = 8)
        The number of centroids or clusters you want.
    max_iter : int (default = 300)
        The more iterations of EM, the more accurate, but slower.
    tol : float64 (default = 1e-4)
        Stopping criterion when centroid means do not change much.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    random_state : int or None (default = None)
        If you want results to be the same when you restart Python, select a
        state.
    init : {'scalable-k-means++', 'k-means||', 'random'} or an \
            ndarray (default = 'scalable-k-means++')

         - ``'scalable-k-means++'`` or ``'k-means||'``: Uses fast and stable
           scalable kmeans++ initialization.
         - ``'random'``: Choose `n_cluster` observations (rows) at random
           from data for the initial centroids.
         - If an ndarray is passed, it should be of
           shape (`n_clusters`, `n_features`) and gives the initial centers.

    n_init: 'auto' or int (default = 'auto')
        Number of instances the k-means algorithm will be called with
        different seeds. The final results will be from the instance
        that produces lowest inertia out of n_init instances.

        When `n_init='auto'`, the number of runs depends on the value of
        `init`: 1 if using `init='"k-means||"` or `init="scalable-k-means++"`;
        10 otherwise.

        .. versionadded:: 25.02
           Added 'auto' option for `n_init`.

        .. versionchanged:: 25.04
            Default value for `n_init` will change from 1 to `'auto'` in version 25.04.

    oversampling_factor : float64 (default = 2.0)
        The amount of points to sample
        in scalable k-means++ initialization for potential centroids.
        Increasing this value can lead to better initial centroids at the
        cost of memory. The total number of centroids sampled in scalable
        k-means++ is oversampling_factor * n_clusters * 8.
    max_samples_per_batch : int (default = 32768)
        The number of data samples to use for batches of the pairwise distance
        computation. This computation is done throughout both fit predict. The
        default should suit most cases. The total number of elements in the
        batched pairwise distance computation is :py:`max_samples_per_batch *
        n_clusters`. It might become necessary to lower this number when
        `n_clusters` becomes prohibitively large.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    cluster_centers_ : array
        The coordinates of the final clusters. This represents of "mean" of
        each data cluster.
    labels_ : array
        Which cluster each datapoint belongs to.

    Notes
    -----
    KMeans requires `n_clusters` to be specified. This means one needs to
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
    labels_ = CumlArrayDescriptor(order="C")
    cluster_centers_ = CumlArrayDescriptor(order="C")

    _cpu_class_path = "sklearn.cluster.KMeans"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "n_init",
            "oversampling_factor",
            "max_samples_per_batch",
            "init",
            "max_iter",
            "n_clusters",
            "random_state",
            "tol",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if callable(model.init):
            raise UnsupportedOnGPU(f"`init={model.init!r}` is not supported")
        elif isinstance(model.init, str):
            if model.init == "k-means++":
                init = "scalable-k-means++"
            elif model.init == "random":
                init = "random"
            else:
                # Should be unreachable, here in case sklearn adds more init values
                raise UnsupportedOnGPU(f"`init={model.init!r}` is not supported")
        else:
            init = model.init  # array-like

        return {
            "n_clusters": model.n_clusters,
            "init": init,
            "n_init": model.n_init,
            "max_iter": model.max_iter,
            "tol": model.tol,
            "random_state": model.random_state,
        }

    def _params_to_cpu(self):
        init = self.init
        if not isinstance(init, str):
            init = to_cpu(init)  # array-like
        elif init == "scalable-k-means++":
            init = "k-means++"
        return {
            "n_clusters": self.n_clusters,
            "init": init,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }

    def _attrs_from_cpu(self, model):
        return {
            "cluster_centers_": to_gpu(model.cluster_centers_, order="C"),
            "labels_": to_gpu(model.labels_, order="C"),
            "inertia_": to_gpu(model.inertia_),
            "n_iter_": model.n_iter_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        try:
            from sklearn.utils._openmp_helpers import (
                _openmp_effective_n_threads,
            )
        except ImportError:
            n_threads = 1
        else:
            n_threads = _openmp_effective_n_threads()

        return {
            "cluster_centers_": to_cpu(self.cluster_centers_),
            "labels_": to_cpu(self.labels_),
            "inertia_": to_cpu(self.inertia_),
            "n_iter_": self.n_iter_,
            # sklearn's KMeans relies on a few private attributes to work
            "_n_features_out": self._n_features_out,
            "_n_threads": n_threads,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        handle=None,
        n_clusters=8,
        max_iter=300,
        tol=1e-4,
        verbose=False,
        random_state=None,
        init='scalable-k-means++',
        n_init="auto",
        oversampling_factor=2.0,
        max_samples_per_batch=1<<15,
        output_type=None,
    ):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.oversampling_factor = oversampling_factor
        self.max_samples_per_batch = max_samples_per_batch

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # Exposed to support sklearn's `get_feature_names_out`
        return self.n_clusters

    @generate_docstring()
    def fit(self, X, y=None, sample_weight=None, *, convert_dtype=True) -> "KMeans":
        """
        Compute k-means clustering with X.

        """
        return self._fit(X, y=y, sample_weight=sample_weight, convert_dtype=convert_dtype)

    def _fit(
        self, X, y=None, sample_weight=None, *, convert_dtype=True, multigpu=False
    ) -> "KMeans":
        # Process input arrays
        X_m, n_rows, n_cols, dtype = input_to_cuml_array(
            X,
            order="C",
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64],
        )
        if sample_weight is None:
            sample_weight_m = CumlArray.ones(shape=n_rows, dtype=dtype)
        else:
            sample_weight_m = input_to_cuml_array(
                sample_weight,
                order="C",
                convert_to_dtype=(dtype if convert_dtype else None),
                check_dtype=dtype,
                check_rows=n_rows,
                check_cols=1,
            ).array

        # Validate input dimensions match what's expected
        for kind, n in [("sample", n_rows), ("feature", n_cols)]:
            if n == 0:
                raise ValueError(
                    f"Found array with 0 {kind}(s) (shape=({n_rows}, {n_cols})) "
                    "while a minimum of 1 is required by KMeans."
                )

        # Skip this check if running in multigpu mode. In that case we don't care if
        # a single partition has fewer rows than clusters
        if not multigpu and n_rows < self.n_clusters:
            raise ValueError(
                f"n_samples={n_rows} should be >= n_clusters={self.n_clusters}."
            )

        # Allocate output cluster_centers_
        if isinstance(self.init, str):
            centers = CumlArray.zeros(
                shape=(self.n_clusters, n_cols), dtype=dtype, order="C",
            )
        else:
            # Initial array provided, coerce to device array and validate
            centers = input_to_cuml_array(
                self.init,
                order="C",
                convert_to_dtype=(dtype if convert_dtype else None),
                check_dtype=dtype,
                deepcopy=True,
            ).array
            if centers.shape[0] != self.n_clusters:
                raise ValueError(
                    f"The shape of the initial centers {centers.shape} does not "
                    f"match the number of clusters {self.n_clusters}."
                )
            if centers.shape[1] != X.shape[1]:
                raise ValueError(
                    f"The shape of the initial centers {centers.shape} does not "
                    f"match the number of features of the data {X.shape[1]}."
                )

        # Prepare for libcuml call
        cdef handle_t* handle_ = <handle_t *><size_t>self.handle.getHandle()
        cdef lib.KMeansParams params
        _kmeans_init_params(self, params)
        n_iter = _kmeans_fit(handle_[0], params, X_m, sample_weight_m, centers)
        labels, inertia = _kmeans_predict(handle_[0], params, X_m, sample_weight_m, centers)
        self.handle.sync()

        # Store fitted attributes and return
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter

        return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Cluster indexes',
                                       'shape': '(n_samples, 1)'})
    def fit_predict(self, X, y=None, sample_weight=None) -> CumlArray:
        """
        Compute cluster centers and predict cluster index for each sample.

        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def _predict_labels_inertia(
        self, X, convert_dtype=True, sample_weight=None
    ) -> typing.Tuple[CumlArray, float]:
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

        sample_weight : array-like (device or host) shape = (n_samples,), default=None # noqa
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : array
        Which cluster each datapoint belongs to.

        inertia : float
        Sum of squared distances of samples to their closest cluster center.
        """
        dtype = self.cluster_centers_.dtype

        X_m, n_rows, _, _ = input_to_cuml_array(
            X,
            order="C",
            check_dtype=dtype,
            convert_to_dtype=(dtype if convert_dtype else None),
            check_cols=self.n_features_in_,
        )

        if sample_weight is None:
            sample_weight_m = CumlArray.ones(shape=n_rows, dtype=dtype)
        else:
            sample_weight_m = input_to_cuml_array(
                sample_weight,
                order="C",
                check_dtype=dtype,
                convert_to_dtype=(dtype if convert_dtype else None),
                check_rows=n_rows,
                check_cols=1,
            )

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef lib.KMeansParams params
        _kmeans_init_params(self, params)

        labels, inertia = _kmeans_predict(
            handle_[0], params, X_m, sample_weight_m, self.cluster_centers_
        )
        self.handle.sync()
        return labels, inertia

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Cluster indexes',
                                       'shape': '(n_samples, 1)'})
    def predict(
        self,
        X,
        *,
        convert_dtype=True,
    ) -> CumlArray:
        """
        Predict the closest cluster each sample in X belongs to.

        """
        labels, _ = self._predict_labels_inertia(X, convert_dtype=convert_dtype)
        return labels

    @generate_docstring(return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Transformed data',
                                       'shape': '(n_samples, n_clusters)'})
    def transform(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Transform X to a cluster-distance space.

        """
        dtype = self.cluster_centers_.dtype

        X_m = input_to_cuml_array(
            X,
            order="C",
            check_dtype=dtype,
            convert_to_dtype=(dtype if convert_dtype else None),
            check_cols=self.cluster_centers_.shape[1],
        ).array

        cdef int64_t n_rows = X_m.shape[0]
        cdef int64_t n_cols = X_m.shape[1]

        out = CumlArray.zeros(
            shape=(n_rows, self.n_clusters), dtype=dtype, order="C",
        )

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t centers_ptr = self.cluster_centers_.ptr
        cdef uintptr_t out_ptr = out.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef lib.KMeansParams params
        _kmeans_init_params(self, params)

        cdef bool values_f32 = dtype == np.float32
        cdef bool indices_i32 = self.labels_.dtype == np.int32

        with nogil:
            if values_f32:
                if indices_i32:
                    lib.transform(
                        handle_[0],
                        params,
                        <float*>centers_ptr,
                        <float*>X_ptr,
                        <int>n_rows,
                        <int>n_cols,
                        <float*>out_ptr,
                    )
                else:
                    lib.transform(
                        handle_[0],
                        params,
                        <float*>centers_ptr,
                        <float*>X_ptr,
                        <int64_t>n_rows,
                        <int64_t>n_cols,
                        <float*>out_ptr,
                    )
            else:
                if indices_i32:
                    lib.transform(
                        handle_[0],
                        params,
                        <double*>centers_ptr,
                        <double*>X_ptr,
                        <int>n_rows,
                        <int>n_cols,
                        <double*>out_ptr,
                    )
                else:
                    lib.transform(
                        handle_[0],
                        params,
                        <double*>centers_ptr,
                        <double*>X_ptr,
                        <int64_t>n_rows,
                        <int64_t>n_cols,
                        <double*>out_ptr,
                    )
        self.handle.sync()
        return out

    @generate_docstring(return_values={'name': 'score',
                                       'type': 'float',
                                       'description': 'Opposite of the value \
                                                        of X on the K-means \
                                                        objective.'})
    def score(self, X, y=None, sample_weight=None, *, convert_dtype=True):
        """
        Opposite of the value of X on the K-means objective.

        """

        inertia = self._predict_labels_inertia(
            X, convert_dtype=convert_dtype, sample_weight=sample_weight
        )[1]
        return -1 * inertia

    @generate_docstring(return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Transformed data',
                                       'shape': '(n_samples, n_clusters)'})
    def fit_transform(
        self, X, y=None, sample_weight=None, *, convert_dtype=False
    ) -> CumlArray:
        """
        Compute clustering and transform X to cluster-distance space.

        """
        self.fit(X, sample_weight=sample_weight)
        return self.transform(X, convert_dtype=convert_dtype)
