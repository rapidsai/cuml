#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base, get_handle
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import ClusterMixin, CMajorInputTagMixin
from cuml.internals.outputs import reflect, run_in_internal_context
from cuml.internals.validation import (
    check_array,
    check_inputs,
    check_is_fitted,
    check_random_seed,
)

from libc.limits cimport INT_MAX
from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

cimport cuml.cluster.cpp.kmeans as lib
from cuml.metrics.distance_type cimport DistanceType


cdef inline bool _kmeans_indices_i32(int64_t n_rows, int64_t n_cols) noexcept nogil:
    cdef int64_t int_max = INT_MAX

    if n_rows < 0 or n_cols < 0:
        return False
    if n_rows > int_max or n_cols > int_max:
        return False
    if n_rows == 0 or n_cols == 0:
        return True

    return n_rows <= ((int_max - 1) // n_cols)


cdef _kmeans_init_params(kmeans, lib.KMeansParams& params):
    """Initialize a passed KMeansParams instance from a KMeans instance."""
    cdef bool multi_gpu = kmeans._multi_gpu

    params.n_clusters = kmeans.n_clusters
    params.max_iter = kmeans.max_iter
    params.tol = kmeans.tol
    params.verbosity = kmeans._verbose_level
    params.metric = DistanceType.L2Expanded
    params.batch_samples = int(kmeans.max_samples_per_batch)
    params.streaming_batch_size = int(kmeans.streaming_batch_size)
    params.oversampling_factor = kmeans.oversampling_factor

    # Ensure random_state is set when running on multi-gpu
    if multi_gpu and kmeans.random_state is None:
        raise ValueError(
            "KMeansMG requires `random_state != None`, please select a consistent "
            "non-None `random_state` to use across all partitions when calling "
            "KMeansMG"
        )
    params.rng_state.seed = check_random_seed(kmeans.random_state)

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

    if multi_gpu and params.oversampling_factor == 0:
        raise ValueError(
            "init='k-means++' or oversampling_factor=0 not supported for KMeansMG"
        )

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
    """Fit the kmeans centers and return `n_iter`.

    `X` and `sample_weight` may live on either the device or the host.
    `centers` always lives on the device.
    """
    cdef int64_t n_rows = X.shape[0]
    cdef int64_t n_cols = X.shape[1]

    cdef bool values_f32 = X.dtype == cp.float32
    # Indices fitting in int32 is for device arrays
    cdef bool host_data = not hasattr(X, "__cuda_array_interface__")
    cdef bool indices_i32 = (not host_data) and _kmeans_indices_i32(n_rows, n_cols)

    cdef uintptr_t X_ptr = X.data.ptr if isinstance(X, cp.ndarray) else X.ctypes.data
    cdef uintptr_t centers_ptr = centers.data.ptr
    cdef uintptr_t sample_weight_ptr = 0
    if sample_weight is not None:
        sample_weight_ptr = (
            sample_weight.data.ptr if isinstance(sample_weight, cp.ndarray)
            else sample_weight.ctypes.data
        )

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
    bool normalize_weights=True,
):
    """Predict labels & inertia from a fit `KMeans`.

    Split out to be shared between `KMeans.fit` and `KMeans.predict`
    """
    cdef int64_t n_rows = X.shape[0]
    cdef int64_t n_cols = X.shape[1]
    cdef int64_t n_clusters = centers.shape[0]

    # Stop-gap: downstream predict code indexes both X and cluster_centers_
    # using the selected index type. Keep the int32 path only when both
    # matrices fit int32 indexing until large-shape support is fully covered.
    cdef bool indices_i32 = (
        _kmeans_indices_i32(n_rows, n_cols)
        and _kmeans_indices_i32(n_clusters, n_cols)
    )
    labels = cp.zeros(shape=n_rows, dtype=(cp.int32 if indices_i32 else cp.int64))

    cdef uintptr_t X_ptr = X.data.ptr
    cdef uintptr_t centers_ptr = centers.data.ptr
    cdef uintptr_t sample_weight_ptr = sample_weight.data.ptr
    cdef uintptr_t labels_ptr = labels.data.ptr

    cdef bool values_f32 = X.dtype == cp.float32

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
                    normalize_weights,
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
                    normalize_weights,
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
                    normalize_weights,
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
                    normalize_weights,
                    <int64_t*>labels_ptr,
                    inertia_f64,
                )

    inertia = inertia_f32 if values_f32 else inertia_f64

    return labels, inertia


cdef _kmeans_predict_host_chunked(
    handle_t& handle,
    lib.KMeansParams& params,
    X,
    sample_weight,
    centers,
    int64_t batch_size,
):
    """Predict labels & total inertia for host-resident `X` in chunks.

    Streams chunks of `batch_size` host rows into a single reusable device
    buffer, runs the existing device-data predict on each chunk, and stitches
    the per-chunk labels into a single host (`numpy.ndarray`) result.

    Returns (`numpy.ndarray` of labels, `float` total inertia).
    """
    cdef int64_t n_rows = X.shape[0]
    cdef int64_t n_cols = X.shape[1]
    cdef int64_t cap = batch_size if batch_size > 0 else n_rows
    if cap > n_rows:
        cap = n_rows

    total_sw = float(sample_weight.sum())
    if total_sw > 0:
        sample_weight_scaled = (
            sample_weight.astype(X.dtype, copy=False) * (n_rows / total_sw)
        ).astype(X.dtype, copy=False)
    else:
        sample_weight_scaled = sample_weight

    # Reusable per-batch device buffers. Allocated once
    X_buf = cp.empty(shape=(cap, n_cols), dtype=X.dtype, order="C")
    sw_buf = cp.empty(shape=cap, dtype=X.dtype)

    labels_dtype = (
        np.int32
        if _kmeans_indices_i32(n_rows, n_cols)
        and _kmeans_indices_i32(centers.shape[0], n_cols)
        else np.int64
    )
    labels_host = np.empty(n_rows, dtype=labels_dtype)

    cdef int64_t start = 0
    cdef int64_t end = 0
    cdef int64_t n = 0
    total_inertia = 0.0
    while start < n_rows:
        end = start + cap
        if end > n_rows:
            end = n_rows
        n = end - start

        # Host -> device copy of this batch.
        X_buf[:n].set(X[start:end])
        sw_buf[:n].set(sample_weight_scaled[start:end])

        batch_labels, batch_inertia = _kmeans_predict(
            handle, params, X_buf[:n], sw_buf[:n], centers, False
        )
        labels_host[start:end] = cp.asnumpy(batch_labels)
        total_inertia += float(batch_inertia)

        start = end

    return labels_host, total_inertia


class KMeans(InteropMixin,
             ClusterMixin,
             CMajorInputTagMixin,
             Base):
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
    init : {'scalable-k-means++', 'k-means||', 'k-means++', 'random'} or an \
            ndarray (default = 'scalable-k-means++')

         - ``'scalable-k-means++'`` or ``'k-means||'``: Uses fast and stable
           scalable kmeans++ initialization. k-means++ is the constrained case of k-means||
           with `oversampling_factor=0`
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
    streaming_batch_size : int (default = 0)
        Number of samples to stream from host to device per GPU batch when
        fitting with host-resident inputs. When set to 0 (default), all
        samples are copied to device at once.
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
    _multi_gpu = False

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
            "streaming_batch_size",
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
        n_clusters=8,
        max_iter=300,
        tol=1e-4,
        verbose=False,
        random_state=None,
        init='scalable-k-means++',
        n_init="auto",
        oversampling_factor=2.0,
        max_samples_per_batch=1<<15,
        streaming_batch_size=0,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.oversampling_factor = oversampling_factor
        self.max_samples_per_batch = max_samples_per_batch
        self.streaming_batch_size = streaming_batch_size

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # Exposed to support sklearn's `get_feature_names_out`
        return self.n_clusters

    @generate_docstring()
    @reflect(reset="type")
    def fit(self, X, y=None, sample_weight=None, *, convert_dtype=True) -> "KMeans":
        """
        Compute k-means clustering with X.

        """
        streaming_batch_size = int(self.streaming_batch_size)
        if streaming_batch_size < 0:
            raise ValueError(
                f"streaming_batch_size must be >= 0, got "
                f"{streaming_batch_size}."
            )

        if streaming_batch_size > 0 and (self._multi_gpu or data_on_device):
            if self._multi_gpu:
                reason = "the multi-GPU KMeans fit path"
            else:
                reason = "device-resident inputs"
            raise ValueError(
                f"streaming_batch_size={streaming_batch_size} is only "
                f"supported for single-GPU fit on host-resident inputs; it is "
                f"not supported for {reason}. Either pass a host array (e.g. "
                f"numpy.ndarray, pandas.DataFrame) or set "
                f"streaming_batch_size=0."
            )

        use_host_path = streaming_batch_size > 0 and not data_on_device
        mem_type = "host" if use_host_path else "device"

        X, sample_weight = check_inputs(
            self,
            X,
            sample_weight=sample_weight,
            dtype=("float32", "float64"),
            convert_dtype=convert_dtype,
            order="C",
            mem_type=mem_type,
            reset=True,
        )
        data_on_device = hasattr(X, "__cuda_array_interface__")

        n_rows, n_cols = X.shape

        if sample_weight is None:
            if use_host_path:
                sample_weight = np.ones(shape=n_rows, dtype=X.dtype)
            else:
                sample_weight = cp.ones(shape=n_rows, dtype=X.dtype)

        if n_rows < self.n_clusters:
            raise ValueError(
                f"n_samples={n_rows} should be >= n_clusters={self.n_clusters}."
            )

        # Allocate output cluster_centers_
        if isinstance(self.init, str):
            centers = cp.zeros(
                shape=(self.n_clusters, n_cols), dtype=X.dtype, order="C",
            )
        else:
            # Initial array provided, coerce to device array and validate
            centers = check_array(
                self.init,
                order="C",
                dtype=X.dtype,
                convert_dtype=convert_dtype,
            ).copy()
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
        # XXX: multi-gpu uses handle attribute to manage comms
        handle = self.handle if self._multi_gpu else get_handle()
        cdef handle_t* handle_ = <handle_t *><size_t>handle.getHandle()
        cdef lib.KMeansParams params
        _kmeans_init_params(self, params)
        n_iter = _kmeans_fit(handle_[0], params, X, sample_weight, centers)
        if use_host_path:
            labels, inertia = _kmeans_predict_host_chunked(
                handle_[0], params, X, sample_weight, centers,
                streaming_batch_size,
            )
        else:
            labels, inertia = _kmeans_predict(
                handle_[0], params, X, sample_weight, centers,
            )
        handle.sync()

        # Store fitted attributes and return
        self.cluster_centers_ = CumlArray(data=centers)
        self.labels_ = CumlArray(data=labels)
        self.inertia_ = inertia
        self.n_iter_ = n_iter

        return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Cluster indexes',
                                       'shape': '(n_samples, 1)'})
    @reflect
    def fit_predict(self, X, y=None, sample_weight=None) -> CumlArray:
        """
        Compute cluster centers and predict cluster index for each sample.

        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def _predict_labels_inertia(self, X, convert_dtype=True, sample_weight=None):
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
        check_is_fitted(self)
        X, sample_weight = check_inputs(
            self,
            X,
            sample_weight=sample_weight,
            dtype=self.cluster_centers_.dtype,
            convert_dtype=convert_dtype,
            order="C",
        )
        if sample_weight is None:
            sample_weight = cp.ones(shape=X.shape[0], dtype=X.dtype)

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef lib.KMeansParams params
        _kmeans_init_params(self, params)

        labels, inertia = _kmeans_predict(
            handle_[0],
            params,
            X,
            sample_weight,
            self.cluster_centers_.to_output("cupy")
        )
        handle.sync()
        return labels, inertia

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Cluster indexes',
                                       'shape': '(n_samples, 1)'})
    @reflect
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
        return CumlArray(data=labels)

    @generate_docstring(return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Transformed data',
                                       'shape': '(n_samples, n_clusters)'})
    @reflect
    def transform(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Transform X to a cluster-distance space.

        """
        check_is_fitted(self)
        X = check_inputs(
            self,
            X,
            dtype=self.cluster_centers_.dtype,
            convert_dtype=convert_dtype,
            order="C",
        )

        cdef int64_t n_rows = X.shape[0]
        cdef int64_t n_cols = X.shape[1]
        cdef int64_t n_clusters = self.n_clusters

        # Stop-gap: the current C++/cuVS transform path does not preserve
        # int64 indexing end-to-end for the output matrix. Reject oversized
        # outputs here until transform supports int64 output indexing.
        if not _kmeans_indices_i32(n_rows, n_clusters):
            raise NotImplementedError(
                "KMeans.transform does not currently support output shapes "
                "that require int64 indexing. Got output shape "
                f"({n_rows}, {n_clusters})."
            )

        out = cp.zeros(
            shape=(n_rows, n_clusters), dtype=X.dtype, order="C",
        )

        cdef uintptr_t X_ptr = X.data.ptr
        cdef uintptr_t centers_ptr = self.cluster_centers_.ptr
        cdef uintptr_t out_ptr = out.data.ptr

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef lib.KMeansParams params
        _kmeans_init_params(self, params)

        cdef bool values_f32 = X.dtype == cp.float32
        cdef bool indices_i32 = _kmeans_indices_i32(n_rows, n_cols)

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
        handle.sync()
        return CumlArray(data=out)

    @generate_docstring(return_values={'name': 'score',
                                       'type': 'float',
                                       'description': 'Opposite of the value \
                                                        of X on the K-means \
                                                        objective.'})
    @run_in_internal_context
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
    @reflect
    def fit_transform(
        self, X, y=None, sample_weight=None, *, convert_dtype=False
    ) -> CumlArray:
        """
        Compute clustering and transform X to cluster-distance space.

        """
        self.fit(X, sample_weight=sample_weight)
        return self.transform(X, convert_dtype=convert_dtype)
