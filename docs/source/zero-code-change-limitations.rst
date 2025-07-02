Known Limitations
-----------------

General Limitations
~~~~~~~~~~~~~~~~~~~

The cuML Accelerator present in RAPIDS release 25.02.01 is a beta version, with
the following general limitations:

* Ingestion of lists of numbers by estimator functions is unsupported. Convert
  lists to structured formats (e.g., NumPy arrays or Pandas DataFrames) to
  ensure compatibility. This limitation will be removed in the next version of
  the cuML Accelerator.

* Labels provided as arrays of strings are not supported. Pre-encode string
  labels into numerical or categorical formats (e.g., using scikit-learn's
  LabelEncoder) prior to processing. This limitation will be removed in the
  next version of the cuML Accelerator.

* The accelerator is compatible with scikit-learn version 1.5 or higher. This
  compatibility ensures that cuML's implementation of scikit-learn compatible
  APIs works as expected.

* When running in Windows Subsystem for Linux 2 (WSL2), managed memory (unified
  memory) is not supported. This means that automatic memory management between
  host and device memory is not available. Users may need to be more careful
  about memory management and consider using the ``--disable-uvm`` flag if
  experiencing memory-related issues.

For notes on each algorithm, please refer to its specific section on this file.

Algorithm-Specific Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``umap.UMAP``
^^^^^^^^^^^^^

* Algorithm Limitations:
    * There may be cases where cuML's UMAP may not achieve the same level of quality as the reference implementation. The trustworthiness score can be used to assess to what extent the local structure is retained in embedding. The upcoming 25.04 and 25.06 will contain significant improvements to both performance and numerical accuracy for cuML's UMAP.
    * The following parameters are not supported : "low_memory", "angular_rp_forest", "transform_seed", "tqdm_kwds", "unique", "densmap", "dens_lambda", "dens_frac", "dens_var_shift", "output_dens", "disconnection_distance".
    * Parallelism during the optimization stage implies numerical imprecisions, which can lead to difference in the results between CPU and GPU in general.
    * Reproducibility with the use of a seed ("random_state" parameter) comes at the relative expense of performance.

* Distance Metrics:
    * Only the following metrics are supported : "l1", "cityblock", "taxicab", "manhattan", "euclidean", "l2", "sqeuclidean", "canberra", "minkowski", "chebyshev", "linf", "cosine", "correlation", "hellinger", "hamming", "jaccard".
    * Other metrics will trigger a CPU fallback, namely : "sokalsneath", "rogerstanimoto", "sokalmichener", "yule", "ll_dirichlet", "russellrao", "kulsinski", "dice", "wminkowski", "mahalanobis", "haversine".

* Embeddings initialization methods :
    * Only the following initialization methods are supported : "spectral" and "random".
    * Other initialization methods will trigger a CPU fallback, namely : "pca", "tswspectral".

While the exact numerical output for UMAP may differ from that obtained without cuml.accel,
we expect the output to be equivalent in the sense that the quality of results will be approximately as good or better
than that obtained without cuml.accel in most cases. A common measure of results quality for UMAP is the trustworthiness score.
You can obtain the trustworthiness by doing the following :

.. code-block:: python

    from umap import UMAP as refUMAP  #  with cuml.accel off
    from cuml.manifold import UMAP
    from cuml.metrics import trustworthiness

    n_neighbors = 15

    ref_model = refUMAP(n_neighbors=n_neighbors)
    ref_embeddings = ref_model.fit_transform(X)

    model = UMAP(n_neighbors=n_neighbors)
    embeddings = model.fit_transform(X)

    ref_score = trustworthiness(X, ref_embeddings, n_neighbors=n_neighbors)
    score = trustworthiness(X, embeddings, n_neighbors=n_neighbors)

    tol = 0.1
    assert score >= (ref_score - tol)


``hdbscan.HDBSCAN``
^^^^^^^^^^^^^^^^^^^
* Algorithm Limitations:
    * GPU HDBSCAN uses a parallel MST implementation, which means the results are not deterministic when there are duplicates in the mutual reachability graph.
    * CPU HDBSCAN offers many choices of different algorithms whereas GPU HDBSCAN uses a single implementation. Everything except `algorithm="auto"` will fallback to the CPU.
    * GPU HDBSCAN supports all functions in the CPU `hdbscan.prediction` module except `approximate_predict_score`.
    * CPU HDBSCAN offers a `hdbscan.branches` module that GPU HDBCAN does not.

* Distance Metrics
    * Only euclidean distance is GPU-accelerated.
    * precompute distance matrix is not supported on GPU.
    * Custom metric functions (callable metrics) are not supported on GPU.

* Learned Attributes Limitations:
    * GPU HDBSCAN does not learn attributes `branch_detection_data_`, `examplers_`, and `relative_validity_`.


``sklearn.cluster.KMeans``
^^^^^^^^^^^^^^^^^^^^^^^^^^

The default initialization algorithm used by ``cuml.accel`` is similar, but different.
``cuml.accel`` uses the ``"scalable-k-means++"`` algorithm, for more details refer to
:class:`cuml.KMeans`.

This means that the ``cluster_centers_`` attribute will not be exactly the same as for
the scikit-learn implementation. The ID of each cluster (``labels_`` attribute) might
change, this means samples labelled to be in cluster zero for scikit-learn might be
labelled to be in cluster one for ``cuml.accel``. The ``inertia_`` attribute might
differ as well if different cluster centers are used. The algorithm might converge
in a different number of iterations, this means the ``n_iter_`` attribute might differ.

To check that the resulting trained estimator is equivalent to the scikit-learn
estimator, you can evaluate the similarity of the clustering result on samples
not used to train the estimator. Both ``adjusted_rand_score`` and ``adjusted_mutual_info_score``
give a single score that should be above ``0.9``. For low dimensional data you
can also visually inspect the resulting cluster assignments.

``cuml.accel`` will not fall back to scikit-learn.


``sklearn.cluster.DBSCAN``
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``DBSCAN`` implementation used by ``cuml.accel`` uses a brute force algorithm
for the epsilon-neighborhood search. By default scikit-learn determines the
algorithm to use based on the shape of the data and which metric is used. All algorithms
are exact, this means the choice is a question of computational efficiency.

To check that the resulting trained estimator is equivalent to the scikit-learn
estimator, you can evaluate the similarity of the clustering result on samples
not used to train the estimator. Both ``adjusted_rand_score`` and ``adjusted_mutual_info_score``
give a single score that should be above ``0.9``. For low dimensional data you
can also visually inspect the resulting cluster assignments.

``cuml.accel`` will fallback to scikit-learn for the following parameters:

* The ``"manhattan"``, ``"chebyshev"`` and ``"minkowski"`` metrics.
* The ``"ball_tree"`` and ``"kd_tree"`` algorithms.


``sklearn.decomposition.PCA``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``PCA`` implementation used by ``cuml.accel`` uses different SVD solvers
than the ones in Scikit-Learn, which may result in numeric differences in the
``components_`` and ``explained_variance_`` values. These differences should be
small for ``svd_solver`` values of ``"auto"``, ``"full"``, or ``"arpack"``, but
may be larger for randomized or less-numerically-stable solvers like
``"randomized"`` or ``"covariance_eigh"``.

Likewise, note that the implementation in ``cuml.accel`` currently may result
in some of the vectors in ``components_`` having inverted signs. This result is
not incorrect, but can make it harder to do direct numeric comparisons without
first normalizing the signs. One common way of handling this is by normalizing
the first non-zero values in each vector to be positive. You might find the
following ``numpy`` function useful for this.

.. code-block:: python

    import numpy as np

    def normalize(components):
        """Normalize the sign of components for easier numeric comparison"""
        nonzero = components != 0
        inds = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), 0)[:, None]
        first_nonzero = np.take_along_axis(components, inds, 1)
        return np.sign(first_nonzero) * components

For more algorithmic details, see :class:`cuml.PCA`.

* Algorithm Limitation:
    * ``n_components="mle"`` will fallback to Scikit-Learn.
    * Parameters for the ``"randomized"`` solver like ``random_state``,
      ``n_oversamples``, ``power_iteration_normalizer`` are ignored.

``sklearn.decomposition.TruncatedSVD``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``TruncatedSVD`` implementation used by ``cuml.accel`` uses different SVD
solvers than the ones in Scikit-Learn, which may result in numeric differences
in the ``components_`` and ``explained_variance_`` values. These differences
should be small for ``algorithm="arpack"``, but may be larger for
``algorithm="randomized"``.

Likewise, note that the implementation in ``cuml.accel`` currently may result
in some of the vectors in ``components_`` having inverted signs. This result is
not incorrect, but can make it harder to do direct numeric comparisons without
first normalizing the signs. One common way of handling this is by normalizing
the first non-zero values in each vector to be positive. You might find the
following ``numpy`` function useful for this.

.. code-block:: python

    import numpy as np

    def normalize(components):
        """Normalize the sign of components for easier numeric comparison"""
        nonzero = components != 0
        inds = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), 0)[:, None]
        first_nonzero = np.take_along_axis(components, inds, 1)
        return np.sign(first_nonzero) * components

For more algorithmic details, see :class:`cuml.TruncatedSVD`.

* Algorithm Limitation:
    * Parameters for the ``"randomized"`` solver like ``random_state``,
      ``n_oversamples``, ``power_iteration_normalizer`` are ignored.

``sklearn.ensemble.RandomForestClassifier`` / ``sklearn.ensemble.RandomForestRegressor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The random forest in ``cuml.accel`` uses a different algorithm to find tree node splits.
When choosing split thresholds, the ``cuml.accel`` random forest considers only quantiles
as threshold candidates, whereas the scikit-learn random forest considers all possible
feature values from the training data. As a result, the ``cuml.accel`` random forest
may choose different split thresholds from the scikit-learn counterpart, leading to
different tree structure. Nevertheless, we expect the output to be
*equivalent* in the sense that the quality of results will be approximately
as good or better than that obtained without ``cuml.accel``. Common
measures of quality for random forests include RMSE (Root Mean Squared Error, for
regression) and Log Loss (for classification). You can use functions from the
``sklearn.metrics`` module to obtain these measures.

Some parameters have limited support:

* ``max_samples`` must be float, not integer.

The following parameters are not supported and will trigger a CPU fallback:

* ``min_weight_fraction_leaf``
* ``monotonic_cst``
* ``ccp_alpha``
* ``class_weight``
* ``warm_start``
* ``oob_score``

The following values for ``criterion`` will trigger a CPU fallback:

* ``log_loss``
* ``friedman_mse``

``sklearn.kernel_ridge.KernelRidge``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``KernelRidge`` results should be almost identical to those of Scikit-Learn
when running with ``cuml.accel`` enabled. In particular, the fitted
``dual_coef_`` should be close enough that they may be compared via
``np.allclose``.

It currently has the following limitations:

* Sparse inputs are not currently supported and will fallback to CPU.

``sklearn.linear_model.LinearRegression``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linear Regression is one of the simpler estimators, where functionality and results
between cuML.accel and Scikit-learn will be quite close, with the following
limitations:

* multi-output target is not currently supported.
* ``positive`` parameter to force positive coefficients is not currently supported,
  and cuml.accel will not accelerate Linear Regression if the parameter is set to
  ``True``
* cuML's Linear Regression only implements dense inputs currently, so cuml.accel offers no
  acceleration for sparse inputs to model training.

Another important consideration is that, unlike more complex models, like manifold
or clustering algorithms, linear models are quite efficient and fast to run. Even on larger
datasets, the execution time can many times be measured in seconds, so taking that
into consideration will be important for example when evaluating results as seen
in `Zero Code Change Benchmarks <zero-code-change-benchmarks.rst>`_

``sklearn.linear_model.LogisticRegression``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

cuML's Logistic Regression main difference from Scikit-learn is the solver that is
used to train the model. cuML using a Quasi-Newton set of solvers (L-BFGS or OWL-QN)
which themselves have algorithmic differences from the solvers of sklearn. Even then,
the results should be comparible between implementations.

* Regardless of which `solver` the original Logist Regression model uses, cuml.accel
  will use `qn` as described above.

``sklearn.linear_model.ElasticNet``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to Linear Regression, Elastic Net has the following limitations:

* ``positive`` parameter to force positive coefficients is not currently supported,
  and cuml.accel will not accelerate Elastic Net if the parameter is set to
  ``True``
* ``warm_start`` parameter is not supported for GPU acceleration.
* ``precompute`` parameter is not supported.
* cuML's ElasticNet only implements dense inputs currently, so cuml.accel offers no
  acceleration for sparse inputs to model training.

``sklearn.linear_model.Ridge``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to Linear Regression, Elastic Net has the following limitations:

* ``positive`` parameter to force positive coefficients is not currently supported,
  and cuml.accel will not accelerate Elastic Net if the parameter is set to
  ``True``
* ``solver`` all solver parameter values are translated to `eig` to use the
  eigendecomposition of the covariance matrix.
* cuML's Ridge only implements dense inputs currently, so cuml.accel offers no
  acceleration for sparse inputs to model training.

``sklearn.linear_model.Lasso``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``precompute`` parameter is not supported.
* cuML's Lasso only implements dense inputs currently, so cuml.accel offers no
  acceleration for sparse inputs to model training.


``sklearn.manifold.TSNE``
^^^^^^^^^^^^^^^^^^^^^^^^^

* Algorithm Limitations:
    * The "learning_rate" parameter cannot be used with value "auto", and will default to 200.0.


* Distance Metrics:
    * Only the following metrics are supported : "l1", "cityblock", "manhattan", "euclidean", "l2", "sqeuclidean", "minkowski", "chebyshev", "cosine", "correlation".
    * The "precomputed" option, or the use of function as metric is not supported


While the exact numerical output for TSNE may differ from that obtained without cuml.accel,
we expect the output to be equivalent in the sense that the quality of results will be approximately as good or better
than that obtained without cuml.accel in most cases. Common measure of results quality for TSNE are the KL divergence and the trustworthiness score.
You can obtain it by doing the following :

.. code-block:: python

    from sklearn.manifold import TSNE as refTSNE  #  with cuml.accel off
    from cuml.manifold import TSNE
    from cuml.metrics import trustworthiness

    n_neighbors = 90

    ref_model = refTSNE() #  with perplexity == 30.0
    ref_embeddings = ref_model.fit_transform(X)

    model = TSNE(n_neighbors=n_neighbors)
    embeddings = model.fit_transform(X)

    ref_score = trustworthiness(X, ref_embeddings, n_neighbors=n_neighbors)
    score = trustworthiness(X, embeddings, n_neighbors=n_neighbors)

    tol = 0.1
    assert score >= (ref_score - tol)
    assert model.kl_divergence_ <= ref_model.kl_divergence_ + tol


``sklearn.neighbors.NearestNeighbors``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Algorithm Limitations:
    * The "kd_tree" and "ball_tree" algorithms are not implemented in CUDA. When specified, the implementation will automatically fall back to using the "brute" force algorithm.

* Distance Metrics:
    * Only Minkowski-family metrics (euclidean, manhattan, minkowski) and cosine similarity are GPU-accelerated
    * Not all metrics are supported for algorithms.
    * The "mahalanobis" metric is not supported on GPU and will trigger a fallback to CPU implementation.
    * The "nan_euclidean" metric for handling missing values is not supported on GPU.
    * Custom metric functions (callable metrics) are not supported on GPU.

* Other Limitations:
    * Only the "uniform" weighting strategy is supported. Other weighting schemes will cause fallback to CPU
    * The "radius" parameter for radius-based neighbor searches is not implemented and will be ignored

``sklearn.neighbors.KNeighborsClassifier``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Algorithm Limitations:
    * The "kd_tree" and "ball_tree" algorithms are not implemented in CUDA. When specified, the implementation will automatically fall back to using the "brute" force algorithm.

* Distance Metrics:
    * Only Minkowski-family metrics (euclidean, manhattan, minkowski) and cosine similarity are GPU-accelerated
    * Not all metrics are supported for algorithms.
    * The "mahalanobis" metric is not supported on GPU and will trigger a fallback to CPU implementation.
    * The "nan_euclidean" metric for handling missing values is not supported on GPU.
    * Custom metric functions (callable metrics) are not supported on GPU.

* Other Limitations:
    * Only the "uniform" weighting strategy is supported for vote counting.
    * Distance-based weights ("distance" option) will trigger CPU fallback.
    * Custom weight functions are not supported on GPU.

``sklearn.neighbors.KNeighborsRegressor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Algorithm Limitations:
    * The "kd_tree" and "ball_tree" algorithms are not implemented in CUDA. When specified, the implementation will automatically fall back to using the "brute" force algorithm.

* Distance Metrics:
    * Only Minkowski-family metrics (euclidean, manhattan, minkowski) and cosine similarity are GPU-accelerated
    * Not all metrics are supported for algorithms.
    * The "mahalanobis" metric is not supported on GPU and will trigger a fallback to CPU implementation.
    * The "nan_euclidean" metric for handling missing values is not supported on GPU.
    * Custom metric functions (callable metrics) are not supported on GPU.

* Regression-Specific Limitations:
    * Only the "uniform" weighting strategy is supported for prediction averaging.
    * Distance-based prediction weights ("distance" option) will trigger CPU fallback.
    * Custom weight functions are not supported on GPU.

``sklearn.svm.SVC``
^^^^^^^^^^^^^^^^^^^

The ``SVC`` implementation in ``cuml.accel`` uses a different solver than that
in scikit-learn. As such, you shouldn't expect equivalent support vectors or
coefficients. To compare results you should compare the performance of the
model using ``model.score`` or ``sklearn.metrics.accuracy_score``.

* Algorithm Limitations:
    * ``probability=True`` will fallback to scikit-learn
    * ``kernel="precomputed"`` or callable kernels will fallback to scikit-learn
    * Multiclass classification will fallback to scikit-learn
    * Sparse inputs will fallback to scikit-learn

``sklearn.svm.SVR``
^^^^^^^^^^^^^^^^^^^

The ``SVR`` implementation in ``cuml.accel`` uses a different solver than that
in scikit-learn. As such, you shouldn't expect equivalent support vectors or
coefficients. To compare results you should compare the performance of the
model using ``model.score`` or ``sklearn.metrics.r2_score``.

* Algorithm Limitations:
    * ``kernel="precomputed"`` or callable kernels will fallback to scikit-learn
    * Sparse inputs will fallback to scikit-learn
