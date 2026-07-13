Supported Functionality
=======================

``cuml.accel`` can accelerate the estimators listed below. This page combines
the complete support inventory with estimator-specific CPU fallback conditions,
missing fitted attributes, and behavioral differences. Use
:doc:`understanding-acceleration` to see which operations in your workload ran
on the GPU and why others fell back to the CPU.

Estimators not listed on the support page remain on the CPU. Listed estimators
may also fall back for particular methods, hyperparameters, input types, or
dependency versions, as documented below. These fallbacks should be
transparent and preserve the original workflow.

Other known differences include:

- Missing fitted attributes. ``cuml.accel`` does not currently generate the
  full set of fitted attributes that ``sklearn`` does. In *most* cases this is
  not a problem, the missing attributes are usually minor things like
  ``n_iters_`` that are useful for inspecting a model fit but not necessary for
  inference. Like unsupported parameters, missing fitted attributes are
  documented in algorithm-specific sections below.

- Differences between fitted models. The algorithms and implementations used in
  ``cuml`` naturally differ from those used in ``sklearn``, this may result in
  differences between fit models. This is to be expected. To compare results
  between models fit with ``cuml.accel`` and those fit without, you should
  compare the model *quality* (using e.g. ``model.score``) and not the numeric
  values of the fitted coefficients.

None of the above should result in bugs (exceptions, failures, poor model
quality, ...). If you find a case that errors or results in a model with
measurably worse quality when run under ``cuml.accel``, please `open an issue`_.

General notes:

- Performance improvements will be most apparent when running on larger data.
  On very small datasets you might see only a small speedup (or even
  potentially a slowdown).

- The accelerator is tested to be compatible with Scikit-Learn versions 1.6
  through 1.9. This ensures that cuML's implementation of Scikit-Learn
  compatible APIs works as expected.

- Some estimators are accelerated using Scikit-Learn's experimental
  `array-api`_ support. These estimators are only accelerated by ``cuml.accel``
  when running with Scikit-Learn versions >= 1.8. Running with an older
  version of Scikit-Learn will use an unaccelerated estimator.

- Error and warning messages and formats may differ from scikit-learn. Some
  errors might present as C++ stacktraces instead of python errors.


.. _sklearn-limitations:

Scikit-Learn
------------

sklearn.cluster
~~~~~~~~~~~~~~~

The algorithms used in ``cuml`` differ from those in ``sklearn``. As such, you
shouldn't expect the fitted attributes (e.g. ``labels_``) to numerically match
an estimator fitted without ``cuml.accel``.

To compare results between estimators, we recommend comparing scores like
``sklearn.metrics.adjusted_rand_score`` or
``sklearn.metrics.adjusted_mutual_info_score``. For low dimensional data you
can also visually inspect the resulting cluster assignments.

.. dropdown:: ``KMeans``
   :name: kmeans

   ``KMeans`` will fall back to CPU in the following cases:

   - If a callable ``init`` is provided.
   - If ``X`` is sparse.


.. dropdown:: ``SpectralClustering``
   :name: spectralclustering

   ``SpectralClustering`` will fall back to CPU in the following cases:

   - If ``assign_labels`` is not ``"kmeans"``.
   - If ``affinity`` is not ``"nearest_neighbors"`` or ``"precomputed"``.
     Note that the default value of ``affinity`` in scikit-learn is ``"rbf"``,
     which is not GPU-accelerated.
   - If ``X`` is sparse.

   The following fitted attributes are currently not computed:

   - ``affinity_matrix_``


.. dropdown:: ``DBSCAN``
   :name: dbscan

   ``DBSCAN`` will fall back to CPU in the following cases:

   - If ``algorithm`` isn't ``"auto"`` or ``"brute"``.
   - If ``metric`` isn't one of the supported metrics (``"l2"``, ``"euclidean"``, ``"cosine"``, ``"precomputed"``).
   - If ``X`` is sparse.

   Additional notes:

   - ONNX export via ``skl2onnx`` is not supported for this estimator.


sklearn.covariance
~~~~~~~~~~~~~~~~~~

.. dropdown:: ``EmpiricalCovariance``
   :name: empiricalcovariance

   ``EmpiricalCovariance`` has no known estimator-specific ``cuml.accel``
   limitations.

.. dropdown:: ``LedoitWolf``
   :name: ledoitwolf

   ``LedoitWolf`` has no known estimator-specific ``cuml.accel`` limitations.


sklearn.decomposition
~~~~~~~~~~~~~~~~~~~~~

The ``sklearn.decomposition`` implementations used by ``cuml.accel`` uses
different SVD solvers than the ones in Scikit-Learn, which may result in
numeric differences in the ``components_`` and ``explained_variance_`` values.
These differences should be small for most algorithms, but may be larger for
randomized or less-numerically-stable solvers like ``"randomized"`` or
``"covariance_eigh"``.

.. dropdown:: ``PCA``
   :name: pca

   ``PCA`` will fall back to CPU in the following cases:

   - If ``n_components=0``.
   - If ``n_components="mle"``.

   Additional notes:

   - Parameters for the ``"randomized"`` solver like ``random_state``,
     ``n_oversamples``, ``power_iteration_normalizer`` are ignored.


.. dropdown:: ``IncrementalPCA``
   :name: incrementalpca

   ``IncrementalPCA`` has no known estimator-specific ``cuml.accel`` limitations.

   Additional notes:

   - ``partial_fit`` does not support sparse input. This matches scikit-learn;
     use ``fit`` for sparse input or provide dense batches to ``partial_fit``.


.. dropdown:: ``TruncatedSVD``
   :name: truncatedsvd

   ``TruncatedSVD`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.


   Additional notes:

   - Parameters for the ``"randomized"`` solver like ``random_state``,
     ``n_oversamples``, ``power_iteration_normalizer`` are ignored.


sklearn.ensemble
~~~~~~~~~~~~~~~~

The random forest implementation used by ``cuml.accel`` algorithmically
differs from the one in ``sklearn``. As such, you
shouldn't expect the fitted attributes (e.g. ``estimators_``) to numerically match
an estimator fitted without ``cuml.accel``.

To compare results between estimators, we recommend comparing scores like
``sklearn.metrics.root_mean_squared_error`` (for regression) or
``sklearn.metrics.log_loss`` (for classification).

.. dropdown:: ``RandomForestClassifier``
   :name: randomforestclassifier

   ``RandomForestClassifier`` will fall back to CPU in the following cases:

   - If ``criterion`` is ``"log_loss"``.
   - If ``oob_score`` is a callable.
   - If ``warm_start=True``.
   - If ``monotonic_cst`` is not ``None``.
   - If ``max_values`` is an integer.
   - If ``min_weight_fraction_leaf`` is not ``0``.
   - If ``ccp_alpha`` is not ``0``.
   - If ``class_weight`` is not ``None``.
   - If ``sample_weight`` is passed to ``fit`` or ``score``.
   - If ``X`` is sparse.
   - If ``X`` contains missing values (represented as ``NaN``).
   - If ``y`` is a multi-output target.


.. dropdown:: ``RandomForestRegressor``
   :name: randomforestregressor

   ``RandomForestRegressor`` will fall back to CPU in the following cases:

   - If ``criterion`` is ``"absolute_error"`` or ``"friedman_mse"``.
   - If ``oob_score`` is a callable.
   - If ``warm_start=True``.
   - If ``monotonic_cst`` is not ``None``.
   - If ``max_values`` is an integer.
   - If ``min_weight_fraction_leaf`` is not ``0``.
   - If ``ccp_alpha`` is not ``0``.
   - If ``sample_weight`` is passed to ``fit`` or ``score``.
   - If ``X`` is sparse.
   - If ``X`` contains missing values (represented as ``NaN``).
   - If ``y`` is a multi-output target.


sklearn.kernel_ridge
~~~~~~~~~~~~~~~~~~~~

.. dropdown:: ``KernelRidge``
   :name: kernelridge

   ``KernelRidge`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - If ``kernel`` is not a string.

   ``KernelRidge`` results should be almost identical to those of Scikit-Learn
   when running with ``cuml.accel`` enabled. In particular, the fitted
   ``dual_coef_`` should be close enough that they may be compared via
   ``np.allclose``.


sklearn.linear_model
~~~~~~~~~~~~~~~~~~~~

The linear model solvers used by ``cuml.accel`` differ from those used in
``sklearn``. As such, you shouldn't expect the fitted attributes (e.g.
``coef_``) to numerically match an estimator fitted without ``cuml.accel``. For
some estimators (e.g. ``LinearRegression``) you might get a close match, but
for others there may larger numeric differences.

To compare results between estimators, we recommend comparing model quality
scores like ``sklearn.metrics.r2_score`` (for regression) or
``sklearn.metrics.accuracy_score`` (for classification).

.. dropdown:: ``LinearRegression``
   :name: linearregression

   ``LinearRegression`` will fall back to CPU in the following cases:

   - If ``positive=True``.

   Additionally, the following fitted attributes are currently not computed:

   - ``rank_``
   - ``singular_``


.. dropdown:: ``LogisticRegression``
   :name: logisticregression

   ``LogisticRegression`` will fall back to CPU in the following cases:

   - If ``warm_start=True``.
   - If ``intercept_scaling`` is not ``1``.
   - If the deprecated ``multi_class`` parameter is used.
   - If a callback is configured with ``set_callbacks``.


.. dropdown:: ``ElasticNet``
   :name: elasticnet

   ``ElasticNet`` will fall back to CPU in the following cases:

   - If ``positive=True``.
   - If ``warm_start=True``.
   - If ``precompute`` is not ``False``.

   Additionally, the following fitted attributes are currently not computed:

   - ``dual_gap_``


.. dropdown:: ``Ridge``
   :name: ridge

   ``Ridge`` will fall back to CPU in the following cases:

   - If ``positive=True`` or ``solver="lbfgs"``.


.. dropdown:: ``Lasso``
   :name: lasso

   ``Lasso`` will fall back to CPU in the following cases:

   - If ``positive=True``.
   - If ``warm_start=True``.
   - If ``precompute`` is not ``False``.

   Additionally, the following fitted attributes are currently not computed:

   - ``dual_gap_``


sklearn.manifold
~~~~~~~~~~~~~~~~

.. dropdown:: ``TSNE``
   :name: tsne

   ``TSNE`` will fall back to CPU in the following cases:

   - If ``n_components`` is not ``2``.
   - If ``init`` is an array.
   - If ``init='pca'`` and X is sparse
   - If ``metric`` isn't one of the supported metrics ( ``"l2"``, ``"euclidean"``,
     ``"sqeuclidean"``, ``"cityblock"``, ``"l1"``, ``"manhattan"``,
     ``"minkowski"``, ``"chebyshev"``, ``"cosine"``, ``"correlation"``).

   Additional notes:

   - Even with a ``random_state``, the TSNE implementation used by ``cuml.accel``
     isn't completely deterministic.
   - ONNX export via ``skl2onnx`` is not supported for this estimator.

   While the exact numerical output for TSNE may differ from that obtained without
   ``cuml.accel``, we expect the *quality* of results will be approximately as
   good in most cases. Beyond comparing the visual representation, you may find
   comparing the trustworthiness score (computed via
   ``sklearn.manifold.trustworthiness``) or the ``kl_divergence_`` fitted
   attribute useful.


.. dropdown:: ``SpectralEmbedding``
   :name: spectralembedding

   ``SpectralEmbedding`` will fall back to CPU in the following cases:

   - If ``affinity`` is not ``"nearest_neighbors"`` or ``"precomputed"``.
   - If ``X`` is sparse.
   - If ``X`` has only 1 feature.


   The following fitted attributes are currently not computed:

   - ``affinity_matrix_``

   Additional notes:

   - ONNX export via ``skl2onnx`` is not supported for this estimator.


sklearn.neighbors
~~~~~~~~~~~~~~~~~

.. dropdown:: ``NearestNeighbors``
   :name: nearestneighbors

   ``NearestNeighbors`` will fall back to CPU in the following cases:

   - If ``metric`` is not one of the supported metrics ( ``"l2"``,
     ``"euclidean"``, ``"l1"``, ``"cityblock"``, ``"manhattan"``, ``"taxicab"``,
     ``"canberra"``, ``"minkowski"``, ``"lp"``, ``"chebyshev"``, ``"linf"``,
     ``"jensenshannon"``, ``"cosine"``, ``"correlation"``, ``"inner_product"``,
     ``"sqeuclidean"``, ``"haversine"``).

   Additional notes:

   - The ``algorithm`` parameter is ignored, the GPU accelerated ``"brute"``
     implementation in cuml will always be used.

   - The ``radius_neighbors`` method isn't implemented in cuml and will always
     fall back to CPU.

   - ONNX export via ``skl2onnx`` is not supported for this estimator.


.. dropdown:: ``KNeighborsClassifier``
   :name: kneighborsclassifier

   ``KNeighborsClassifier`` will fall back to CPU in the following cases:

   - If ``metric`` is not one of the supported metrics ( ``"l2"``,
     ``"euclidean"``, ``"l1"``, ``"cityblock"``, ``"manhattan"``, ``"taxicab"``,
     ``"canberra"``, ``"minkowski"``, ``"lp"``, ``"chebyshev"``, ``"linf"``,
     ``"jensenshannon"``, ``"cosine"``, ``"correlation"``, ``"inner_product"``,
     ``"sqeuclidean"``, ``"haversine"``).

   Additional notes:

   - The ``algorithm`` parameter is ignored, the GPU accelerated ``"brute"``
     implementation in cuml will always be used.


.. dropdown:: ``KNeighborsRegressor``
   :name: kneighborsregressor

   ``KNeighborsRegressor`` will fall back to CPU in the following cases:

   - If ``metric`` is not one of the supported metrics ( ``"l2"``,
     ``"euclidean"``, ``"l1"``, ``"cityblock"``, ``"manhattan"``, ``"taxicab"``,
     ``"canberra"``, ``"minkowski"``, ``"lp"``, ``"chebyshev"``, ``"linf"``,
     ``"jensenshannon"``, ``"cosine"``, ``"correlation"``, ``"inner_product"``,
     ``"sqeuclidean"``, ``"haversine"``).

   Additional notes:

   - The ``algorithm`` parameter is ignored, the GPU accelerated ``"brute"``
     implementation in cuml will always be used.


.. dropdown:: ``KernelDensity``
   :name: kerneldensity

   ``KernelDensity`` will fall back to CPU in the following cases:

   - If ``metric`` is not one of the supported metrics (``"cityblock"``,
     ``"cosine"``, ``"euclidean"``, ``"l1"``, ``"l2"``, ``"manhattan"``,
     ``"sqeuclidean"``, ``"canberra"``, ``"chebyshev"``, ``"minkowski"``,
     ``"hellinger"``, ``"correlation"``, ``"jensenshannon"``, ``"hamming"``,
     ``"kldivergence"``, ``"russellrao"``, ``"nan_euclidean"``).

   Additional notes:

   - The ``algorithm``, ``atol``, ``rtol``, ``breadth_first``, and ``leaf_size``
     parameters are ignored. The GPU accelerated pairwise brute-force
     implementation in cuml will always be used.


sklearn.preprocessing
~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: ``StandardScaler``
   :name: standardscaler

   ``StandardScaler`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - When run on scikit-learn < 1.8.
   - If a callback is configured with ``set_callbacks``.


.. dropdown:: ``MinMaxScaler``
   :name: minmaxscaler

   ``MinMaxScaler`` will fall back to CPU in the following cases:

   - When run on scikit-learn < 1.8.


.. dropdown:: ``MaxAbsScaler``
   :name: maxabsscaler

   ``MaxAbsScaler`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - When run on scikit-learn < 1.8.


.. dropdown:: ``PolynomialFeatures``
   :name: polynomialfeatures

   ``PolynomialFeatures`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - If ``order`` is ``"F"``.
   - When run on scikit-learn < 1.8.


.. dropdown:: ``LabelEncoder``
   :name: labelencoder

   ``LabelEncoder`` has no known estimator-specific ``cuml.accel`` limitations.


.. dropdown:: ``LabelBinarizer``
   :name: labelbinarizer

   ``LabelBinarizer`` has no known estimator-specific ``cuml.accel`` limitations.


.. dropdown:: ``TargetEncoder``
   :name: targetencoder

   ``TargetEncoder`` will fall back to CPU in the following cases:

   - If ``categories`` is not ``"auto"``.
   - If ``y`` is a multiclass target.
   - If ``random_state`` is a ``numpy.random.RandomState`` object (integer seeds
     work fine).

   Additional notes:

   - cuML and sklearn use different cross-validation fold assignment strategies
     during ``fit_transform``. Both are valid target encoding implementations, but
     samples are assigned to different folds, resulting in different leave-fold-out
     encodings for training data. The ``transform`` method on test data produces
     equivalent results since it uses global statistics computed from all training
     samples.


sklearn.svm
~~~~~~~~~~~

The SVM used by ``cuml.accel`` differ from those used in ``sklearn``. As such,
you shouldn't expect the fitted attributes (e.g. ``coef_`` or
``support_vectors_``) to numerically match an estimator fitted without
``cuml.accel``.

To compare results between estimators, we recommend comparing model quality
scores like ``sklearn.metrics.r2_score`` (for regression) or
``sklearn.metrics.accuracy_score`` (for classification).

.. dropdown:: ``SVC``
   :name: svc

   ``SVC`` will fall back to CPU in the following cases:

   - If ``kernel="precomputed"`` or is a callable.
   - If ``y`` is multiclass.
   - If ``probability=True``. The ``probability`` parameter is deprecated in
     ``scikit-learn>=1.9``, as well as in ``cuml>=26.06``. We recommend using
     wrapping ``SVC`` with ``sklearn.calibration.CalibratedClassifierCV`` like
     ``CalibratedClassifierCV(SVC(), ensemble=False)`` instead. This will be
     supported across ``scikit-learn`` versions, and won't require CPU fallback.


.. dropdown:: ``SVR``
   :name: svr

   ``SVR`` will fall back to CPU in the following cases:

   - If ``kernel="precomputed"`` or is a callable.


.. dropdown:: ``LinearSVC``
   :name: linearsvc

   ``LinearSVC`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - If ``intercept_scaling`` is not ``1``.
   - If ``multi_class`` is not ``"ovr"``.

   Additional notes:

   - Use of sample weights may not produce exactly equivalent results when
     compared to replicating data according to weights.


.. dropdown:: ``LinearSVR``
   :name: linearsvr

   ``LinearSVR`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - If ``intercept_scaling`` is not ``1``.

   Additional notes:

   - Use of sample weights may not produce exactly equivalent results when
     compared to replicating data according to weights.


UMAP
----

.. dropdown:: ``UMAP``
   :name: umap-limitations

   ``UMAP`` will fall back to CPU in the following cases:

   - If ``init`` is not ``"random"`` or ``"spectral"``.
   - If ``metric`` is not one of the supported metrics (``"l1"``, ``"cityblock"``,
     ``"taxicab"``, ``"manhattan"``, ``"euclidean"``, ``"l2"``, ``"sqeuclidean"``,
     ``"canberra"``, ``"minkowski"``, ``"chebyshev"``, ``"linf"``, ``"cosine"``,
     ``"correlation"``, ``"hellinger"``, ``"hamming"``, ``"jaccard"``).
   - If ``target_metric`` is not one of the supported metrics (``"categorical"``,
     ``"l2"``, ``"euclidean"``).
   - If ``unique=True``.
   - If ``densmap=True``.
   - If ``ensure_all_finite`` is not ``True``.

   Additional notes:

   - Reproducibility with the use of a seed (the ``random_state`` parameter) comes
     at the relative expense of performance.

   - Parallelism during the optimization stage implies numerical imprecisions,
     which can lead to difference in the results between CPU and GPU in general.

   - ONNX export via ``skl2onnx`` is not supported for this estimator.

   - We have observed compatibility isuess with UMAP for numba versions 0.62.0 and
     above. For best stability, we recommend using numba versions earlier than
     0.62.0 when accelerating UMAP with cuml.accel.

   While the exact numerical output for UMAP may differ from that obtained without
   ``cuml.accel``, we expect the *quality* of results will be approximately as
   good in most cases. Beyond comparing the visual representation, you may find
   comparing the trustworthiness score (computed via
   ``sklearn.manifold.trustworthiness``) useful.


HDBSCAN
-------

.. dropdown:: ``HDBSCAN``
   :name: hdbscan-limitations

   ``HDBSCAN`` will fall back to CPU in the following cases:

   - If ``metric`` is not ``"l2"`` or ``"euclidean"``.
   - If a ``memory`` location is configured.
   - If ``match_reference_implementation=True``.
   - If ``branch_detection_data=True``.

   Additionally, the following fitted attributes are currently not computed:

   - ``exemplars_``
   - ``outlier_scores_``
   - ``relative_validity_``

   Additional notes:

   - The ``HDBSCAN`` in ``cuml`` uses a parallel MST implementation, which means
     the results are not deterministic when there are duplicates in the mutual
     reachability graph.
   - ONNX export via ``skl2onnx`` is not supported for this estimator.


.. _open an issue: https://github.com/rapidsai/cuml/issues
.. _array-api: https://scikit-learn.org/stable/modules/array_api.html
