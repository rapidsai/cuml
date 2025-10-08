Limitations
===========

The ``cuml.accel`` zero code change accelerator is currently a beta feature. As
such, it has a number of known limitations and bugs. The team is working to
address these, and expect the number of limitations to reduce with every
release.

These limitations fall into a few categories:

- Estimators that are fully unaccelerated. For example, while we currently
  provide GPU acceleration for models like ``sklearn.linear_model.Ridge``, we
  don't accelerate other models like ``sklearn.linear_model.BayesianRidge``.
  Unaccelerated estimators won't result in bugs or failures, but also won't run
  any faster than they would under ``sklearn``. If you don't see an estimator on
  listed on this page, we do not provide acceleration for it.

- Estimators that are only partially accelerated. ``cuml.accel`` will fall back
  to using the CPU implementations for some algorithms in the presence of
  certain hyperparameters or input types. These cases are documented below in
  estimator-specific sections. See :doc:`logging-and-profiling` for how to
  enable logging to gain insight into when ``cuml.accel`` needs to fall back to
  CPU.

- Missing fitted attributes. ``cuml.accel`` does not currently generate the
  full set of fitted attributes that ``sklearn`` does. In _most_ cases this is
  not a problem, the missing attributes are usually minor things like
  ``n_iters_`` that are useful for inspecting a model fit but not necessary for
  inference. Like unsupported parameters, missing fitted attributes are
  documented in algorithm-specific sections below.

- Differences between fit models. The algorithms and implementations used in
  ``cuml`` naturally differ from those used in ``sklearn``, this may result in
  differences between fit models. This is to be expected. To compare results
  between models fit with ``cuml.accel`` and those fit without, you should
  compare the model *quality* (using e.g. ``model.score``) and not the numeric
  values of the fitted coefficients.

None of the above should result in bugs (exceptions, failures, poor model
quality, ...). That said, as a beta feature there are likely bugs. If you find
a case that errors or results in a model with measurably worse quality when
run under ``cuml.accel``, please `open an issue`_.

A few additional general notes:

- Performance improvements will be most apparent when running on larger data.
  On very small datasets you might see only a small speedup (or even
  potentially a slowdown).

- For most algorithms, ``y`` must already be converted to numeric values;
  arrays of strings are not supported. Pre-encode string labels into numerical
  or categorical formats (e.g., using scikit-learn's LabelEncoder) prior to
  processing.

- The accelerator is compatible with scikit-learn version 1.4 or higher. This
  compatibility ensures that cuML's implementation of scikit-learn compatible
  APIs works as expected.

- Error and warning messages and formats may differ from scikit-learn. Some
  errors might present as C++ stacktraces instead of python errors.

For notes on each algorithm, please refer to its specific section on this file.


hdbscan
-------

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


sklearn.cluster
---------------

The algorithms used in ``cuml`` differ from those in ``sklearn``. As such, you
shouldn't expect the fitted attributes (e.g. ``labels_``) to numerically match
an estimator fitted without ``cuml.accel``.

To compare results between estimators, we recommend comparing scores like
``sklearn.metrics.adjusted_rand_score`` or
``sklearn.metrics.adjusted_mutual_info_score``. For low dimensional data you
can also visually inspect the resulting cluster assignments.

KMeans
^^^^^^

``KMeans`` will fall back to CPU in the following cases:

- If a callable ``init`` is provided.
- If ``X`` is sparse.

DBSCAN
^^^^^^

``DBSCAN`` will fall back to CPU in the following cases:

- If ``algorithm`` isn't ``"auto"`` or ``"brute"``.
- If ``metric`` isn't one of the supported metrics (``"l2"``, ``"euclidean"``, ``"cosine"``, ``"precomputed"``).
- If ``X`` is sparse.


sklearn.decomposition
---------------------

The ``sklearn.decomposition`` implementations used by ``cuml.accel`` uses
different SVD solvers than the ones in Scikit-Learn, which may result in
numeric differences in the ``components_`` and ``explained_variance_`` values.
These differences should be small for most algorithms, but may be larger for
randomized or less-numerically-stable solvers like ``"randomized"`` or
``"covariance_eigh"``.

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

PCA
^^^

``PCA`` will fall back to CPU in the following cases:

- If ``n_components="mle"``.

Additional notes:

- Parameters for the ``"randomized"`` solver like ``random_state``,
  ``n_oversamples``, ``power_iteration_normalizer`` are ignored.

TruncatedSVD
^^^^^^^^^^^^

``TruncatedSVD`` will fall back to CPU in the following cases:

- If ``X`` is sparse.


Additional notes:

- Parameters for the ``"randomized"`` solver like ``random_state``,
  ``n_oversamples``, ``power_iteration_normalizer`` are ignored.


sklearn.ensemble
----------------

The random forest implementation used by ``cuml.accel`` algorithmically
differs from the one in ``sklearn``. As such, you
shouldn't expect the fitted attributes (e.g. ``estimators_``) to numerically match
an estimator fitted without ``cuml.accel``.

To compare results between estimators, we recommend comparing scores like
``sklearn.metrics.root_mean_squared_error`` (for regression) or
``sklearn.metrics.log_loss`` (for classification).

RandomForestClassifier
^^^^^^^^^^^^^^^^^^^^^^

``RandomForestClassifier`` will fall back to CPU in the following cases:

- If ``criterion`` is ``"log_loss"``.
- If ``oob_score=True``.
- If ``warm_start=True``.
- If ``monotonic_cst`` is not ``None``.
- If ``max_values`` is an integer.
- If ``min_weight_fraction_leaf`` is not ``0``.
- If ``ccp_alpha`` is not ``0``.
- If ``class_weight`` is not ``None``.
- If ``sample_weight`` is passed to ``fit`` or ``score``.
- If ``X`` is sparse.

RandomForestRegressor
^^^^^^^^^^^^^^^^^^^^^

``RandomForestRegressor`` will fall back to CPU in the following cases:

- If ``criterion`` is ``"absolute_error"`` or ``"friedman_mse"``.
- If ``oob_score=True``.
- If ``warm_start=True``.
- If ``monotonic_cst`` is not ``None``.
- If ``max_values`` is an integer.
- If ``min_weight_fraction_leaf`` is not ``0``.
- If ``ccp_alpha`` is not ``0``.
- If ``sample_weight`` is passed to ``fit`` or ``score``.
- If ``X`` is sparse.


sklearn.kernel_ridge
--------------------

KernelRidge
^^^^^^^^^^^

``KernelRidge`` will fall back to CPU in the following cases:

- If ``X`` is sparse.

``KernelRidge`` results should be almost identical to those of Scikit-Learn
when running with ``cuml.accel`` enabled. In particular, the fitted
``dual_coef_`` should be close enough that they may be compared via
``np.allclose``.


sklearn.linear_model
--------------------

The linear model solvers used by ``cuml.accel`` differ from those used in
``sklearn``. As such, you shouldn't expect the fitted attributes (e.g.
``coef_``) to numerically match an estimator fitted without ``cuml.accel``. For
some estimators (e.g. ``LinearRegression``) you might get a close match, but
for others there may larger numeric differences.

To compare results between estimators, we recommend comparing model quality
scores like ``sklearn.metrics.r2_score`` (for regression) or
``sklearn.metrics.accuracy_score`` (for classification).

LinearRegression
^^^^^^^^^^^^^^^^

``LinearRegression`` will fall back to CPU in the following cases:

- If ``positive=True``.
- If ``X`` is sparse.

Additionally, the following fitted attributes are currently not computed:

- ``rank_``
- ``singular_``

LogisticRegression
^^^^^^^^^^^^^^^^^^

``LogisticRegression`` will fall back to CPU in the following cases:

- If ``warm_start=True``.
- If ``intercept_scaling`` is not ``1``.
- If the deprecated ``multi_class`` parameter is used.

ElasticNet
^^^^^^^^^^

``ElasticNet`` will fall back to CPU in the following cases:

- If ``positive=True``.
- If ``warm_start=True``.
- If ``precompute`` is not ``False``.
- If ``X`` is sparse.

Additionally, the following fitted attributes are currently not computed:

- ``dual_gap_``
- ``n_iter_``

Ridge
^^^^^

``Ridge`` will fall back to CPU in the following cases:

- If ``positive=True``.
- If ``solver="lbfgs"``.
- If ``X`` is sparse.
- If ``X`` has more columns than rows.
- If ``y`` is multioutput.

Additionally, the following fitted attributes are currently not computed:

- ``n_iter_``

Lasso
^^^^^

``Lasso`` will fall back to CPU in the following cases:

- If ``positive=True``.
- If ``warm_start=True``.
- If ``precompute`` is not ``False``.
- If ``X`` is sparse.

Additionally, the following fitted attributes are currently not computed:

- ``dual_gap_``
- ``n_iter_``


sklearn.manifold
----------------

TSNE
^^^^

``TSNE`` will fall back to CPU in the following cases:

- If ``n_components`` is not ``2``.
- If ``init`` is an array.
- If ``metric`` isn't one of the supported metrics ( ``"l2"``, ``"euclidean"``,
  ``"sqeuclidean"``, ``"cityblock"``, ``"l1"``, ``"manhattan"``,
  ``"minkowski"``, ``"chebyshev"``, ``"cosine"``, ``"correlation"``).

Additional notes:

- Even with a ``random_state``, the TSNE implementation used by ``cuml.accel``
  isn't completely deterministic.

While the exact numerical output for TSNE may differ from that obtained without
``cuml.accel``, we expect the *quality* of results will be approximately as
good in most cases. Beyond comparing the visual representation, you may find
comparing the trustworthiness score (computed via
``sklearn.manifold.trustworthiness``) or the ``kl_divergence_`` fitted
attribute useful.

SpectralEmbedding
^^^^^^^^^^^^^^^^^

``SpectralEmbedding`` will fall back to CPU in the following cases:

- If ``affinity`` is not ``"nearest_neighbors"`` or ``"precomputed"``.
- If ``X`` is sparse.
- If ``X`` has only 1 feature.


The following fitted attributes are currently not computed:

- ``affinity_matrix_``


sklearn.neighbors
-----------------

NearestNeighbors
^^^^^^^^^^^^^^^^

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

KNeighborsClassifier
^^^^^^^^^^^^^^^^^^^^

``KNeighborsClassifier`` will fall back to CPU in the following cases:

- If ``metric`` is not one of the supported metrics ( ``"l2"``,
  ``"euclidean"``, ``"l1"``, ``"cityblock"``, ``"manhattan"``, ``"taxicab"``,
  ``"canberra"``, ``"minkowski"``, ``"lp"``, ``"chebyshev"``, ``"linf"``,
  ``"jensenshannon"``, ``"cosine"``, ``"correlation"``, ``"inner_product"``,
  ``"sqeuclidean"``, ``"haversine"``).
- If ``weights`` is not ``"uniform"``.

Additional notes:

- The ``algorithm`` parameter is ignored, the GPU accelerated ``"brute"``
  implementation in cuml will always be used.

KNeighborsRegressor
^^^^^^^^^^^^^^^^^^^

``KNeighborsRegressor`` will fall back to CPU in the following cases:

- If ``metric`` is not one of the supported metrics ( ``"l2"``,
  ``"euclidean"``, ``"l1"``, ``"cityblock"``, ``"manhattan"``, ``"taxicab"``,
  ``"canberra"``, ``"minkowski"``, ``"lp"``, ``"chebyshev"``, ``"linf"``,
  ``"jensenshannon"``, ``"cosine"``, ``"correlation"``, ``"inner_product"``,
  ``"sqeuclidean"``, ``"haversine"``).
- If ``weights`` is not ``"uniform"``.

Additional notes:

- The ``algorithm`` parameter is ignored, the GPU accelerated ``"brute"``
  implementation in cuml will always be used.


sklearn.svm
-----------

The SVM used by ``cuml.accel`` differ from those used in ``sklearn``. As such,
you shouldn't expect the fitted attributes (e.g. ``coef_`` or
``support_vectors_``) to numerically match an estimator fitted without
``cuml.accel``.

To compare results between estimators, we recommend comparing model quality
scores like ``sklearn.metrics.r2_score`` (for regression) or
``sklearn.metrics.accuracy_score`` (for classification).

SVC
^^^

``SVC`` will fall back to CPU in the following cases:

- If ``kernel="precomputed"`` or is a callable.
- If ``X`` is sparse.
- If ``y`` is multiclass.

Additionally, the following fitted attributes are currently not computed:

- ``class_weight_``
- ``n_iter_``

SVR
^^^

``SVR`` will fall back to CPU in the following cases:

- If ``kernel="precomputed"`` or is a callable.
- If ``X`` is sparse.

Additionally, the following fitted attributes are currently not computed:

- ``n_iter_``

LinearSVC
^^^^^^^^^

``LinearSVC`` will fall back to CPU in the following cases:

- If ``X`` is sparse.
- If ``intercept_scaling`` is not ``1``.
- If ``multi_class`` is not ``"ovr"``.

The following fitted attributes are currently not computed:

- ``n_iter_``

Additional notes:

- Sample weight functionality may not produce equivalent results to replicating data according to weights.
- Use of sample weights may not produce exactly equivalent results when compared to replicating data according to weights.
- Models may not be picklable; pickling or unpickling may fail.
- Multi-class models may have coefficient shape differences that cause pickling failures.

LinearSVR
^^^^^^^^^

``LinearSVR`` will fall back to CPU in the following cases:

- If ``X`` is sparse.
- If ``intercept_scaling`` is not ``1``.

The following fitted attributes are currently not computed:

- ``n_iter_``

Additional notes:

- Use of sample weights may not produce exactly equivalent results when compared to replicating data according to weights.
- Models may not be picklable under certain conditions; pickling or unpickling may fail.

umap
----

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

Additional notes:

- Reproducibility with the use of a seed (the ``random_state`` parameter) comes
  at the relative expense of performance.

- Parallelism during the optimization stage implies numerical imprecisions,
  which can lead to difference in the results between CPU and GPU in general.

While the exact numerical output for UMAP may differ from that obtained without
``cuml.accel``, we expect the *quality* of results will be approximately as
good in most cases. Beyond comparing the visual representation, you may find
comparing the trustworthiness score (computed via
``sklearn.manifold.trustworthiness``) useful.


.. _open an issue: https://github.com/rapidsai/cuml/issues
