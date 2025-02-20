Known Limitations
-----------------

General Limitations
~~~~~~~~~~~~~~~~~~~

TODO(wphicks): Fill this in
TODO(wphicks): Pickle

Algorithm-Specific Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO(wphicks): Fills these in. Document when each will fall back to CPU, how to
assess equivalence with CPU implementations, and significant differences in
algorithm, as well as any other known issues.


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
different tree structure. You can tune the fineness of the quantiles by adjusting the
``n_bins`` parameter.

Some parameters have limited support:
* ``max_samples`` must be float, not integer.

The following parameters are not supported:
* ``min_weight_fraction_leaf``
* ``monotonic_cst``
* ``ccp_alpha``
* ``class_weight``
* ``warm_start``
* ``oob_score``

TODO(hcho3): Add the list of supported ``criterion``, once the PR for mapping ``split_criterion``
to ``criterion`` lands.

TODO(hcho3): If the PR mapping ``max_leaves`` to ``max_leaf_nodes`` lands, add explanation about
the behavior of ``max_leaf_nodes``. The cuML RF treats this parameter as a "soft constraint".

``sklearn.kernel_ridge.KernelRidge``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.linear_model.LinearRegression``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.linear_model.LogisticRegression``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.linear_model.ElasticNet``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.linear_model.Ridge``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.linear_model.Lasso``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

``umap.UMAP``
^^^^^^^^^^^^^

* Algorithm Limitations:
    * The following parameters are not supported : "low_memory", "angular_rp_forest", "transform_seed", "tqdm_kwds", "unique", "densmap", "dens_lambda", "dens_frac", "dens_var_shift", "output_dens", "disconnection_distance".
    * Parallelism during the optimization stage implies numerical imprecisions.
    * There may be cases where cuML's UMAP may not achieve the same level of quality as the reference implementation. The trustworthiness score can be used to assess to what extent the local structure is retained in embedding.
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
