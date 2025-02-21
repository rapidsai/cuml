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

``sklearn.decomposition.PCA``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sklearn.decomposition.TruncatedSVD``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    from umap import UMAP as refUMAP
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
