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

``hdbscan.HDBSCAN``
^^^^^^^^^^^^^^^^^^^
