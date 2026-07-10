Supported Functionality
=======================

``cuml.accel`` can accelerate the estimators listed below. Support is not
necessarily all-or-nothing: a listed estimator may use the CPU for particular
methods, parameter values, input types, or dependency versions. Estimators not
listed here continue to use their original CPU implementations.

The supported Scikit-Learn range is versions 1.6 through 1.9. The preprocessing
estimators marked below rely on Scikit-Learn's array API support and require
version 1.8 or newer for GPU acceleration. See :doc:`limitations` for exact
estimator-specific conditions, and use :doc:`acceleration diagnostics
<understanding-acceleration>` to observe GPU execution and CPU fallback in a
real workload.

Scikit-Learn
------------

``sklearn.cluster``
~~~~~~~~~~~~~~~~~~~

* ``DBSCAN``
* ``KMeans``
* ``SpectralClustering``

``sklearn.covariance``
~~~~~~~~~~~~~~~~~~~~~~

* ``EmpiricalCovariance``
* ``LedoitWolf``

``sklearn.decomposition``
~~~~~~~~~~~~~~~~~~~~~~~~~

* ``IncrementalPCA``
* ``PCA``
* ``TruncatedSVD``

``sklearn.ensemble``
~~~~~~~~~~~~~~~~~~~~

* ``RandomForestClassifier``
* ``RandomForestRegressor``

``sklearn.kernel_ridge``
~~~~~~~~~~~~~~~~~~~~~~~~

* ``KernelRidge``

``sklearn.linear_model``
~~~~~~~~~~~~~~~~~~~~~~~~

* ``ElasticNet``
* ``Lasso``
* ``LinearRegression``
* ``LogisticRegression``
* ``Ridge``

``sklearn.manifold``
~~~~~~~~~~~~~~~~~~~~

* ``SpectralEmbedding``
* ``TSNE``

``sklearn.neighbors``
~~~~~~~~~~~~~~~~~~~~~

* ``KNeighborsClassifier``
* ``KNeighborsRegressor``
* ``KernelDensity``
* ``NearestNeighbors``

``sklearn.preprocessing``
~~~~~~~~~~~~~~~~~~~~~~~~~

* ``LabelBinarizer``
* ``LabelEncoder``
* ``MaxAbsScaler`` (requires Scikit-Learn 1.8 or newer)
* ``MinMaxScaler`` (requires Scikit-Learn 1.8 or newer)
* ``PolynomialFeatures`` (requires Scikit-Learn 1.8 or newer)
* ``StandardScaler`` (requires Scikit-Learn 1.8 or newer)
* ``TargetEncoder``

``sklearn.svm``
~~~~~~~~~~~~~~~

* ``LinearSVC``
* ``LinearSVR``
* ``SVC``
* ``SVR``

For parameter, input, method, and fitted-attribute details, see the
:ref:`Scikit-Learn limitations <sklearn-limitations>` and use
:doc:`understanding-acceleration` to identify fallbacks at runtime.

UMAP
----

* ``umap.UMAP``

See the :ref:`UMAP limitations <umap-limitations>` and
:doc:`understanding-acceleration` for runtime diagnostics.

HDBSCAN
-------

* ``hdbscan.HDBSCAN``

See the :ref:`HDBSCAN limitations <hdbscan-limitations>` and
:doc:`understanding-acceleration` for runtime diagnostics.
