~~~~~~~~~~~~~
API Reference
~~~~~~~~~~~~~

.. role:: py(code)
   :language: python
   :class: highlight


Module Configuration
====================

.. _output-data-type-configuration:

Output Data Type Configuration
------------------------------

 .. autofunction:: cuml.internals.memory_utils.set_global_output_type
 .. autofunction:: cuml.internals.memory_utils.using_output_type

.. _verbosity-levels:

Verbosity Levels
----------------

cuML follows a verbosity model similar to Scikit-learn's: The verbose parameter
can be a boolean, or a numeric value, and higher numeric values mean more verbosity. The exact values can be set directly, or through the cuml.common.logger module, and
they are:

.. list-table:: Verbosity Levels
   :widths: 25 25 50
   :header-rows: 1

   * - Numeric value
     - cuml.common.logger value
     - Verbosity level
   * - 0
     - cuml.common.logger.level_enum.off
     - Disables all log messages
   * - 1
     - cuml.common.logger.level_enum.critical
     - Enables only critical messages
   * - 2
     - cuml.common.logger.level_enum.error
     - Enables all messages up to and including errors.
   * - 3
     - cuml.common.logger.level_enum.warn
     - Enables all messages up to and including warnings.
   * - 4 or False
     - cuml.common.logger.level_enum.info
     - Enables all messages up to and including information messages.
   * - 5 or True
     - cuml.common.logger.level_enum.debug
     - Enables all messages up to and including debug messages.
   * - 6
     - cuml.common.logger.level_enum.trace
     - Enables all messages up to and including trace messages.


Preprocessing, Metrics, and Utilities
=====================================

Model Selection and Data Splitting
----------------------------------

 .. autofunction:: cuml.model_selection.train_test_split

Feature and Label Encoding (Single-GPU)
---------------------------------------

 .. autoclass:: cuml.preprocessing.LabelEncoder.LabelEncoder
    :members:

 .. autoclass:: cuml.preprocessing.LabelBinarizer
    :members:

 .. autofunction:: cuml.preprocessing.label_binarize

 .. autoclass:: cuml.preprocessing.OneHotEncoder
    :members:

 .. autoclass:: cuml.preprocessing.TargetEncoder.TargetEncoder
    :members:

Feature Scaling and Normalization (Single-GPU)
----------------------------------------------
.. autoclass:: cuml.preprocessing.MaxAbsScaler
    :members:
.. autoclass:: cuml.preprocessing.MinMaxScaler
    :members:
.. autoclass:: cuml.preprocessing.Normalizer
    :members:
.. autoclass:: cuml.preprocessing.RobustScaler
    :members:
.. autoclass:: cuml.preprocessing.StandardScaler
    :members:
.. autofunction:: cuml.preprocessing.maxabs_scale
.. autofunction:: cuml.preprocessing.minmax_scale
.. autofunction:: cuml.preprocessing.normalize
.. autofunction:: cuml.preprocessing.robust_scale
.. autofunction:: cuml.preprocessing.scale

Other preprocessing methods (Single-GPU)
----------------------------------------
.. autoclass:: cuml.preprocessing.Binarizer
    :members:
.. autoclass:: cuml.preprocessing.FunctionTransformer
    :members:
.. autoclass:: cuml.preprocessing.KBinsDiscretizer
    :members:
.. autoclass:: cuml.preprocessing.KernelCenterer
    :members:
.. autoclass:: cuml.preprocessing.MissingIndicator
    :members:
.. autoclass:: cuml.preprocessing.PolynomialFeatures
    :members:
.. autoclass:: cuml.preprocessing.PowerTransformer
    :members:
.. autoclass:: cuml.preprocessing.QuantileTransformer
    :members:
.. autoclass:: cuml.preprocessing.SimpleImputer
    :members:
.. autofunction:: cuml.preprocessing.add_dummy_feature
.. autofunction:: cuml.preprocessing.binarize

.. automodule:: cuml.compose
   :members: ColumnTransformer, make_column_transformer, make_column_selector

Text Preprocessing (Single-GPU)
-------------------------------
 .. autoclass:: cuml.preprocessing.text.stem.PorterStemmer
    :members:

Feature and Label Encoding (Dask-based Multi-GPU)
-------------------------------------------------

 .. autoclass:: cuml.dask.preprocessing.LabelBinarizer
    :members:

 .. autoclass:: cuml.dask.preprocessing.LabelEncoder.LabelEncoder
    :members:

 .. autoclass:: cuml.dask.preprocessing.OneHotEncoder
    :members:

Feature Extraction (Single-GPU)
-------------------------------

  .. autoclass:: cuml.feature_extraction.text.CountVectorizer
    :members:

  .. autoclass:: cuml.feature_extraction.text.HashingVectorizer
    :members:

  .. autoclass:: cuml.feature_extraction.text.TfidfVectorizer
    :members:

Feature Extraction (Dask-based Multi-GPU)
-----------------------------------------
  .. autoclass:: cuml.dask.feature_extraction.text.TfidfTransformer
    :members:

Dataset Generation (Single-GPU)
-------------------------------
  .. glossary::
    random_state
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
  .. autofunction:: cuml.datasets.make_blobs
  .. autofunction:: cuml.datasets.make_classification
  .. autofunction:: cuml.datasets.make_regression
  .. autofunction:: cuml.datasets.make_arima


Dataset Generation (Dask-based Multi-GPU)
-----------------------------------------
  .. automodule:: cuml.dask.datasets.blobs
     :members:

  .. automodule:: cuml.dask.datasets.classification
     :members:

  .. automodule:: cuml.dask.datasets.regression
     :members:


Metrics (regression, classification, and distance)
--------------------------------------------------

  .. automodule:: cuml.metrics.regression
    :members:

  .. autofunction:: cuml.metrics.accuracy_score

  .. autofunction:: cuml.metrics.confusion_matrix

  .. autofunction:: cuml.metrics.kl_divergence

  .. autofunction:: cuml.metrics.log_loss

  .. autofunction:: cuml.metrics.roc_auc_score

  .. autofunction:: cuml.metrics.precision_recall_curve

  .. automodule:: cuml.metrics.pairwise_distances
    :members:

  .. automodule:: cuml.metrics.pairwise_kernels
    :members:


Metrics (clustering and manifold learning)
------------------------------------------
  .. automodule:: cuml.metrics.trustworthiness
    :members:

  .. automodule:: cuml.metrics.cluster.adjusted_rand_index
    :members:

  .. automodule:: cuml.metrics.cluster.entropy
    :members:

  .. automodule:: cuml.metrics.cluster.homogeneity_score
    :members:

  .. automodule:: cuml.metrics.cluster.silhouette_score
    :members:

  .. automodule:: cuml.metrics.cluster.completeness_score
    :members:

  .. automodule:: cuml.metrics.cluster.mutual_info_score
    :members:

  .. automodule:: cuml.metrics.cluster.v_measure_score
    :members:

Benchmarking
------------

  .. automodule:: cuml.benchmark.algorithms
    :members:

  .. automodule:: cuml.benchmark.runners
    :members:

  .. automodule:: cuml.benchmark.datagen
    :members:


Regression and Classification
=============================

Linear Regression
-----------------

.. autoclass:: cuml.LinearRegression
    :members:

Logistic Regression
-------------------

.. autoclass:: cuml.LogisticRegression
    :members:

Ridge Regression
----------------

.. autoclass:: cuml.Ridge
    :members:

Lasso Regression
----------------

.. autoclass:: cuml.Lasso
    :members:

ElasticNet Regression
---------------------

.. autoclass:: cuml.ElasticNet
    :members:

Mini Batch SGD Classifier
-------------------------

.. autoclass:: cuml.MBSGDClassifier
    :members:

Mini Batch SGD Regressor
------------------------

.. autoclass:: cuml.MBSGDRegressor
    :members:

Multiclass Classification
-------------------------

.. autoclass:: cuml.multiclass.MulticlassClassifier
    :members:

.. autoclass:: cuml.multiclass.OneVsOneClassifier
    :members:

.. autoclass:: cuml.multiclass.OneVsRestClassifier
    :members:

Naive Bayes
-----------

.. autoclass:: cuml.naive_bayes.MultinomialNB
    :members:

.. autoclass:: cuml.naive_bayes.BernoulliNB
    :members:

.. autoclass:: cuml.naive_bayes.ComplementNB
    :members:

.. autoclass:: cuml.naive_bayes.GaussianNB
    :members:

.. autoclass:: cuml.naive_bayes.CategoricalNB
    :members:

Stochastic Gradient Descent
---------------------------

.. autoclass:: cuml.SGD
    :members:

Random Forest
-------------

.. autoclass:: cuml.ensemble.RandomForestClassifier
    :members:

.. autoclass:: cuml.ensemble.RandomForestRegressor
    :members:

Forest Inferencing
------------------

.. autoclass:: cuml.ForestInference
    :members:
    :inherited-members:

Coordinate Descent
------------------

.. autoclass:: cuml.CD
    :members:

Quasi-Newton
------------

.. autoclass:: cuml.QN
    :members:

Support Vector Machines
-----------------------

.. autoclass:: cuml.svm.SVR
    :members:

.. autoclass:: cuml.svm.SVC
    :members: decision_function, fit, predict, predict_log_proba, predict_proba

.. autoclass:: cuml.svm.LinearSVC
    :members:

.. autoclass:: cuml.svm.LinearSVR
    :members:

Nearest Neighbors Classification
--------------------------------

.. autoclass:: cuml.neighbors.KNeighborsClassifier
    :members:
    :noindex:

Nearest Neighbors Regression
----------------------------

.. autoclass:: cuml.neighbors.KNeighborsRegressor
    :members:
    :noindex:

Kernel Ridge Regression
-----------------------

.. autoclass:: cuml.KernelRidge
    :members:


Clustering
==========

K-Means Clustering
------------------

.. autoclass:: cuml.KMeans
    :members:

DBSCAN
------

.. autoclass:: cuml.DBSCAN
    :members:

Agglomerative Clustering
------------------------

.. autoclass:: cuml.AgglomerativeClustering
    :members:


HDBSCAN
-------
.. autoclass:: cuml.cluster.hdbscan.HDBSCAN
    :members:

.. autofunction:: cuml.cluster.hdbscan.all_points_membership_vectors

.. autofunction:: cuml.cluster.hdbscan.membership_vector

.. autofunction:: cuml.cluster.hdbscan.approximate_predict


Dimensionality Reduction and Manifold Learning
==============================================

Principal Component Analysis
-----------------------------

.. autoclass:: cuml.PCA
    :members:

Incremental PCA
---------------
.. autoclass:: cuml.IncrementalPCA
   :members:

Truncated SVD
--------------

.. autoclass:: cuml.TruncatedSVD
    :members:

UMAP
----

.. autoclass:: cuml.UMAP
    :members:

.. autofunction:: cuml.manifold.umap.fuzzy_simplicial_set

.. autofunction:: cuml.manifold.umap.simplicial_set_embedding


Random Projections
------------------

.. autoclass:: cuml.random_projection.GaussianRandomProjection
    :members:

.. autoclass:: cuml.random_projection.SparseRandomProjection
    :members:

.. autofunction:: cuml.random_projection.johnson_lindenstrauss_min_dim


TSNE
----

.. autoclass:: cuml.TSNE
    :members:

Spectral Embedding
------------------

.. autoclass:: cuml.manifold.SpectralEmbedding
    :members:

.. autofunction:: cuml.manifold.spectral_embedding

Neighbors
==========

Nearest Neighbors
-----------------

.. autoclass:: cuml.neighbors.NearestNeighbors
    :members:

Nearest Neighbors Classification
--------------------------------

.. autoclass:: cuml.neighbors.KNeighborsClassifier
    :members:

Nearest Neighbors Regression
----------------------------

.. autoclass:: cuml.neighbors.KNeighborsRegressor
    :members:

Kernel Density Estimation
-------------------------

.. autoclass:: cuml.neighbors.KernelDensity
    :members:

Time Series
===========

HoltWinters
-----------

.. autoclass:: cuml.ExponentialSmoothing
    :members:

ARIMA
-----

.. autoclass:: cuml.tsa.ARIMA
    :members:

.. autoclass:: cuml.tsa.auto_arima.AutoARIMA
    :members:

Model Explainability
====================

SHAP Kernel Explainer
---------------------

.. autoclass:: cuml.explainer.KernelExplainer
   :members:

SHAP Permutation Explainer
--------------------------

.. autoclass:: cuml.explainer.PermutationExplainer
   :members:


Multi-Node, Multi-GPU Algorithms
================================

DBSCAN Clustering
-----------------

.. autoclass:: cuml.dask.cluster.DBSCAN
    :members:

K-Means Clustering
------------------

.. autoclass:: cuml.dask.cluster.KMeans
    :members:

Nearest Neighbors
-----------------

.. autoclass:: cuml.dask.neighbors.NearestNeighbors
    :members:

.. autoclass:: cuml.dask.neighbors.KNeighborsRegressor
    :members:

.. autoclass:: cuml.dask.neighbors.KNeighborsClassifier
    :members:


Principal Component Analysis
----------------------------
.. autoclass:: cuml.dask.decomposition.PCA
    :members:

Random Forest
-------------

.. autoclass:: cuml.dask.ensemble.RandomForestClassifier
    :members:

.. autoclass:: cuml.dask.ensemble.RandomForestRegressor
    :members:

Truncated SVD
-------------

.. autoclass:: cuml.dask.decomposition.TruncatedSVD
    :members:

Manifold
--------

.. autoclass:: cuml.dask.manifold.UMAP
    :members:

Linear Models
-------------

.. autoclass:: cuml.dask.linear_model.LinearRegression
    :members:

.. autoclass:: cuml.dask.linear_model.Ridge
    :members:

.. autoclass:: cuml.dask.linear_model.Lasso
    :members:

.. autoclass:: cuml.dask.linear_model.ElasticNet
    :members:

Naive Bayes
-----------

.. autoclass:: cuml.dask.naive_bayes.MultinomialNB
    :members:

Solvers
-------

.. autoclass:: cuml.dask.solvers.CD
    :members:

Dask Base Classes and Mixins
----------------------------
.. autoclass:: cuml.dask.common.base.BaseEstimator
   :members:

.. autoclass:: cuml.dask.common.base.DelayedParallelFunc
   :members:

.. autoclass:: cuml.dask.common.base.DelayedPredictionMixin
   :members:

.. autoclass:: cuml.dask.common.base.DelayedTransformMixin
   :members:

.. autoclass:: cuml.dask.common.base.DelayedInverseTransformMixin
   :members:

cuml.accel
==========

.. autofunction:: cuml.accel.install

.. autofunction:: cuml.accel.enabled

.. autofunction:: cuml.accel.profile

.. autofunction:: cuml.accel.is_proxy

Experimental
============

.. warning:: The `cuml.experimental` module contains features that are still
    under development. It is not recommended to depend on features in this
    module as they may change in future releases.

.. note:: Due to the nature of this module, it is not imported by default by
    the root `cuml` package. Each `experimental` submodule must be imported
    separately.

Linear Models
-------------
.. autoclass:: cuml.experimental.linear_model.Lars
   :members:

Model Explainability
--------------------
.. autoclass:: cuml.explainer.TreeExplainer
   :members:
