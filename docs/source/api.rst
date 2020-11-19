~~~~~~~~~~~~~~~~~~~
cuML API Reference
~~~~~~~~~~~~~~~~~~~

Module Configuration
====================

.. _output-data-type-configuration:

Output Data Type Configuration
------------------------------

 .. automethod:: cuml.common.memory_utils.set_global_output_type
 .. automethod:: cuml.common.memory_utils.using_output_type

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
     - cuml.common.logger.level_off
     - Disables all log messages
   * - 1
     - cuml.common.logger.level_critical
     - Enables only critical messages
   * - 2
     - cuml.common.logger.level_error
     - Enables all messages up to and including errors.
   * - 3
     - cuml.common.logger.level_warn
     - Enables all messages up to and including warnings.
   * - 4 or False
     - cuml.common.logger.level_info
     - Enables all messages up to and including information messages.
   * - 5 or True
     - cuml.common.logger.level_debug
     - Enables all messages up to and including debug messages.
   * - 6
     - cuml.common.logger.level_trace
     - Enables all messages up to and including trace messages.


Preprocessing, Metrics, and Utilities
=====================================

Model Selection and Data Splitting
----------------------------------

 .. automethod:: cuml.preprocessing.model_selection.train_test_split

Feature and Label Encoding (Single-GPU)
---------------------------------------

 .. autoclass:: cuml.preprocessing.LabelEncoder.LabelEncoder
    :members:

 .. autoclass:: cuml.preprocessing.LabelBinarizer
    :members:

 .. automethod:: cuml.preprocessing.label_binarize

 .. autoclass:: cuml.preprocessing.OneHotEncoder
    :members:

 .. autoclass:: cuml.preprocessing.TargetEncoder.TargetEncoder
    :members:


Text Preprocessing (Single-GPU)
---------------------------------------
 .. autoclass:: cuml.preprocessing.text.stem.PorterStemmer
    :members:

Feature and Label Encoding (Dask-based Multi-GPU)
-------------------------------------------------

 .. autoclass:: cuml.dask.preprocessing.LabelBinarizer
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

Dataset Generation (Single-GPU)
-------------------------------
  .. glossary::
    random_state
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
  .. automethod:: cuml.datasets.make_blobs
  .. automethod:: cuml.datasets.make_classification
  .. automethod:: cuml.datasets.make_regression
  .. automethod:: cuml.datasets.make_arima


Dataset Generation (Dask-based Multi-GPU)
-----------------------------------------
  .. automodule:: cuml.dask.datasets.blobs
     :members:

  .. automodule:: cuml.dask.datasets.classification
     :members:

  .. automodule:: cuml.dask.datasets.regression
     :members:

Array Wrappers (Internal API)
-----------------------------

.. autoclass:: cuml.common.CumlArray
    :members:

Metrics (regression, classification, and distance)
---------------------------------------------------

  .. automodule:: cuml.metrics.regression
    :members:

  .. automodule:: cuml.metrics.accuracy
    :members:

  .. automethod:: cuml.metrics.confusion_matrix

  .. automethod:: cuml.metrics.roc_auc_score

  .. automethod:: cuml.metrics.precision_recall_curve

  .. automodule:: cuml.metrics.pairwise_distances
    :members:

Metrics (clustering and trustworthiness)
----------------------------------------
  .. automodule:: cuml.metrics.trustworthiness
    :members:

  .. automodule:: cuml.metrics.cluster.adjusted_rand_index
    :members:

  .. automodule:: cuml.metrics.cluster.entropy
    :members:

  .. automodule:: cuml.metrics.cluster.homogeneity_score
    :members:

  .. automodule:: cuml.metrics.cluster.completeness_score
    :members:

  .. automodule:: cuml.metrics.cluster.mutual_info_score
    :members:

Benchmarking
-------------

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

Mutinomial Naive Bayes
----------------------

.. autoclass:: cuml.MultinomialNB
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

Coordinate Descent
------------------

.. autoclass:: cuml.CD
    :members:

Quasi-Newton
------------

.. autoclass:: cuml.QN
    :members:

Support Vector Machines
------------------------

.. autoclass:: cuml.svm.SVC
    :members:

.. autoclass:: cuml.svm.SVR
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

Clustering
==========

K-Means Clustering
--------------------

.. autoclass:: cuml.KMeans
    :members:

DBSCAN
-------

.. autoclass:: cuml.DBSCAN
    :members:

Dimensionality Reduction and Manifold Learning
==============================================

Principal Component Analysis
-----------------------------

.. autoclass:: cuml.PCA
    :members:

Truncated SVD
--------------

.. autoclass:: cuml.TruncatedSVD
    :members:

UMAP
-------------

.. autoclass:: cuml.UMAP
    :members:

Random Projections
------------------

.. autoclass:: cuml.random_projection.GaussianRandomProjection
    :members:

.. autoclass:: cuml.random_projection.SparseRandomProjection
    :members:

.. automethod:: cuml.random_projection.johnson_lindenstrauss_min_dim


TSNE
-------------

.. autoclass:: cuml.TSNE
    :members:

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
--------------------------------

.. autoclass:: cuml.neighbors.KNeighborsRegressor
    :members:

Time Series
============

HoltWinters
-------------

.. autoclass:: cuml.ExponentialSmoothing
    :members:

ARIMA
-----

.. autoclass:: cuml.tsa.ARIMA
    :members:

.. autoclass:: cuml.tsa.auto_arima.AutoARIMA
    :members:

Multi-Node, Multi-GPU Algorithms
================================

K-Means Clustering
--------------------

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
-----------------------------
.. autoclass:: cuml.dask.decomposition.PCA
    :members:

Random Forest
-------------

.. autoclass:: cuml.dask.ensemble.RandomForestClassifier
    :members:

.. autoclass:: cuml.dask.ensemble.RandomForestRegressor
    :members:

Truncated SVD
--------------

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

Experimental
============

.. warning:: The `cuml.experimental` module contains features that are still
    under development. It is not recommended to depend on features in this
    module as they may change in future releases.

.. note:: Due to the nature of this module, it is not imported by default by
    the root `cuml` package. Each `experimental` submodule must be imported
    separately.

Decomposition
-------------
.. autoclass:: cuml.experimental.decomposition.IncrementalPCA
   :members:

Preprocessing
-------------
.. automodule:: cuml.experimental.preprocessing
   :members: Binarizer, KBinsDiscretizer, MaxAbsScaler, MinMaxScaler,
      Normalizer, RobustScaler, SimpleImputer, StandardScaler,
      add_dummy_feature, binarize, minmax_scale, normalize,
      PolynomialFeatures, robust_scale, scale
