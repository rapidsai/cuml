~~~~~~~~~~~~~~~~~~~
cuML API Reference
~~~~~~~~~~~~~~~~~~~

Datatype Configuration
======================

Output Type
-----------

 .. automethod:: cuml.utils.memory_utils.set_global_output_type
 .. automethod:: cuml.utils.memory_utils.using_output_type


Preprocessing, Metrics, and Utilities
=====================================

Model Selection and Data Splitting
----------------------------------

 .. automethod:: cuml.preprocessing.model_selection.train_test_split

Label Encoding
--------------

 .. autoclass:: cuml.preprocessing.LabelEncoder
    :members:

 .. autoclass:: cuml.preprocessing.LabelBinarizer
    :members:

 .. autoclass:: cuml.dask.preprocessing.LabelBinarizer
    :members:

 .. automethod:: cuml.preprocessing.label_binarize

Dataset Generation (Single-GPU)
-------------------------------

  .. automethod:: cuml.datasets.make_blobs
  .. automethod:: cuml.datasets.make_regression


Dataset Generation (Dask-based Multi-GPU)
-----------------------------------------
  .. automodule:: cuml.dask.datasets.blobs
     :members:

  .. automodule:: cuml.dask.datasets.regression
     :members:


Metrics
---------

  .. automodule:: cuml.metrics.regression
    :members:

  .. automodule:: cuml.metrics.accuracy
    :members:

  .. automodule:: cuml.metrics.trustworthiness
    :members:

  .. automodule:: cuml.metrics.cluster.adjustedrandindex
    :members:

  .. automodule:: cuml.metrics.cluster.entropy
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

Nearest Neighbors Regression
----------------------------

.. autoclass:: cuml.neighbors.KNeighborsRegressor
    :members:

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

Linear Models
-------------

.. autoclass:: cuml.dask.linear_model.LinearRegression
    :members:

.. autoclass:: cuml.dask.linear_model.Ridge
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
