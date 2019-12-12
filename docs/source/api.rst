~~~~~~~~~~~~~~~~~~~
cuML API Reference
~~~~~~~~~~~~~~~~~~~



Preprocessing, Metrics, and Utilities
=====================================

Model Selection and Data Splitting
----------------------------------

 .. automodule:: cuml.preprocessing.model_selection
    :members:

Label Encoding
--------------

 .. autoclass:: cuml.preprocessing.LabelEncoder
    :members:

Dataset Generation (Single-GPU)
-------------------------------

  .. automethod:: cuml.datasets.make_blobs


Dataset Generation (Dask-based Multi-GPU)
-----------------------------------------
  .. automodule:: cuml.dask.datasets.blobs
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

Benchmarking
-------------

  .. automodule:: cuml.benchmark.algorithms
    :members:

  .. automodule:: cuml.benchmark.runners
    :members:

  .. automodule:: cuml.benchmark.datagen
    :members:



Utilities for I/O and Numba
---------------------------

  .. automodule:: cuml.utils.input_utils
    :members:

  .. automodule:: cuml.utils.numba_utils
    :members:

Utilities for Dask and Multi-GPU Preprocessing
-----------------------------------------------

  .. automodule:: cuml.dask.common.utils
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

Quasi-Newton
------------

.. autoclass:: cuml.QN
    :members:

Support Vector Machines
------------------------

.. autoclass:: cuml.svm.SVC
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

Nearest Neighbors Classification
--------------------------------

.. autoclass:: cuml.neighbors.KNeighborsRegressor
    :members:

Time Series
============

HoltWinters
-------------

.. autoclass:: cuml.ExponentialSmoothing
    :members:

Kalman Filter
-------------

.. autoclass:: cuml.KalmanFilter
    :members:

ARIMA
-----

.. autoclass:: cuml.tsa.ARIMAModel
    :members:


Multi-Node, Multi-GPU Algorithms
================================

K-Means Clustering
--------------------

.. autoclass:: cuml.dask.cluster.KMeans
    :members:

Random Forest
-------------

.. autoclass:: cuml.dask.ensemble.RandomForestClassifier
    :members:

.. autoclass:: cuml.dask.ensemble.RandomForestRegressor
    :members:


Principal Component Analysis
-----------------------------
.. autoclass:: cuml.dask.decomposition.PCA
    :members:

Truncated SVD
--------------

.. autoclass:: cuml.dask.decomposition.TruncatedSVD
    :members:
