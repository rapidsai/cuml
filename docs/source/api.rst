~~~~~~~~~~~~~~~~~~~
cuML API Reference
~~~~~~~~~~~~~~~~~~~



Preprocessing
==============

Model Selection and Data Splitting
----------------------------------

 .. automodule:: cuml.preprocessing.model_selection
    :members:

Label Encoding
--------------

 .. autoclass:: cuml.preprocessing.LabelEncoder
    :members:

Dataset Generation
------------------

  .. automethod:: cuml.datasets.make_blobs

Regression and Classification
=============================

Linear Regression
-----------------

.. autoclass:: cuml.LinearRegression
    :members:

Logistic Regression
-----------------

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

Random Forest Inferencing
-------------------------

.. autoclass:: cuml.ForestInference
    :members:

Quasi-Newton
------------

.. autoclass:: cuml.QN
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

.. autoclass:: cuml.NearestNeighbors
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
