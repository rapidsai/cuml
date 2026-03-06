cuml.dask
=========

Multi-node, multi-GPU algorithms using Dask.

Cluster
-------

.. currentmodule:: cuml.dask.cluster

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   DBSCAN
   KMeans

Decomposition
-------------

.. currentmodule:: cuml.dask.decomposition

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   PCA
   TruncatedSVD

Ensemble
--------

.. currentmodule:: cuml.dask.ensemble

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   RandomForestClassifier
   RandomForestRegressor

Linear Models
-------------

.. currentmodule:: cuml.dask.linear_model

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   LinearRegression
   Ridge
   Lasso
   ElasticNet

Manifold
--------

.. currentmodule:: cuml.dask.manifold

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   UMAP

Naive Bayes
-----------

.. currentmodule:: cuml.dask.naive_bayes

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   MultinomialNB

Neighbors
---------

.. currentmodule:: cuml.dask.neighbors

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   NearestNeighbors
   KNeighborsClassifier
   KNeighborsRegressor

Preprocessing
-------------

.. currentmodule:: cuml.dask.preprocessing

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   LabelBinarizer
   OneHotEncoder

Feature Extraction
------------------

.. currentmodule:: cuml.dask.feature_extraction.text

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   TfidfTransformer

Datasets
--------

.. currentmodule:: cuml.dask.datasets

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   make_blobs
   make_classification
   make_regression

Solvers
-------

.. currentmodule:: cuml.dask.solvers

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   CD

Base Classes and Mixins
-----------------------

.. currentmodule:: cuml.dask.common.base

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   BaseEstimator
   DelayedParallelFunc
   DelayedPredictionMixin
   DelayedTransformMixin
   DelayedInverseTransformMixin
