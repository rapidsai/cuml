CPU / GPU Device Selection (Experimental)
========================================================

cuML provides experimental support for running selected estimators and operators on either the GPU or CPU. This document covers the set of operators for which CPU/GPU device selection capabilities are supported as of the current nightly packages. If an operator isn't listed here, it can only be run on the GPU. Prior versions of cuML may have reduced support compared to the following table.


.. list-table:: Operators Supporting CPU/GPU Device Selection and Execution
   :header-rows: 1
   :align: center
   :widths: auto

   * - Category
     - Operator
   * - Clustering
     - HDBSCAN
   * - Clustering
     - KMeans
   * - Dimensionality Reduction and Manifold Learning
     - PCA
   * - Dimensionality Reduction and Manifold Learning
     - TruncatedSVD
   * - Dimensionality Reduction and Manifold Learning
     - UMAP
   * - Neighbors
     - KNeighborsClassifier
   * - Neighbors
     - KNeighborsRegressor
   * - Neighbors
     - NearestNeighbors
   * - Regression and Classification
     - LinearRegression
   * - Regression and Classification
     - LogisticRegression
   * - Regression and Classification
     - Lasso
   * - Regression and Classification
     - Ridge
   * - Regression and Classification
     - ElasticNet
   * - Regression and Classification
     - KNeighborsRegressor
   * - Regression and Classification
     - KNeighborsClassifier


.. list-table:: CPU / GPU Device Selection Support Matrix
   :header-rows: 1
   :align: center

   * - Category
     - Operator
     - Supports Device Selection
   * - Preprocessing, Metrics, and Utilities
     - LabelEncoder
     - 
   * - Preprocessing, Metrics, and Utilities
     - LabelBinarizer
     - 
   * - Preprocessing, Metrics, and Utilities
     - OneHotEncoder
     - 
   * - Preprocessing, Metrics, and Utilities
     - MaxAbsScaler
     - 
   * - Preprocessing, Metrics, and Utilities
     - MinMaxScaler
     - 
   * - Clustering
     - HDBSCAN
     - X