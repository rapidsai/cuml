:html_theme.sidebar_secondary.remove:

.. _api_ref:

=============
API Reference
=============

This is the class and function reference of cuML. Please refer to the
:doc:`User Guide </user_guide>` for further details, as the raw specifications
of classes and functions may not be enough to give full guidelines on their
use.

.. toctree::
   :maxdepth: 2
   :hidden:

   cuml
   cuml.accel
   cuml.benchmark
   cuml.cluster
   cuml.compose
   cuml.covariance
   cuml.dask
   cuml.datasets
   cuml.decomposition
   cuml.ensemble
   cuml.experimental
   cuml.explainer
   cuml.feature_extraction
   cuml.fil
   cuml.kernel_ridge
   cuml.linear_model
   cuml.manifold
   cuml.metrics
   cuml.model_selection
   cuml.multiclass
   cuml.naive_bayes
   cuml.neighbors
   cuml.preprocessing
   cuml.random_projection
   cuml.solvers
   cuml.svm
   cuml.tsa


.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Object
     - Description
     - Module

   * - :obj:`~cuml.set_global_output_type`
     - Set global output type for cuML estimators.
     - :mod:`cuml`
   * - :obj:`~cuml.using_output_type`
     - Context manager to temporarily set output type.
     - :mod:`cuml`

   * - :obj:`~cuml.accel.install`
     - Install cuML acceleration hooks.
     - :mod:`cuml.accel`
   * - :obj:`~cuml.accel.enabled`
     - Check if cuML acceleration is enabled.
     - :mod:`cuml.accel`
   * - :obj:`~cuml.accel.profile`
     - Profile cuML acceleration.
     - :mod:`cuml.accel`
   * - :obj:`~cuml.accel.is_proxy`
     - Check if an object is a cuML proxy.
     - :mod:`cuml.accel`

   * - :obj:`~cuml.cluster.AgglomerativeClustering`
     - Agglomerative Clustering.
     - :mod:`cuml.cluster`
   * - :obj:`~cuml.cluster.DBSCAN`
     - Density-Based Spatial Clustering of Applications with Noise.
     - :mod:`cuml.cluster`
   * - :obj:`~cuml.cluster.hdbscan.HDBSCAN`
     - Hierarchical DBSCAN clustering.
     - :mod:`cuml.cluster`
   * - :obj:`~cuml.cluster.KMeans`
     - K-Means clustering.
     - :mod:`cuml.cluster`
   * - :obj:`~cuml.cluster.SpectralClustering`
     - Spectral clustering.
     - :mod:`cuml.cluster`

   * - :obj:`~cuml.compose.ColumnTransformer`
     - Applies transformers to columns of an array or DataFrame.
     - :mod:`cuml.compose`

   * - :obj:`~cuml.covariance.LedoitWolf`
     - Ledoit-Wolf shrinkage covariance estimator.
     - :mod:`cuml.covariance`

   * - :obj:`~cuml.datasets.make_blobs`
     - Generate isotropic Gaussian blobs for clustering.
     - :mod:`cuml.datasets`
   * - :obj:`~cuml.datasets.make_classification`
     - Generate a random classification problem.
     - :mod:`cuml.datasets`
   * - :obj:`~cuml.datasets.make_regression`
     - Generate a random regression problem.
     - :mod:`cuml.datasets`

   * - :obj:`~cuml.decomposition.PCA`
     - Principal Component Analysis.
     - :mod:`cuml.decomposition`
   * - :obj:`~cuml.decomposition.IncrementalPCA`
     - Incremental Principal Component Analysis.
     - :mod:`cuml.decomposition`
   * - :obj:`~cuml.decomposition.TruncatedSVD`
     - Dimensionality reduction using truncated SVD.
     - :mod:`cuml.decomposition`

   * - :obj:`~cuml.ensemble.RandomForestClassifier`
     - Random Forest classifier.
     - :mod:`cuml.ensemble`
   * - :obj:`~cuml.ensemble.RandomForestRegressor`
     - Random Forest regressor.
     - :mod:`cuml.ensemble`

   * - :obj:`~cuml.explainer.KernelExplainer`
     - SHAP Kernel Explainer.
     - :mod:`cuml.explainer`
   * - :obj:`~cuml.explainer.PermutationExplainer`
     - SHAP Permutation Explainer.
     - :mod:`cuml.explainer`
   * - :obj:`~cuml.explainer.TreeExplainer`
     - SHAP Tree Explainer.
     - :mod:`cuml.explainer`

   * - :obj:`~cuml.feature_extraction.text.CountVectorizer`
     - Convert a collection of text documents to a matrix of token counts.
     - :mod:`cuml.feature_extraction`
   * - :obj:`~cuml.feature_extraction.text.HashingVectorizer`
     - Convert a collection of text documents to a matrix of token occurrences.
     - :mod:`cuml.feature_extraction`
   * - :obj:`~cuml.feature_extraction.text.TfidfVectorizer`
     - Convert a collection of raw documents to a matrix of TF-IDF features.
     - :mod:`cuml.feature_extraction`

   * - :obj:`~cuml.fil.ForestInference`
     - Forest Inference for fast prediction of tree-based models.
     - :mod:`cuml.fil`

   * - :obj:`~cuml.kernel_ridge.KernelRidge`
     - Kernel Ridge Regression.
     - :mod:`cuml.kernel_ridge`

   * - :obj:`~cuml.linear_model.LinearRegression`
     - Ordinary least squares Linear Regression.
     - :mod:`cuml.linear_model`
   * - :obj:`~cuml.linear_model.LogisticRegression`
     - Logistic Regression classifier.
     - :mod:`cuml.linear_model`
   * - :obj:`~cuml.linear_model.Ridge`
     - Ridge regression.
     - :mod:`cuml.linear_model`
   * - :obj:`~cuml.linear_model.Lasso`
     - Lasso regression.
     - :mod:`cuml.linear_model`
   * - :obj:`~cuml.linear_model.ElasticNet`
     - ElasticNet regression.
     - :mod:`cuml.linear_model`
   * - :obj:`~cuml.linear_model.MBSGDClassifier`
     - Mini Batch SGD Classifier.
     - :mod:`cuml.linear_model`
   * - :obj:`~cuml.linear_model.MBSGDRegressor`
     - Mini Batch SGD Regressor.
     - :mod:`cuml.linear_model`

   * - :obj:`~cuml.manifold.UMAP`
     - Uniform Manifold Approximation and Projection.
     - :mod:`cuml.manifold`
   * - :obj:`~cuml.manifold.TSNE`
     - t-Distributed Stochastic Neighbor Embedding.
     - :mod:`cuml.manifold`
   * - :obj:`~cuml.manifold.SpectralEmbedding`
     - Spectral Embedding for non-linear dimensionality reduction.
     - :mod:`cuml.manifold`

   * - :obj:`~cuml.metrics.accuracy_score`
     - Accuracy classification score.
     - :mod:`cuml.metrics`
   * - :obj:`~cuml.metrics.confusion_matrix`
     - Compute confusion matrix.
     - :mod:`cuml.metrics`
   * - :obj:`~cuml.metrics.roc_auc_score`
     - Compute Area Under the ROC Curve.
     - :mod:`cuml.metrics`

   * - :obj:`~cuml.model_selection.train_test_split`
     - Split arrays into random train and test subsets.
     - :mod:`cuml.model_selection`
   * - :obj:`~cuml.model_selection.KFold`
     - K-Fold cross-validator.
     - :mod:`cuml.model_selection`

   * - :obj:`~cuml.multiclass.OneVsOneClassifier`
     - One-vs-one multiclass strategy.
     - :mod:`cuml.multiclass`
   * - :obj:`~cuml.multiclass.OneVsRestClassifier`
     - One-vs-the-rest multiclass strategy.
     - :mod:`cuml.multiclass`

   * - :obj:`~cuml.naive_bayes.BernoulliNB`
     - Naive Bayes classifier for multivariate Bernoulli models.
     - :mod:`cuml.naive_bayes`
   * - :obj:`~cuml.naive_bayes.CategoricalNB`
     - Naive Bayes classifier for categorical features.
     - :mod:`cuml.naive_bayes`
   * - :obj:`~cuml.naive_bayes.ComplementNB`
     - Complement Naive Bayes classifier.
     - :mod:`cuml.naive_bayes`
   * - :obj:`~cuml.naive_bayes.GaussianNB`
     - Gaussian Naive Bayes.
     - :mod:`cuml.naive_bayes`
   * - :obj:`~cuml.naive_bayes.MultinomialNB`
     - Naive Bayes classifier for multinomial models.
     - :mod:`cuml.naive_bayes`

   * - :obj:`~cuml.neighbors.NearestNeighbors`
     - Unsupervised nearest neighbors.
     - :mod:`cuml.neighbors`
   * - :obj:`~cuml.neighbors.KNeighborsClassifier`
     - K-Nearest Neighbors classifier.
     - :mod:`cuml.neighbors`
   * - :obj:`~cuml.neighbors.KNeighborsRegressor`
     - K-Nearest Neighbors regressor.
     - :mod:`cuml.neighbors`
   * - :obj:`~cuml.neighbors.KernelDensity`
     - Kernel Density Estimation.
     - :mod:`cuml.neighbors`

   * - :obj:`~cuml.preprocessing.StandardScaler`
     - Standardize features by removing the mean and scaling to unit variance.
     - :mod:`cuml.preprocessing`
   * - :obj:`~cuml.preprocessing.MinMaxScaler`
     - Transform features by scaling each feature to a given range.
     - :mod:`cuml.preprocessing`
   * - :obj:`~cuml.preprocessing.MaxAbsScaler`
     - Scale each feature by its maximum absolute value.
     - :mod:`cuml.preprocessing`
   * - :obj:`~cuml.preprocessing.RobustScaler`
     - Scale features using statistics that are robust to outliers.
     - :mod:`cuml.preprocessing`
   * - :obj:`~cuml.preprocessing.Normalizer`
     - Normalize samples individually to unit norm.
     - :mod:`cuml.preprocessing`
   * - :obj:`~cuml.preprocessing.LabelEncoder`
     - Encode target labels with value between 0 and n_classes-1.
     - :mod:`cuml.preprocessing`
   * - :obj:`~cuml.preprocessing.LabelBinarizer`
     - Binarize labels in a one-vs-all fashion.
     - :mod:`cuml.preprocessing`
   * - :obj:`~cuml.preprocessing.OneHotEncoder`
     - Encode categorical features as a one-hot numeric array.
     - :mod:`cuml.preprocessing`
   * - :obj:`~cuml.preprocessing.TargetEncoder`
     - Target Encoder for regression and classification targets.
     - :mod:`cuml.preprocessing`
   * - :obj:`~cuml.preprocessing.PolynomialFeatures`
     - Generate polynomial and interaction features.
     - :mod:`cuml.preprocessing`
   * - :obj:`~cuml.preprocessing.SimpleImputer`
     - Univariate imputer for completing missing values.
     - :mod:`cuml.preprocessing`

   * - :obj:`~cuml.random_projection.GaussianRandomProjection`
     - Reduce dimensionality through Gaussian random projection.
     - :mod:`cuml.random_projection`
   * - :obj:`~cuml.random_projection.SparseRandomProjection`
     - Reduce dimensionality through sparse random projection.
     - :mod:`cuml.random_projection`

   * - :obj:`~cuml.solvers.CD`
     - Coordinate Descent solver.
     - :mod:`cuml.solvers`
   * - :obj:`~cuml.solvers.QN`
     - Quasi-Newton solver.
     - :mod:`cuml.solvers`
   * - :obj:`~cuml.solvers.SGD`
     - Stochastic Gradient Descent solver.
     - :mod:`cuml.solvers`

   * - :obj:`~cuml.svm.SVC`
     - C-Support Vector Classification.
     - :mod:`cuml.svm`
   * - :obj:`~cuml.svm.SVR`
     - Epsilon-Support Vector Regression.
     - :mod:`cuml.svm`
   * - :obj:`~cuml.svm.LinearSVC`
     - Linear Support Vector Classification.
     - :mod:`cuml.svm`
   * - :obj:`~cuml.svm.LinearSVR`
     - Linear Support Vector Regression.
     - :mod:`cuml.svm`

   * - :obj:`~cuml.tsa.ARIMA`
     - ARIMA time series model.
     - :mod:`cuml.tsa`
   * - :obj:`~cuml.tsa.auto_arima.AutoARIMA`
     - Automatic ARIMA model selection.
     - :mod:`cuml.tsa`
   * - :obj:`~cuml.ExponentialSmoothing`
     - Holt-Winters Exponential Smoothing.
     - :mod:`cuml.tsa`

   * - :obj:`~cuml.dask.cluster.DBSCAN`
     - Multi-GPU DBSCAN clustering.
     - :mod:`cuml.dask`
   * - :obj:`~cuml.dask.cluster.KMeans`
     - Multi-GPU K-Means clustering.
     - :mod:`cuml.dask`
   * - :obj:`~cuml.dask.decomposition.PCA`
     - Multi-GPU Principal Component Analysis.
     - :mod:`cuml.dask`
   * - :obj:`~cuml.dask.decomposition.TruncatedSVD`
     - Multi-GPU Truncated SVD.
     - :mod:`cuml.dask`
   * - :obj:`~cuml.dask.ensemble.RandomForestClassifier`
     - Multi-GPU Random Forest classifier.
     - :mod:`cuml.dask`
   * - :obj:`~cuml.dask.ensemble.RandomForestRegressor`
     - Multi-GPU Random Forest regressor.
     - :mod:`cuml.dask`
   * - :obj:`~cuml.dask.linear_model.LinearRegression`
     - Multi-GPU Linear Regression.
     - :mod:`cuml.dask`
   * - :obj:`~cuml.dask.linear_model.Ridge`
     - Multi-GPU Ridge Regression.
     - :mod:`cuml.dask`
   * - :obj:`~cuml.dask.manifold.UMAP`
     - Multi-GPU UMAP.
     - :mod:`cuml.dask`
   * - :obj:`~cuml.dask.neighbors.NearestNeighbors`
     - Multi-GPU Nearest Neighbors.
     - :mod:`cuml.dask`
