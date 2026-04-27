#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from functools import partial

import pytest
from sklearn.utils import estimator_checks

from cuml.cluster import (
    DBSCAN,
    HDBSCAN,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from cuml.covariance import LedoitWolf
from cuml.decomposition import PCA, IncrementalPCA, TruncatedSVD
from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
from cuml.kernel_ridge import KernelRidge
from cuml.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from cuml.manifold import TSNE, UMAP
from cuml.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from cuml.neighbors import (
    KernelDensity,
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from cuml.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
from cuml.svm import SVC, SVR, LinearSVC, LinearSVR

# Skip these tests as parameterize_with_checks does not support
# strict_xfail in older versions of scikit-learn.
pytest.importorskip("sklearn", minversion="1.8")


ESTIMATORS = [
    GaussianRandomProjection(n_components=2),
    SparseRandomProjection(n_components=2),
    DBSCAN(),
    HDBSCAN(),
    AgglomerativeClustering(),
    KernelRidge(),
    GaussianNB(),
    ComplementNB(),
    CategoricalNB(),
    BernoulliNB(),
    MultinomialNB(),
    UMAP(),
    TSNE(),
    TruncatedSVD(),
    IncrementalPCA(),
    PCA(),
    SVR(),
    SVC(),
    LinearSVR(),
    LinearSVC(),
    NearestNeighbors(),
    KNeighborsRegressor(),
    KNeighborsClassifier(),
    KernelDensity(),
    LedoitWolf(),
    Ridge(),
    ElasticNet(),
    Lasso(),
    LinearRegression(),
    # TODO(26.08): Remove explicit default
    RandomForestClassifier(max_depth=None),
    RandomForestRegressor(max_depth=None),
    KMeans(),
    SpectralClustering(),
    LogisticRegression(),
]


XFAILS = {
    KMeans: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_sample_weight_equivalence_on_dense_data": "KMeans sample weight equivalence not implemented",
        "check_transformer_data_not_an_array": "KMeans does not handle non-array data",
    },
    KernelRidge: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_sample_weights_pandas_series": "KernelRidge does not handle pandas Series sample weights",
        "check_sample_weights_not_an_array": "KernelRidge does not handle non-array sample weights",
        "check_all_zero_sample_weights_error": "KernelRidge does not validate all-zero sample weights",
        "check_dtype_object": "KernelRidge does not handle object dtype",
        "check_estimators_empty_data_messages": "KernelRidge does not handle empty data",
        "check_estimators_nan_inf": "KernelRidge does not check for NaN and inf",
        "check_regressors_train": "KernelRidge does not handle list inputs",
        "check_regressors_train(readonly_memmap=True)": "KernelRidge does not handle readonly memmap",
        "check_regressors_train(readonly_memmap=True,X_dtype=float32)": "KernelRidge does not handle readonly memmap with float32",
        "check_regressor_data_not_an_array": "KernelRidge does not handle non-array data",
        "check_supervised_y_2d": "KernelRidge does not handle 2D y",
        "check_supervised_y_no_nan": "KernelRidge does not check for NaN in y",
        "check_requires_y_none": "KernelRidge does not handle y=None",
    },
    LogisticRegression: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_sample_weight_equivalence_on_dense_data": "LogisticRegression sample weight equivalence not implemented",
        "check_sample_weight_equivalence_on_sparse_data": "LogisticRegression does not handle sparse data",
        "check_class_weight_classifiers": "LogisticRegression does not handle class weights properly",
        "check_classifier_data_not_an_array": "LogisticRegression does not handle non-array data",
    },
    LinearRegression: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_regressor_data_not_an_array": "LinearRegression does not handle non-array data",
    },
    Ridge: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_regressor_data_not_an_array": "Ridge does not handle non-array data",
        "check_non_transformer_estimators_n_iter": "Ridge `n_iter_` may be `None`",
    },
    RandomForestRegressor: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_do_not_raise_errors_in_init_or_set_params": "RandomForestRegressor raises errors in init or set_params",
        "check_dtype_object": "RandomForestRegressor does not handle object dtype",
        "check_estimators_empty_data_messages": "RandomForestRegressor does not handle empty data",
        "check_estimators_nan_inf": "RandomForestRegressor does not check for NaN and inf",
        "check_regressors_train": "RandomForestRegressor does not handle list inputs",
        "check_regressors_train(readonly_memmap=True)": "RandomForestRegressor does not handle readonly memmap",
        "check_regressors_train(readonly_memmap=True,X_dtype=float32)": "RandomForestRegressor does not handle readonly memmap with float32",
        "check_regressor_data_not_an_array": "RandomForestRegressor does not handle non-array data",
        "check_supervised_y_2d": "RandomForestRegressor does not handle 2D y",
        "check_supervised_y_no_nan": "RandomForestRegressor does not check for NaN in y",
        "check_dict_unchanged": "RandomForestRegressor modifies input dictionaries",
        "check_requires_y_none": "RandomForestRegressor does not handle y=None",
    },
    KNeighborsClassifier: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_do_not_raise_errors_in_init_or_set_params": "KNeighborsClassifier raises errors in init or set_params",
        "check_dtype_object": "KNeighborsClassifier does not handle object dtype",
        "check_estimators_empty_data_messages": "KNeighborsClassifier does not handle empty data",
        "check_estimators_nan_inf": "KNeighborsClassifier does not check for NaN and inf",
        "check_classifier_data_not_an_array": "KNeighborsClassifier does not handle non-array data",
        "check_classifiers_train": "KNeighborsClassifier does not validate input data properly",
    },
    RandomForestClassifier: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_do_not_raise_errors_in_init_or_set_params": "RandomForestClassifier raises errors in init or set_params",
        "check_dtype_object": "RandomForestClassifier does not handle object dtype",
        "check_estimators_empty_data_messages": "RandomForestClassifier does not handle empty data",
        "check_estimators_nan_inf": "RandomForestClassifier does not check for NaN and inf",
        "check_classifier_data_not_an_array": "RandomForestClassifier does not handle non-array data",
        "check_classifiers_train": "RandomForestClassifier does not handle list inputs",
        "check_classifiers_train(readonly_memmap=True)": "RandomForestClassifier does not handle readonly memmap",
        "check_classifiers_train(readonly_memmap=True,X_dtype=float32)": "RandomForestClassifier does not handle readonly memmap with float32",
        "check_dict_unchanged": "RandomForestClassifier modifies input dictionaries",
    },
    KNeighborsRegressor: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_do_not_raise_errors_in_init_or_set_params": "KNeighborsRegressor raises errors in init or set_params",
        "check_dtype_object": "KNeighborsRegressor does not handle object dtype",
        "check_estimators_empty_data_messages": "KNeighborsRegressor does not handle empty data",
        "check_estimators_nan_inf": "KNeighborsRegressor does not check for NaN and inf",
        "check_regressors_train": "KNeighborsRegressor does not handle list inputs",
        "check_regressors_train(readonly_memmap=True)": "KNeighborsRegressor does not handle readonly memmap",
        "check_regressors_train(readonly_memmap=True,X_dtype=float32)": "KNeighborsRegressor does not handle readonly memmap with float32",
        "check_regressor_data_not_an_array": "KNeighborsRegressor does not handle non-array data",
        "check_supervised_y_2d": "KNeighborsRegressor does not handle 2D y",
        "check_supervised_y_no_nan": "KNeighborsRegressor does not check for NaN in y",
        "check_requires_y_none": "KNeighborsRegressor does not handle y=None",
    },
    NearestNeighbors: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_dtype_object": "NearestNeighbors does not handle object dtype",
        "check_estimators_empty_data_messages": "NearestNeighbors does not handle empty data",
        "check_estimators_nan_inf": "NearestNeighbors does not check for NaN and inf",
    },
    LinearSVC: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_sample_weights_not_an_array": "LinearSVC does not handle non-array sample weights",
        "check_sample_weight_equivalence_on_dense_data": "LinearSVC sample weight equivalence not implemented",
        "check_all_zero_sample_weights_error": "LinearSVC does not validate all-zero sample weights",
        "check_dtype_object": "LinearSVC does not handle object dtype",
        "check_estimators_nan_inf": "LinearSVC does not check for NaN and inf",
        "check_classifier_data_not_an_array": "LinearSVC does not handle non-array data",
        "check_classifiers_train": "LinearSVC does not handle list inputs",
        "check_classifiers_train(readonly_memmap=True)": "LinearSVC does not handle readonly memmap",
        "check_classifiers_train(readonly_memmap=True,X_dtype=float32)": "LinearSVC does not handle readonly memmap with float32",
    },
    LinearSVR: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_sample_weights_not_an_array": "LinearSVR does not handle non-array sample weights",
        "check_sample_weights_list": "LinearSVR does not handle list sample weights",
        "check_sample_weight_equivalence_on_dense_data": "LinearSVR sample weight equivalence not implemented",
        "check_all_zero_sample_weights_error": "LinearSVR does not validate all-zero sample weights",
        "check_dtype_object": "LinearSVR does not handle object dtype",
        "check_estimators_nan_inf": "LinearSVR does not check for NaN and inf",
        "check_regressors_train": "LinearSVR does not handle list inputs",
        "check_regressors_train(readonly_memmap=True)": "LinearSVR does not handle readonly memmap",
        "check_regressors_train(readonly_memmap=True,X_dtype=float32)": "LinearSVR does not handle readonly memmap with float32",
        "check_regressor_data_not_an_array": "LinearSVR does not handle non-array data",
        "check_supervised_y_2d": "LinearSVR does not handle 2D y",
        "check_supervised_y_no_nan": "LinearSVR does not check for NaN in y",
        "check_requires_y_none": "LinearSVR does not handle y=None",
    },
    SVC: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_sample_weights_not_an_array": "SVC does not handle non-array sample weights",
        "check_sample_weight_equivalence_on_dense_data": "SVC sample weight equivalence not implemented",
        "check_sample_weight_equivalence_on_sparse_data": "SVC does not handle sparse data",
        "check_all_zero_sample_weights_error": "SVC does not validate all-zero sample weights",
        "check_estimators_nan_inf": "SVC does not check for NaN and inf",
        "check_classifier_data_not_an_array": "SVC does not handle non-array data",
        "check_classifiers_train": "SVC does not handle list inputs",
        "check_classifiers_train(readonly_memmap=True)": "SVC does not handle readonly memmap",
        "check_classifiers_train(readonly_memmap=True,X_dtype=float32)": "SVC does not handle readonly memmap with float32",
    },
    SVR: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_sample_weights_not_an_array": "SVR does not handle non-array sample weights",
        "check_sample_weights_list": "SVR does not handle list sample weights",
        "check_sample_weight_equivalence_on_dense_data": "SVR sample weight equivalence not implemented",
        "check_sample_weight_equivalence_on_sparse_data": "SVR does not handle sparse data",
        "check_all_zero_sample_weights_error": "SVR does not validate all-zero sample weights",
        "check_dtype_object": "SVR does not handle object dtype",
        "check_estimators_nan_inf": "SVR does not check for NaN and inf",
        "check_regressors_train": "SVR does not handle list inputs",
        "check_regressors_train(readonly_memmap=True)": "SVR does not handle readonly memmap",
        "check_regressors_train(readonly_memmap=True,X_dtype=float32)": "SVR does not handle readonly memmap with float32",
        "check_regressor_data_not_an_array": "SVR does not handle non-array data",
        "check_supervised_y_2d": "SVR does not handle 2D y",
        "check_supervised_y_no_nan": "SVR does not check for NaN in y",
        "check_requires_y_none": "SVR does not handle y=None",
    },
    PCA: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_transformer_data_not_an_array": "PCA does not handle non-array data",
        "check_fit2d_1sample": "PCA does not handle single sample",
        "check_fit2d_1feature": "PCA does not handle single feature",
    },
    IncrementalPCA: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_transformer_data_not_an_array": "IncrementalPCA does not handle non-array data",
    },
    TruncatedSVD: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_transformer_data_not_an_array": "TruncatedSVD does not handle non-array data",
        "check_fit2d_1sample": "TruncatedSVD does not handle single sample",
        "check_fit2d_1feature": "TruncatedSVD does not handle single feature",
    },
    TSNE: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_dont_overwrite_parameters": "TSNE overwrites parameters during fit",
        "check_dtype_object": "TSNE does not handle object dtype",
        "check_estimators_empty_data_messages": "TSNE does not handle empty data",
        "check_pipeline_consistency": "TSNE results are not deterministic",
        "check_estimators_nan_inf": "TSNE does not check for NaN and inf",
        "check_methods_sample_order_invariance": "TSNE results depend on sample order",
        "check_methods_subset_invariance": "TSNE results depend on data subset",
        "check_fit2d_1sample": "TSNE does not handle single sample",
        "check_fit2d_1feature": "TSNE does not handle single feature",
        "check_fit2d_predict1d": "TSNE only supports n_components = 2",
    },
    UMAP: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_dtype_object": "UMAP does not handle object dtype",
        "check_transformer_data_not_an_array": "UMAP does not handle non-array data",
        "check_methods_sample_order_invariance": "UMAP results depend on sample order",
        "check_transformer_general": "UMAP does not have consistent fit_transform and transform outputs",
        "check_methods_subset_invariance": "UMAP results depend on data subset",
    },
    Lasso: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_regressor_data_not_an_array": "Lasso does not handle non-array data",
        "check_sample_weight_equivalence_on_sparse_data": "Lasso QN solver has issues with sample weights",
    },
    ElasticNet: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_regressor_data_not_an_array": "ElasticNet does not handle non-array data",
        "check_sample_weight_equivalence_on_sparse_data": "ElasticNet QN solver has issues with sample weights",
    },
    KernelDensity: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_sample_weights_not_an_array": "KernelDensity does not handle non-array sample weights",
        "check_sample_weights_list": "KernelDensity does not handle list sample weights",
        "check_all_zero_sample_weights_error": "KernelDensity does not validate all-zero sample weights",
        "check_dtype_object": "KernelDensity does not handle object dtype",
        "check_estimators_nan_inf": "KernelDensity does not check for NaN and inf",
    },
    LedoitWolf: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_dtype_object": "LedoitWolf does not handle object dtype",
        "check_estimators_empty_data_messages": "LedoitWolf does not handle empty data",
        "check_estimators_nan_inf": "LedoitWolf does not check for NaN and inf",
    },
    DBSCAN: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
    },
    HDBSCAN: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
    },
    AgglomerativeClustering: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
    },
    SpectralClustering: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
    },
    GaussianNB: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_estimators_dtypes": "GaussianNB does not handle dtypes properly",
        "check_sample_weights_pandas_series": "GaussianNB does not handle pandas Series sample weights",
        "check_sample_weights_not_an_array": "GaussianNB does not handle non-array sample weights",
        "check_sample_weights_shape": "GaussianNB does not validate sample weights shape",
        "check_all_zero_sample_weights_error": "GaussianNB does not validate all-zero sample weights",
        "check_dtype_object": "GaussianNB does not handle object dtype",
        "check_estimators_empty_data_messages": "GaussianNB does not handle empty data",
        "check_estimators_nan_inf": "GaussianNB does not check for NaN and inf",
        "check_classifier_data_not_an_array": "GaussianNB does not handle non-array data",
        "check_classifiers_classes": "GaussianNB does not handle string data properly",
        "check_classifiers_train": "GaussianNB does not handle list inputs",
        "check_classifiers_train(readonly_memmap=True)": "GaussianNB does not handle readonly memmap",
        "check_classifiers_train(readonly_memmap=True,X_dtype=float32)": "GaussianNB does not handle readonly memmap with float32",
        "check_classifiers_regression_target": "GaussianNB does not handle regression targets",
        "check_supervised_y_no_nan": "GaussianNB does not check for NaN in y",
        "check_supervised_y_2d": "GaussianNB does not handle 2D y",
        "check_requires_y_none": "GaussianNB does not handle y=None",
        "check_sample_weights_list": "GaussianNB does not handle list sample weights",
    },
    GaussianRandomProjection: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_transformer_data_not_an_array": "GaussianRandomProjection does not handle non-array data",
    },
    SparseRandomProjection: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_transformer_data_not_an_array": "SparseRandomProjection does not handle non-array data",
    },
    BernoulliNB: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_estimators_dtypes": "BernoulliNB expects specific dtypes, not bool",
        "check_sample_weights_pandas_series": "BernoulliNB does not handle pandas Series sample weights",
        "check_sample_weights_not_an_array": "BernoulliNB does not handle non-array sample weights",
        "check_sample_weights_shape": "BernoulliNB does not validate sample weights shape",
        "check_sample_weight_equivalence_on_dense_data": "BernoulliNB sample weight equivalence not implemented",
        "check_all_zero_sample_weights_error": "BernoulliNB does not validate all-zero sample weights",
        "check_dtype_object": "BernoulliNB does not handle object dtype",
        "check_estimators_empty_data_messages": "BernoulliNB does not provide proper error messages for empty data",
        "check_estimators_nan_inf": "BernoulliNB does not check for NaN and inf",
        "check_classifier_data_not_an_array": "BernoulliNB does not handle list inputs",
        "check_classifiers_classes": "BernoulliNB does not handle string labels properly",
        "check_classifiers_train": "BernoulliNB does not handle list inputs",
        "check_classifiers_train(readonly_memmap=True)": "BernoulliNB does not handle list inputs",
        "check_classifiers_train(readonly_memmap=True,X_dtype=float32)": "BernoulliNB does not handle list inputs",
        "check_classifiers_regression_target": "BernoulliNB does not validate target is classification",
        "check_supervised_y_no_nan": "BernoulliNB does not check for NaN in y",
        "check_supervised_y_2d": "BernoulliNB does not handle 2D y input gracefully",
        "check_requires_y_none": "BernoulliNB does not require y for fit",
    },
    ComplementNB: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_estimators_dtypes": "ComplementNB expects specific dtypes, not bool",
        "check_sample_weights_pandas_series": "ComplementNB does not handle pandas Series sample weights",
        "check_sample_weights_not_an_array": "ComplementNB does not handle non-array sample weights",
        "check_sample_weights_shape": "ComplementNB does not validate sample weights shape",
        "check_sample_weight_equivalence_on_dense_data": "ComplementNB sample weight equivalence not implemented",
        "check_all_zero_sample_weights_error": "ComplementNB does not validate all-zero sample weights",
        "check_dtype_object": "ComplementNB does not handle object dtype",
        "check_estimators_empty_data_messages": "ComplementNB does not provide proper error messages for empty data",
        "check_estimators_nan_inf": "ComplementNB does not check for NaN and inf",
        "check_classifier_data_not_an_array": "ComplementNB does not handle list inputs",
        "check_classifiers_classes": "ComplementNB does not handle string labels properly",
        "check_classifiers_train": "ComplementNB does not handle list inputs",
        "check_classifiers_train(readonly_memmap=True)": "ComplementNB does not handle list inputs",
        "check_classifiers_train(readonly_memmap=True,X_dtype=float32)": "ComplementNB does not handle list inputs",
        "check_classifiers_regression_target": "ComplementNB does not validate target is classification",
        "check_supervised_y_no_nan": "ComplementNB does not check for NaN in y",
        "check_supervised_y_2d": "ComplementNB does not handle 2D y input gracefully",
        "check_requires_y_none": "ComplementNB does not require y for fit",
    },
    CategoricalNB: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_estimators_dtypes": "CategoricalNB expects specific dtypes, not bool",
        "check_sample_weights_pandas_series": "CategoricalNB does not handle pandas Series sample weights",
        "check_sample_weights_not_an_array": "CategoricalNB does not handle non-array sample weights",
        "check_sample_weights_shape": "CategoricalNB does not validate sample weights shape",
        "check_sample_weight_equivalence_on_dense_data": "CategoricalNB sample weight equivalence not implemented",
        "check_all_zero_sample_weights_error": "CategoricalNB does not validate all-zero sample weights",
        "check_dtype_object": "CategoricalNB does not handle object dtype",
        "check_estimators_empty_data_messages": "CategoricalNB does not provide proper error messages for empty data",
        "check_estimators_nan_inf": "CategoricalNB does not check for NaN and inf",
        "check_classifier_data_not_an_array": "CategoricalNB does not handle list inputs",
        "check_classifiers_classes": "CategoricalNB does not handle string labels properly",
        "check_classifiers_train": "CategoricalNB does not handle list inputs",
        "check_classifiers_train(readonly_memmap=True)": "CategoricalNB does not handle list inputs",
        "check_classifiers_train(readonly_memmap=True,X_dtype=float32)": "CategoricalNB does not handle list inputs",
        "check_classifiers_regression_target": "CategoricalNB does not validate target is classification",
        "check_supervised_y_no_nan": "CategoricalNB does not check for NaN in y",
        "check_supervised_y_2d": "CategoricalNB does not handle 2D y input gracefully",
        "check_requires_y_none": "CategoricalNB does not require y for fit",
    },
    MultinomialNB: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_estimators_dtypes": "MultinomialNB does not handle all dtypes properly",
        "check_sample_weights_pandas_series": "MultinomialNB does not handle pandas Series sample weights",
        "check_sample_weights_not_an_array": "MultinomialNB does not handle non-array sample weights",
        "check_sample_weights_shape": "MultinomialNB does not validate sample weights shape",
        "check_sample_weight_equivalence_on_dense_data": "MultinomialNB sample weight equivalence not implemented",
        "check_all_zero_sample_weights_error": "MultinomialNB does not validate all-zero sample weights",
        "check_dtype_object": "MultinomialNB does not handle object dtype",
        "check_estimators_empty_data_messages": "MultinomialNB does not provide proper error messages for empty data",
        "check_estimators_nan_inf": "MultinomialNB does not check for NaN and inf",
        "check_classifier_data_not_an_array": "MultinomialNB does not handle non-array data",
        "check_classifiers_classes": "MultinomialNB does not handle string labels properly",
        "check_classifiers_train": "MultinomialNB does not validate X and y shapes are consistent",
        "check_classifiers_train(readonly_memmap=True)": "MultinomialNB does not validate X and y shapes are consistent",
        "check_classifiers_train(readonly_memmap=True,X_dtype=float32)": "MultinomialNB does not validate X and y shapes are consistent",
        "check_classifiers_regression_target": "MultinomialNB does not validate target is classification",
        "check_supervised_y_no_nan": "MultinomialNB does not check for NaN in y",
        "check_supervised_y_2d": "MultinomialNB does not handle 2D y input gracefully",
        "check_requires_y_none": "MultinomialNB does not require y for fit",
    },
}


# Sanity check that xfails listed have at least one estimator instance
if missing := set(XFAILS).difference((type(est) for est in ESTIMATORS)):
    raise ValueError(
        f"xfails defined for {missing}, but that estimator isn't tested!"
    )


def _check_name(check):
    if hasattr(check, "__wrapped__"):
        return _check_name(check.__wrapped__)
    return (
        check.func.__name__ if isinstance(check, partial) else check.__name__
    )


@estimator_checks.parametrize_with_checks(
    ESTIMATORS,
    expected_failed_checks=lambda est: XFAILS.get(type(est), {}),
    xfail_strict=True,
)
@pytest.mark.filterwarnings(
    "ignore:ValueError occurred during set_params.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:TypeError occurred during set_params.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:perplexity.*should be less than n_samples.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:Estimator's parameters changed after set_params raised ValueError:UserWarning"
)
@pytest.mark.filterwarnings("ignore:Changing solver to 'svd'.*:UserWarning")
@pytest.mark.filterwarnings("ignore:The number of bins.*:UserWarning")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_sklearn_compatible_estimator(estimator, check):
    # Check that all estimators pass the "common estimator" checks
    # provided by scikit-learn
    check_name = _check_name(check)

    if check_name in ["check_estimators_nan_inf"] and isinstance(
        estimator, UMAP
    ):
        pytest.skip("UMAP does not handle Nans and infinities")

    check(estimator)
