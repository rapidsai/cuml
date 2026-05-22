#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from sklearn.utils import estimator_checks

from cuml.cluster import (
    DBSCAN,
    HDBSCAN,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from cuml.covariance import EmpiricalCovariance, LedoitWolf
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
    UMAP(n_neighbors=5),
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
    EmpiricalCovariance(),
    LedoitWolf(),
    Ridge(),
    ElasticNet(),
    Lasso(),
    LinearRegression(),
    # rapids-pre-commit-hooks: disable-next-line
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
        "check_regressor_data_not_an_array": "KernelRidge does not handle non-array data",
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
        "check_regressor_data_not_an_array": "RandomForestRegressor does not handle non-array data",
        "check_dict_unchanged": "RandomForestRegressor modifies input dictionaries",
    },
    RandomForestClassifier: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_do_not_raise_errors_in_init_or_set_params": "RandomForestClassifier raises errors in init or set_params",
        "check_classifier_data_not_an_array": "RandomForestClassifier does not handle non-array data",
        "check_dict_unchanged": "RandomForestClassifier modifies input dictionaries",
    },
    KNeighborsClassifier: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_do_not_raise_errors_in_init_or_set_params": "KNeighborsClassifier raises errors in init or set_params",
        "check_classifier_data_not_an_array": "KNeighborsClassifier does not handle non-array data",
    },
    KNeighborsRegressor: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_do_not_raise_errors_in_init_or_set_params": "KNeighborsRegressor raises errors in init or set_params",
        "check_regressor_data_not_an_array": "KNeighborsRegressor does not handle non-array data",
        "check_supervised_y_2d": "KNeighborsRegressor does not handle 2D y",
    },
    NearestNeighbors: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
    },
    LinearSVC: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_classifier_data_not_an_array": "LinearSVC does not handle non-array data",
        "check_sample_weight_equivalence_on_dense_data": "LinearSVC sample weight equivalence not implemented",
    },
    LinearSVR: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_sample_weight_equivalence_on_dense_data": "LinearSVR sample weight equivalence not implemented",
        "check_regressor_data_not_an_array": "LinearSVR does not handle non-array data",
    },
    SVC: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_sample_weight_equivalence_on_dense_data": "SVC sample weight equivalence not implemented",
        "check_sample_weight_equivalence_on_sparse_data": "SVC does not handle sparse data",
        "check_classifier_data_not_an_array": "SVC does not handle non-array data",
    },
    SVR: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_sample_weight_equivalence_on_dense_data": "SVR sample weight equivalence not implemented",
        "check_sample_weight_equivalence_on_sparse_data": "SVR does not handle sparse data",
        "check_regressor_data_not_an_array": "SVR does not handle non-array data",
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
        "check_pipeline_consistency": "TSNE results are not deterministic",
        "check_methods_sample_order_invariance": "TSNE results depend on sample order",
        "check_methods_subset_invariance": "TSNE results depend on data subset",
        "check_fit2d_1sample": "TSNE does not handle single sample",
        "check_fit2d_1feature": "TSNE does not handle single feature",
        "check_fit2d_predict1d": "TSNE only supports n_components = 2",
    },
    UMAP: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
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
    },
    EmpiricalCovariance: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
    },
    LedoitWolf: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
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
    GaussianRandomProjection: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_transformer_data_not_an_array": "GaussianRandomProjection does not handle non-array data",
    },
    SparseRandomProjection: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_transformer_data_not_an_array": "SparseRandomProjection does not handle non-array data",
    },
    GaussianNB: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_classifier_data_not_an_array": "GaussianNB does not handle non-array data",
    },
    BernoulliNB: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_classifier_data_not_an_array": "bug in reflection prevents this",
        "check_sample_weights_pandas_series": "sample_weight not implemented",
        "check_sample_weights_not_an_array": "sample_weight not implemented",
        "check_sample_weights_shape": "sample_weight not implemented",
        "check_sample_weight_equivalence_on_dense_data": "sample_weight not implemented",
        "check_sample_weights_list": "sample_weight not implemented",
        "check_sample_weights_not_overwritten": "sample_weight not implemented",
        "check_sample_weight_equivalence_on_sparse_data": "sample_weight not implemented",
        "check_classifiers_one_label_sample_weights": "sample_weight not implemented",
    },
    ComplementNB: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_classifier_data_not_an_array": "bug in reflection prevents this",
        "check_sample_weights_pandas_series": "sample_weight not implemented",
        "check_sample_weights_not_an_array": "sample_weight not implemented",
        "check_sample_weights_shape": "sample_weight not implemented",
        "check_sample_weight_equivalence_on_dense_data": "sample_weight not implemented",
        "check_sample_weights_list": "sample_weight not implemented",
        "check_sample_weights_not_overwritten": "sample_weight not implemented",
        "check_sample_weight_equivalence_on_sparse_data": "sample_weight not implemented",
        "check_classifiers_one_label_sample_weights": "sample_weight not implemented",
    },
    CategoricalNB: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_classifier_data_not_an_array": "bug in reflection prevents this",
        "check_sample_weights_pandas_series": "sample_weight not implemented",
        "check_sample_weights_not_an_array": "sample_weight not implemented",
        "check_sample_weights_shape": "sample_weight not implemented",
        "check_sample_weight_equivalence_on_dense_data": "sample_weight not implemented",
        "check_sample_weights_list": "sample_weight not implemented",
        "check_sample_weights_not_overwritten": "sample_weight not implemented",
        "check_sample_weight_equivalence_on_sparse_data": "sample_weight not implemented",
        "check_classifiers_one_label_sample_weights": "sample_weight not implemented",
    },
    MultinomialNB: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_classifier_data_not_an_array": "bug in reflection prevents this",
        "check_sample_weights_pandas_series": "sample_weight not implemented",
        "check_sample_weights_not_an_array": "sample_weight not implemented",
        "check_sample_weights_shape": "sample_weight not implemented",
        "check_sample_weight_equivalence_on_dense_data": "sample_weight not implemented",
        "check_sample_weights_list": "sample_weight not implemented",
        "check_sample_weights_not_overwritten": "sample_weight not implemented",
        "check_sample_weight_equivalence_on_sparse_data": "sample_weight not implemented",
        "check_classifiers_one_label_sample_weights": "sample_weight not implemented",
    },
}


# Sanity check that xfails listed have at least one estimator instance
if missing := set(XFAILS).difference((type(est) for est in ESTIMATORS)):
    raise ValueError(
        f"xfails defined for {missing}, but that estimator isn't tested!"
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
    check(estimator)
