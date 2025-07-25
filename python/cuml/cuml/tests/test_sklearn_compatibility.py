#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from sklearn.utils import estimator_checks

from cuml.cluster import KMeans
from cuml.linear_model import LogisticRegression

PER_ESTIMATOR_XFAIL_CHECKS = {
    KMeans: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_no_attributes_set_in_init": "KMeans sets attributes during init",
        "check_dont_overwrite_parameters": "KMeans overwrites parameters during fit",
        "check_estimators_unfitted": "KMeans does not raise NotFittedError before fit",
        "check_do_not_raise_errors_in_init_or_set_params": "KMeans raises errors in init or set_params",
        "check_n_features_in_after_fitting": "KMeans does not check n_features_in consistency",
        "check_sample_weights_not_an_array": "KMeans does not handle non-array sample weights",
        "check_sample_weights_list": "KMeans does not handle list sample weights",
        "check_sample_weights_shape": "KMeans does not validate sample weights shape",
        "check_sample_weight_equivalence_on_dense_data": "KMeans sample weight equivalence not implemented",
        "check_complex_data": "KMeans does not handle complex data",
        "check_dtype_object": "KMeans does not handle object dtype",
        "check_estimators_nan_inf": "KMeans does not check for NaN and inf",
        "check_estimator_sparse_tag": "KMeans does not support sparse data",
        "check_estimator_sparse_array": "KMeans does not handle sparse arrays gracefully",
        "check_estimator_sparse_matrix": "KMeans does not handle sparse matrices gracefully",
        "check_transformer_data_not_an_array": "KMeans does not handle non-array data",
        "check_parameters_default_constructible": "KMeans parameters are mutated on init",
        "check_fit_check_is_fitted": "KMeans passes check_is_fitted before being fit",
        "check_fit1d": "KMeans does not raise ValueError for 1D input",
        "check_fit2d_predict1d": "KMeans does not handle 1D prediction input gracefully",
    },
    LogisticRegression: {
        "check_estimator_tags_renamed": "No support for modern tags infrastructure",
        "check_no_attributes_set_in_init": "LogisticRegression sets attributes during init",
        "check_dont_overwrite_parameters": "LogisticRegression overwrites parameters during fit",
        "check_estimators_unfitted": "LogisticRegression does not raise NotFittedError before fit",
        "check_do_not_raise_errors_in_init_or_set_params": "LogisticRegression raises errors in init or set_params",
        "check_n_features_in_after_fitting": "LogisticRegression does not check n_features_in consistency",
        "check_sample_weights_not_an_array": "LogisticRegression does not handle non-array sample weights",
        "check_sample_weights_list": "LogisticRegression does not handle list sample weights",
        "check_sample_weight_equivalence_on_dense_data": "LogisticRegression sample weight equivalence not implemented",
        "check_sample_weight_equivalence_on_sparse_data": "LogisticRegression does not handle sparse data",
        "check_complex_data": "LogisticRegression does not handle complex data",
        "check_dtype_object": "LogisticRegression does not handle object dtype",
        "check_estimators_empty_data_messages": "LogisticRegression does not handle empty data",
        "check_estimators_nan_inf": "LogisticRegression does not check for NaN and inf",
        "check_estimator_sparse_tag": "LogisticRegression does not support sparse data",
        "check_classifier_data_not_an_array": "LogisticRegression does not handle non-array data",
        "check_classifiers_one_label": "LogisticRegression cannot train with one class",
        "check_classifiers_train": "LogisticRegression does not handle list inputs",
        "check_classifiers_train(readonly_memmap=True)": "LogisticRegression does not handle readonly memmap",
        "check_classifiers_train(readonly_memmap=True,X_dtype=float32)": "LogisticRegression does not handle readonly memmap with float32",
        "check_classifiers_regression_target": "LogisticRegression does not handle regression targets",
        "check_supervised_y_no_nan": "LogisticRegression does not check for NaN in y",
        "check_supervised_y_2d": "LogisticRegression does not handle 2D y",
        "check_class_weight_classifiers": "LogisticRegression does not handle class weights properly",
        "check_parameters_default_constructible": "LogisticRegression parameters are mutated on init",
        "check_fit2d_1sample": "LogisticRegression does not handle single sample",
        "check_set_params": "LogisticRegression does not handle set_params properly",
        "check_fit1d": "LogisticRegression does not raise ValueError for 1D input",
        "check_fit2d_predict1d": "LogisticRegression does not handle 1D prediction input gracefully",
        "check_requires_y_none": "LogisticRegression does not handle y=None",
    },
}


def get_xfails(estimator):
    return PER_ESTIMATOR_XFAIL_CHECKS.get(type(estimator), {})


def _check_sklearn_version():
    """Check if scikit-learn version is >= 1.7"""
    import sklearn
    from packaging import version

    return version.parse(sklearn.__version__) >= version.parse("1.7")


# Conditionally define the test, older versions of `parametrize_with_checks`
# do not support the `expected_failed_checks` parameter.
if _check_sklearn_version():

    @estimator_checks.parametrize_with_checks(
        [KMeans(), LogisticRegression()], expected_failed_checks=get_xfails
    )
    def test_sklearn_compatible_estimator(estimator, check):
        # Check that all estimators pass the "common estimator" checks
        # provided by scikit-learn

        check(estimator)
