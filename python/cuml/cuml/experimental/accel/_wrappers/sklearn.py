#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

from ..estimator_proxy import intercept

wrapped_estimators = {
    "KMeans": ("cuml.cluster", "KMeans"),
    "DBSCAN": ("cuml.cluster", "DBSCAN"),
    "PCA": ("cuml.decomposition", "PCA"),
    "TruncatedSVD": ("cuml.decomposition", "TruncatedSVD"),
    "KernelRidge": ("cuml.kernel_ridge", "KernelRidge"),
    "LinearRegression": "cuml.linear_model.LinearRegression",
    "LogisticRegression": ("cuml.linear_model", "LogisticRegression"),
    "ElasticNet": ("cuml.linear_model", "ElasticNet"),
    "Ridge": ("cuml.linear_model", "Ridge"),
    "Lasso": ("cuml.linear_model", "Lasso"),
    "TSNE": ("cuml.manifold", "TSNE"),
    "NearestNeighbors": ("cuml.neighbors", "NearestNeighbors"),
    "KNeighborsClassifier": ("cuml.neighbors", "KNeighborsClassifier"),
    "KNeighborsRegressor": ("cuml.neighbors", "KNeighborsRegressor"),
}


###############################################################################
#                              Clustering Estimators                          #
###############################################################################

# AgglomerativeClustering = intercept(original_module="sklearn.cluster",
#                              accelerated_module="cuml.cluster",
#                              original_class_name="AgglomerativeClustering")

KMeans = intercept(
    original_module="sklearn.cluster",
    accelerated_module="cuml.cluster",
    original_class_name="KMeans",
)

DBSCAN = intercept(
    original_module="sklearn.cluster",
    accelerated_module="cuml.cluster",
    original_class_name="DBSCAN",
)

# HDBSCAN = intercept(
#     original_module="sklearn.cluster",
#     accelerated_module="cuml.cluster",
#     original_class_name="HDBSCAN",
# )


###############################################################################
#                              Decomposition Estimators                       #
###############################################################################


PCA = intercept(
    original_module="sklearn.decomposition",
    accelerated_module="cuml.decomposition",
    original_class_name="PCA",
)


# IncrementalPCA = intercept(original_module="sklearn.decomposition",
#                              accelerated_module="cuml.decomposition",
#                              original_class_name="IncrementalPCA")


TruncatedSVD = intercept(
    original_module="sklearn.decomposition",
    accelerated_module="cuml.decomposition",
    original_class_name="TruncatedSVD",
)


###############################################################################
#                              Ensemble Estimators                            #
###############################################################################


# RandomForestClassifier = intercept(original_module="sklearn.ensemble",
#                              accelerated_module="cuml.ensemble",
#                              original_class_name="RandomForestClassifier")

# RandomForestRegressor = intercept(original_module="sklearn.decomposition",
#                              accelerated_module="cuml.decomposition",
#                              original_class_name="RandomForestRegressor")


###############################################################################
#                              Linear Estimators                              #
###############################################################################

KernelRidge = intercept(
    original_module="sklearn.kernel_ridge",
    accelerated_module="cuml.kernel_ridge",
    original_class_name="KernelRidge",
)

LinearRegression = intercept(
    original_module="sklearn.linear_model",
    accelerated_module="cuml.linear_model",
    original_class_name="LinearRegression",
)

LogisticRegression = intercept(
    original_module="sklearn.linear_model",
    accelerated_module="cuml.linear_model",
    original_class_name="LogisticRegression",
)

ElasticNet = intercept(
    original_module="sklearn.linear_model",
    accelerated_module="cuml.linear_model",
    original_class_name="ElasticNet",
)

Ridge = intercept(
    original_module="sklearn.linear_model",
    accelerated_module="cuml.linear_model",
    original_class_name="Ridge",
)

Lasso = intercept(
    original_module="sklearn.linear_model",
    accelerated_module="cuml.linear_model",
    original_class_name="Lasso",
)


###############################################################################
#                              Manifold Estimators                            #
###############################################################################

TSNE = intercept(
    original_module="sklearn.manifold",
    accelerated_module="cuml.manifold",
    original_class_name="TSNE",
)


###############################################################################
#                              Bayes Estimators                               #
###############################################################################

# GaussianNB = intercept(original_module="sklearn.naive_bayes",
#                              accelerated_module="cuml.naive_bayes",
#                              original_class_name="GaussianNB")

# MultinomialNB = intercept(original_module="sklearn.naive_bayes",
#                              accelerated_module="cuml.naive_bayes",
#                              original_class_name="MultinomialNB")

# BernoulliNB = intercept(original_module="sklearn.naive_bayes",
#                              accelerated_module="cuml.naive_bayes",
#                              original_class_name="BernoulliNB")

# ComplementNB = intercept(original_module="sklearn.naive_bayes",
#                              accelerated_module="cuml.naive_bayes",
#                              original_class_name="ComplementNB")


###############################################################################
#                              Neighbors Estimators                           #
###############################################################################


NearestNeighbors = intercept(
    original_module="sklearn.neighbors",
    accelerated_module="cuml.neighbors",
    original_class_name="NearestNeighbors",
)

KNeighborsClassifier = intercept(
    original_module="sklearn.neighbors",
    accelerated_module="cuml.neighbors",
    original_class_name="KNeighborsClassifier",
)

KNeighborsRegressor = intercept(
    original_module="sklearn.neighbors",
    accelerated_module="cuml.neighbors",
    original_class_name="KNeighborsRegressor",
)

###############################################################################
#                              Rand Proj Estimators                           #
###############################################################################


# GaussianRandomProjection = intercept(original_module="sklearn.random_projection",
#                              accelerated_module="cuml.random_projection",
#                              original_class_name="GaussianRandomProjection")


# SparseRandomProjection = intercept(original_module="sklearn.random_projection",
#                              accelerated_module="cuml.random_projection",
#                              original_class_name="SparseRandomProjection")


###############################################################################
#                              SVM Estimators                                 #
###############################################################################


# LinearSVC = intercept(original_module="sklearn.svm",
#                              accelerated_module="cuml.svm",
#                              original_class_name="LinearSVC")

# LinearSVR = intercept(original_module="sklearn.svm",
#                              accelerated_module="cuml.svm",
#                              original_class_name="LinearSVR")

# SVC = intercept(original_module="sklearn.svm",
#                              accelerated_module="cuml.svm",
#                              original_class_name="SVC")

# SVR = intercept(original_module="sklearn.svm",
#                              accelerated_module="cuml.svm",
#                              original_class_name="SVR")


###############################################################################
#                              TSA Estimators                                 #
###############################################################################


# not supported yet
