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

from cuml.accel.estimator_proxy import intercept


###############################################################################
#                              Clustering Estimators                          #
###############################################################################

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


###############################################################################
#                              Decomposition Estimators                       #
###############################################################################


PCA = intercept(
    original_module="sklearn.decomposition",
    accelerated_module="cuml.decomposition",
    original_class_name="PCA",
)


TruncatedSVD = intercept(
    original_module="sklearn.decomposition",
    accelerated_module="cuml.decomposition",
    original_class_name="TruncatedSVD",
)


###############################################################################
#                              Linear Estimators                              #
###############################################################################

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
#                              Ensemble  Estimators                           #
###############################################################################


RandomForestRegressor = intercept(
    original_module="sklearn.ensemble",
    accelerated_module="cuml.ensemble",
    original_class_name="RandomForestRegressor",
)

RandomForestClassifier = intercept(
    original_module="sklearn.ensemble",
    accelerated_module="cuml.ensemble",
    original_class_name="RandomForestClassifier",
)
