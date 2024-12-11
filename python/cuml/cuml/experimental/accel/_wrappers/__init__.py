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

wrapped_estimators = {
    "KMeans": ("cuml.cluster", "KMeans"),
    "DBSCAN": ("cuml.cluster", "DBSCAN"),
    "PCA": ("cuml.decomposition", "PCA"),
    "TruncatedSVD": ("cuml.decomposition", "TruncatedSVD"),
    "KernelRidge": ("cuml.kernel_ridge", "KernelRidge"),
    "LinearRegression": ("cuml.linear_model", "LinearRegression"),
    "LogisticRegression": ("cuml.linear_model", "LogisticRegression"),
    "ElasticNet": ("cuml.linear_model", "ElasticNet"),
    "Ridge": ("cuml.linear_model", "Ridge"),
    "Lasso": ("cuml.linear_model", "Lasso"),
    "TSNE": ("cuml.manifold", "TSNE"),
    "NearestNeighbors": ("cuml.neighbors", "NearestNeighbors"),
    "KNeighborsClassifier": ("cuml.neighbors", "KNeighborsClassifier"),
    "KNeighborsRegressor": ("cuml.neighbors", "KNeighborsRegressor"),
    "UMAP": ("cuml.manifold", "UMAP"),
    "HDBSCAN": ("cuml.cluster", "HDBSCAN"),
}
