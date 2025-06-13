/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuml/common/distance_type.hpp>

#include <cstdint>

namespace raft {
class handle_t;
}

namespace ML {

namespace Metrics {

/**
 * Calculates the "Coefficient of Determination" (R-Squared) score
 * normalizing the sum of squared errors by the total sum of squares
 * with single precision.
 *
 * This score indicates the proportionate amount of variation in an
 * expected response variable is explained by the independent variables
 * in a linear regression model. The larger the R-squared value, the
 * more variability is explained by the linear regression model.
 *
 * @param handle: raft::handle_t
 * @param y: Array of ground-truth response variables
 * @param y_hat: Array of predicted response variables
 * @param n: Number of elements in y and y_hat
 * @return: The R-squared value.
 */
float r2_score_py(const raft::handle_t& handle, float* y, float* y_hat, int n);

/**
 * Calculates the "Coefficient of Determination" (R-Squared) score
 * normalizing the sum of squared errors by the total sum of squares
 * with double precision.
 *
 * This score indicates the proportionate amount of variation in an
 * expected response variable is explained by the independent variables
 * in a linear regression model. The larger the R-squared value, the
 * more variability is explained by the linear regression model.
 *
 * @param handle: raft::handle_t
 * @param y: Array of ground-truth response variables
 * @param y_hat: Array of predicted response variables
 * @param n: Number of elements in y and y_hat
 * @return: The R-squared value.
 */
double r2_score_py(const raft::handle_t& handle, double* y, double* y_hat, int n);

/**
 * Calculates the "rand index"
 *
 * This metric is a measure of similarity between two data clusterings.
 *
 * @param handle: raft::handle_t
 * @param y: Array of response variables of the first clustering classifications
 * @param y_hat: Array of response variables of the second clustering classifications
 * @param n: Number of elements in y and y_hat
 * @return: The rand index value
 */

double rand_index(const raft::handle_t& handle, double* y, double* y_hat, int n);

/**
 * Calculates the "Silhouette Score"
 *
 * The Silhouette Coefficient is calculated using the mean intra-cluster distance (a)
 * and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient
 * for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample
 * and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient
 * is only defined if number of labels is 2 <= n_labels <= n_samples - 1.
 *
 * @param handle: raft::handle_t
 * @param y: Array of data samples with dimensions (nRows x nCols)
 * @param nRows: number of data samples
 * @param nCols: number of features
 * @param labels: Array containing labels for every data sample (1 x nRows)
 * @param nLabels: number of Labels
 * @param metric: the numerical value that maps to the type of distance metric to be used in the
 * calculations
 * @param silScores: Array that is optionally taken in as input if required to be populated with the
 * silhouette score for every sample (1 x nRows), else nullptr is passed
 */
double silhouette_score(const raft::handle_t& handle,
                        double* y,
                        int nRows,
                        int nCols,
                        int* labels,
                        int nLabels,
                        double* silScores,
                        ML::distance::DistanceType metric);

namespace Batched {
/**
 * Calculates Batched "Silhouette Score" by tiling the pairwise distance matrix to remove use of
 * quadratic memory
 *
 * The Silhouette Coefficient is calculated using the mean intra-cluster distance (a)
 * and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient
 * for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample
 * and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient
 * is only defined if number of labels is 2 <= n_labels <= n_samples - 1.
 *
 * @param[in] handle: raft::handle_t
 * @param[in] X: Array of data samples with dimensions (n_rows x n_cols)
 * @param[in] n_rows: number of data samples
 * @param[in] n_cols: number of features
 * @param[in] y: Array containing labels for every data sample (1 x n_rows)
 * @param[in] n_labels: number of Labels
 * @param[in] metric: the numerical value that maps to the type of distance metric to be used in the
 * calculations
 * @param[in] chunk: the row-wise chunk size on which the pairwise distance matrix is tiled
 * @param[out] scores: Array that is optionally taken in as input if required to be populated with
 * the silhouette score for every sample (1 x nRows), else nullptr is passed
 */
float silhouette_score(const raft::handle_t& handle,
                       float* X,
                       int n_rows,
                       int n_cols,
                       int* y,
                       int n_labels,
                       float* scores,
                       int chunk,
                       ML::distance::DistanceType metric);
double silhouette_score(const raft::handle_t& handle,
                        double* X,
                        int n_rows,
                        int n_cols,
                        int* y,
                        int n_labels,
                        double* scores,
                        int chunk,
                        ML::distance::DistanceType metric);

}  // namespace Batched
/**
 * Calculates the "adjusted rand index"
 *
 * This metric is the corrected-for-chance version of the rand index
 *
 * @param handle: raft::handle_t
 * @param y: Array of response variables of the first clustering classifications
 * @param y_hat: Array of response variables of the second clustering classifications
 * @param n: Number of elements in y and y_hat
 * @return: The adjusted rand index value
 * @{
 */
double adjusted_rand_index(const raft::handle_t& handle,
                           const int64_t* y,
                           const int64_t* y_hat,
                           const int64_t n);
double adjusted_rand_index(const raft::handle_t& handle,
                           const int* y,
                           const int* y_hat,
                           const int n);
/** @} */

/**
 * Calculates the "Kullback-Leibler Divergence"
 *
 * The KL divergence tells us how well the probability distribution Q
 * approximates the probability distribution P
 * It is often also used as a 'distance metric' between two probability distributions (not
 * symmetric)
 *
 * @param handle: raft::handle_t
 * @param y: Array of probabilities corresponding to distribution P
 * @param y_hat: Array of probabilities corresponding to distribution Q
 * @param n: Number of elements in y and y_hat
 * @return: The KL Divergence value
 */
double kl_divergence(const raft::handle_t& handle, const double* y, const double* y_hat, int n);

/**
 * Calculates the "Kullback-Leibler Divergence"
 *
 * The KL divergence tells us how well the probability distribution Q
 * approximates the probability distribution P
 * It is often also used as a 'distance metric' between two probability distributions (not
 * symmetric)
 *
 * @param handle: raft::handle_t
 * @param y: Array of probabilities corresponding to distribution P
 * @param y_hat: Array of probabilities corresponding to distribution Q
 * @param n: Number of elements in y and y_hat
 * @return: The KL Divergence value
 */
float kl_divergence(const raft::handle_t& handle, const float* y, const float* y_hat, int n);

/**
 * Calculates the "entropy" of a labelling
 *
 * This metric is a measure of the purity/polarity of the clustering
 *
 * @param handle: raft::handle_t
 * @param y: Array of response variables of the clustering
 * @param n: Number of elements in y
 * @param lower_class_range: the lowest value in the range of classes
 * @param upper_class_range: the highest value in the range of classes
 * @return: The entropy value of the clustering
 */
double entropy(const raft::handle_t& handle,
               const int* y,
               const int n,
               const int lower_class_range,
               const int upper_class_range);

/**
 * Calculates the "Mutual Information score" between two clusters
 *
 * Mutual Information is a measure of the similarity between two labels of
 * the same data.
 *
 * @param handle: raft::handle_t
 * @param y: Array of response variables of the first clustering classifications
 * @param y_hat: Array of response variables of the second clustering classifications
 * @param n: Number of elements in y and y_hat
 * @param lower_class_range: the lowest value in the range of classes
 * @param upper_class_range: the highest value in the range of classes
 * @return: The mutual information score
 */
double mutual_info_score(const raft::handle_t& handle,
                         const int* y,
                         const int* y_hat,
                         const int n,
                         const int lower_class_range,
                         const int upper_class_range);

/**
 * Calculates the "homogeneity score" between two clusters
 *
 * A clustering result satisfies homogeneity if all of its clusters
 * contain only data points which are members of a single class.
 *
 * @param handle: raft::handle_t
 * @param y: truth labels
 * @param y_hat: predicted labels
 * @param n: Number of elements in y and y_hat
 * @param lower_class_range: the lowest value in the range of classes
 * @param upper_class_range: the highest value in the range of classes
 * @return: The homogeneity score
 */
double homogeneity_score(const raft::handle_t& handle,
                         const int* y,
                         const int* y_hat,
                         const int n,
                         const int lower_class_range,
                         const int upper_class_range);

/**
 * Calculates the "completeness score" between two clusters
 *
 * A clustering result satisfies completeness if all the data points
 * that are members of a given class are elements of the same cluster.
 *
 * @param handle: raft::handle_t
 * @param y: truth labels
 * @param y_hat: predicted labels
 * @param n: Number of elements in y and y_hat
 * @param lower_class_range: the lowest value in the range of classes
 * @param upper_class_range: the highest value in the range of classes
 * @return: The completeness score
 */
double completeness_score(const raft::handle_t& handle,
                          const int* y,
                          const int* y_hat,
                          const int n,
                          const int lower_class_range,
                          const int upper_class_range);

/**
 * Calculates the "v-measure" between two clusters
 *
 * v-measure is the harmonic mean between the homogeneity
 * and completeness scores of 2 cluster classifications
 *
 * @param handle: raft::handle_t
 * @param y: truth labels
 * @param y_hat: predicted labels
 * @param n: Number of elements in y and y_hat
 * @param lower_class_range: the lowest value in the range of classes
 * @param upper_class_range: the highest value in the range of classes
 * @param beta: Ratio of weight attributed to homogeneity vs completeness
 * @return: The v-measure
 */
double v_measure(const raft::handle_t& handle,
                 const int* y,
                 const int* y_hat,
                 const int n,
                 const int lower_class_range,
                 const int upper_class_range,
                 double beta);

/**
 * Calculates the "accuracy" between two input numpy arrays/ cudf series
 *
 * The accuracy metric is used to calculate the accuracy of the predict labels
 * predict labels
 *
 * @param handle: raft::handle_t
 * @param predictions: predicted labels
 * @param ref_predictions: truth labels
 * @param n: Number of elements in y and y_hat
 * @return: The accuracy
 */
float accuracy_score_py(const raft::handle_t& handle,
                        const int* predictions,
                        const int* ref_predictions,
                        int n);

/**
 * @brief Calculates the ij pairwise distances between two input arrays of
 *        double type
 *
 * @param handle raft::handle_t
 * @param x pointer to the input data samples array (mRows x kCols)
 * @param y pointer to the second input data samples array. Can use the same
 *          pointer as x (nRows x kCols)
 * @param dist output pointer where the results will be stored (mRows x nCols)
 * @param m number of rows in x
 * @param n number of rows in y
 * @param k number of cols in x and y (must be the same)
 * @param metric the distance metric to use for the calculation
 * @param isRowMajor specifies whether the x and y data pointers are row (C
 *                   type array) or col (F type array) major
 * @param metric_arg the value of `p` for Minkowski (l-p) distances.
 */
void pairwise_distance(const raft::handle_t& handle,
                       const double* x,
                       const double* y,
                       double* dist,
                       int m,
                       int n,
                       int k,
                       ML::distance::DistanceType metric,
                       bool isRowMajor   = true,
                       double metric_arg = 2.0);

/**
 * @brief Calculates the ij pairwise distances between two input arrays of float type
 *
 * @param handle raft::handle_t
 * @param x pointer to the input data samples array (mRows x kCols)
 * @param y pointer to the second input data samples array. Can use the same
 *          pointer as x (nRows x kCols)
 * @param dist output pointer where the results will be stored (mRows x nCols)
 * @param m number of rows in x
 * @param n number of rows in y
 * @param k number of cols in x and y (must be the same)
 * @param metric the distance metric to use for the calculation
 * @param isRowMajor specifies whether the x and y data pointers are row (C
 *                   type array) or col (F type array) major
 * @param metric_arg the value of `p` for Minkowski (l-p) distances.
 */
void pairwise_distance(const raft::handle_t& handle,
                       const float* x,
                       const float* y,
                       float* dist,
                       int m,
                       int n,
                       int k,
                       ML::distance::DistanceType metric,
                       bool isRowMajor  = true,
                       float metric_arg = 2.0f);

void pairwiseDistance_sparse(const raft::handle_t& handle,
                             double* x,
                             double* y,
                             double* dist,
                             int x_nrows,
                             int y_nrows,
                             int n_cols,
                             int x_nnz,
                             int y_nnz,
                             int* x_indptr,
                             int* y_indptr,
                             int* x_indices,
                             int* y_indices,
                             ML::distance::DistanceType metric,
                             float metric_arg);
void pairwiseDistance_sparse(const raft::handle_t& handle,
                             float* x,
                             float* y,
                             float* dist,
                             int x_nrows,
                             int y_nrows,
                             int n_cols,
                             int x_nnz,
                             int y_nnz,
                             int* x_indptr,
                             int* y_indptr,
                             int* x_indices,
                             int* y_indices,
                             ML::distance::DistanceType metric,
                             float metric_arg);

/**
 * @brief Compute the trustworthiness score
 *
 * @param h Raft handle
 * @param X Data in original dimension
 * @param X_embedded Data in target dimension (embedding)
 * @param n Number of samples
 * @param m Number of features in high/original dimension
 * @param d Number of features in low/embedded dimension
 * @param n_neighbors Number of neighbors considered by trustworthiness score
 * @param batchSize Batch size
 * @tparam distance_type: Distance type to consider
 * @return Trustworthiness score
 */
template <typename math_t, ML::distance::DistanceType distance_type>
double trustworthiness_score(const raft::handle_t& h,
                             const math_t* X,
                             math_t* X_embedded,
                             int n,
                             int m,
                             int d,
                             int n_neighbors,
                             int batchSize = 512);

}  // namespace Metrics
}  // namespace ML
