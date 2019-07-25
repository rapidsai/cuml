/*
* Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuML.hpp>

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
* @param handle: cumlHandle
* @param y: Array of ground-truth response variables
* @param y_hat: Array of predicted response variables
* @param n: Number of elements in y and y_hat
* @return: The R-squared value.
*/
float r2_score_py(const cumlHandle &handle, float *y, float *y_hat, int n);

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
* @param handle: cumlHandle
* @param y: Array of ground-truth response variables
* @param y_hat: Array of predicted response variables
* @param n: Number of elements in y and y_hat
* @return: The R-squared value.
*/
double r2_score_py(const cumlHandle &handle, double *y, double *y_hat, int n);

/**
* Calculates the "rand index"
*
* This metric is a measure of similarity between two data clusterings.
*
* @param handle: cumlHandle
* @param y: Array of response variables of the first clustering classifications
* @param y_hat: Array of response variables of the second clustering classifications
* @param n: Number of elements in y and y_hat
* @return: The rand index value
*/

double randIndex(const cumlHandle &handle, double *y, double *y_hat, int n);

/**
* Calculates the "Silhouette Score"
*
* The Silhouette Coefficient is calculated using the mean intra-cluster distance (a)
* and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient
* for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample 
* and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient
* is only defined if number of labels is 2 <= n_labels <= n_samples - 1.
*
* @param handle: cumlHandle
* @param y: Array of data samples with dimensions (nRows x nCols)
* @param nRows: number of data samples
* @param nCols: number of features
* @param labels: Array containing labels for every data sample (1 x nRows)
* @param nLabels: number of Labels
* @param metric: the numerical value that maps to the type of distance metric to be used in the calculations
* @param silScores: Array that is optionally taken in as input if required to be populated with the silhouette score for every sample (1 x nRows), else nullptr is passed
*/
double silhouetteScore(const cumlHandle &handle, double *y, int nRows,
                       int nCols, int *labels, int nLabels, double *silScores,
                       int metric);
/**
* Calculates the "adjusted rand index"
*
* This metric is the corrected-for-chance version of the rand index
*
* @param handle: cumlHandle
* @param y: Array of response variables of the first clustering classifications
* @param y_hat: Array of response variables of the second clustering classifications
* @param n: Number of elements in y and y_hat
* @param lower_class_range: the lowest value in the range of classes
* @param upper_class_range: the highest value in the range of classes
* @return: The adjusted rand index value
*/
double adjustedRandIndex(const cumlHandle &handle, const int *y,
                         const int *y_hat, const int n,
                         const int lower_class_range,
                         const int upper_class_range);

/**
* Calculates the "Kullback-Leibler Divergence"
*
* The KL divergence tells us how well the probability distribution Q
* approximates the probability distribution P
* It is often also used as a 'distance metric' between two probablity ditributions (not symmetric)
*
* @param handle: cumlHandle
* @param y: Array of probabilities corresponding to distribution P
* @param y_hat: Array of probabilities corresponding to distribution Q
* @param n: Number of elements in y and y_hat
* @return: The KL Divergence value
*/
double klDivergence(const cumlHandle &handle, const double *y,
                    const double *y_hat, int n);

/**
* Calculates the "Kullback-Leibler Divergence"
*
* The KL divergence tells us how well the probability distribution Q
* approximates the probability distribution P
* It is often also used as a 'distance metric' between two probablity ditributions (not symmetric)
*
* @param handle: cumlHandle
* @param y: Array of probabilities corresponding to distribution P
* @param y_hat: Array of probabilities corresponding to distribution Q
* @param n: Number of elements in y and y_hat
* @return: The KL Divergence value
*/
float klDivergence(const cumlHandle &handle, const float *y, const float *y_hat,
                   int n);

/**
* Calculates the "entropy" of a labelling
*
* This metric is a measure of the purity/polarity of the clustering 
*
* @param handle: cumlHandle
* @param y: Array of response variables of the clustering
* @param n: Number of elements in y
* @param lower_class_range: the lowest value in the range of classes
* @param upper_class_range: the highest value in the range of classes
* @return: The entropy value of the clustering
*/
double entropy(const cumlHandle &handle, const int *y, const int n,
               const int lower_class_range, const int upper_class_range);

/**
* Calculates the "Mutual Information score" between two clusters
*
* Mutual Information is a measure of the similarity between two labels of
* the same data.
*
* @param handle: cumlHandle
* @param y: Array of response variables of the first clustering classifications
* @param y_hat: Array of response variables of the second clustering classifications
* @param n: Number of elements in y and y_hat
* @param lower_class_range: the lowest value in the range of classes
* @param upper_class_range: the highest value in the range of classes
* @return: The mutual information score
*/
double mutualInfoScore(const cumlHandle &handle, const int *y, const int *y_hat,
                       const int n, const int lower_class_range,
                       const int upper_class_range);

/**
* Calculates the "homogeneity score" between two clusters
*
* A clustering result satisfies homogeneity if all of its clusters
* contain only data points which are members of a single class.
*
* @param handle: cumlHandle
* @param y: truth labels
* @param y_hat: predicted labels
* @param n: Number of elements in y and y_hat
* @param lower_class_range: the lowest value in the range of classes
* @param upper_class_range: the highest value in the range of classes
* @return: The homogeneity score
*/
double homogeneityScore(const cumlHandle &handle, const int *y,
                        const int *y_hat, const int n,
                        const int lower_class_range,
                        const int upper_class_range);

/**
* Calculates the "completeness score" between two clusters
*
* A clustering result satisfies completeness if all the data points
* that are members of a given class are elements of the same cluster.
*
* @param handle: cumlHandle
* @param y: truth labels
* @param y_hat: predicted labels
* @param n: Number of elements in y and y_hat
* @param lower_class_range: the lowest value in the range of classes
* @param upper_class_range: the highest value in the range of classes
* @return: The completeness score
*/
double completenessScore(const cumlHandle &handle, const int *y,
                         const int *y_hat, const int n,
                         const int lower_class_range,
                         const int upper_class_range);

/**
* Calculates the "v-measure" between two clusters
*
* v-measure is the harmonic mean between the homogeneity
* and completeness scores of 2 cluster classifications
*
* @param handle: cumlHandle
* @param y: truth labels
* @param y_hat: predicted labels
* @param n: Number of elements in y and y_hat
* @param lower_class_range: the lowest value in the range of classes
* @param upper_class_range: the highest value in the range of classes
* @return: The v-measure
*/
double vMeasure(const cumlHandle &handle, const int *y, const int *y_hat,
                const int n, const int lower_class_range,
                const int upper_class_range);

float accuracy_score_py(const cumlHandle &handle, const int *predictions,
                        const int *ref_predictions, int n);
}  // namespace Metrics
}  // namespace ML