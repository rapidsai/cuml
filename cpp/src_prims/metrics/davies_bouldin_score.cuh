/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cub/cub.cuh>
#include <cuml/metrics/metrics.hpp>
// #include <iostream>
#include <raft/cuda_utils.cuh>
#include <raft/distance/distance.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/matrix/matrix.hpp>
#include <raft/stats/mean.hpp>
#include <rmm/device_scalar.hpp>

namespace MLCommon {
namespace Metrics {


/**
 * @brief function to calculate the bincounts of number of samples in every label
 * @tparam DataT: type of the data samples
 * @tparam LabelT: type of the labels
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param binCountArray: pointer to the 1D array that contains the count of samples per cluster (1 x
 * nLabels)
 * @param nRows: number of data samples
 * @param nUniqueLabels: number of Labels
 * @param workspace: device buffer containing workspace memory
 * @param stream: the cuda stream where to launch this kernel
 */
template <typename DataT, typename LabelT>
void countLabels(LabelT* labels,
                 DataT* binCountArray,
                 int nRows,
                 int nUniqueLabels,
                 rmm::device_uvector<char>& workspace,
                 cudaStream_t stream)
{
  int num_levels            = nUniqueLabels + 1;
  LabelT lower_level        = 0;
  LabelT upper_level        = nUniqueLabels;
  size_t temp_storage_bytes = 0;

  rmm::device_uvector<int> countArray(nUniqueLabels, stream); 
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    binCountArray,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    nRows,
                                                    stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    binCountArray,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    nRows,
                                                    stream));
}

/**
 * @brief main function that returns the Davies Bouldin score for a given set of data and its
 * clusterings
 * @tparam DataT: type of the data samples
 * @tparam LabelT: type of the labels
 * @param X_in: pointer to the input Data samples array (nRows x nCols)
 * @param nRows: number of data samples
 * @param nCols: number of features
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param nLabels: number of clusters
 * @param stream: the cuda stream where to launch this kernel
 * @param metric: the numerical value that maps to the type of distance metric to be used in the
 * calculations
 */
template <typename DataT, typename LabelT>
DataT davies_bouldin_score(
  const raft::handle_t& handle,
  DataT* X_in,
  int nRows,
  int nCols,
  LabelT* labels,
  int nLabels,
  cudaStream_t stream,
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2Unexpanded)
{
  rmm::device_uvector<char> workspace(1, stream);

  // the bincountarray to get the sample counts for each class
  rmm::device_uvector<LabelT> binCountArray(nLabels, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(binCountArray.data(), 0, nLabels * sizeof(LabelT), stream));
  countLabels(labels, binCountArray.data(), nRows, nLabels, workspace, stream);

  rmm::device_uvector<LabelT> d_labels(nRows, stream);
  raft::update_device(d_labels.data(), labels, nRows, stream);

  // cluster centroids 
  rmm::device_uvector<DataT> d_centroids(nLabels * nCols, stream);
  DataT h_sArray[nLabels];

  for (int l = 0; l< nLabels; l++){
    LabelT sampleIndex = l;
    LabelT class_size = binCountArray.element(sampleIndex, stream);

    // Elements of a cluster and corresponding labels 
    rmm::device_uvector<DataT> X_cluster(class_size * nCols, stream);
    rmm::device_uvector<LabelT> class_labels(class_size, stream);

    for (int i = 0, j = 0; i < nRows && j < class_size; ++i){
      if (d_labels.element(i,stream) == sampleIndex){
        class_labels.set_element(j, i, stream);
        j++;
      }
    }

    // Copy data points for the identified indices to X_cluster
    raft::matrix::copyRows(X_in, class_size, nCols, X_cluster.data(), class_labels.data(), class_size, stream, true);
    rmm::device_uvector<DataT> centroid(1 * nCols, stream);
    raft::stats::mean(centroid.data(), X_cluster.data(), nCols, class_size, false, true, stream);

    // can we just use an update for a specified row...
    for (int c =0; c<nCols; c++){
      d_centroids.set_element(l*nCols+c, centroid.element(c, stream), stream);
    }

    // Within cluster distances
    rmm::device_uvector<DataT> clusterdistanceMatrix(class_size * 1, stream);
    ML::Metrics::pairwise_distance(
        handle, X_cluster.data(), centroid.data(), clusterdistanceMatrix.data(), class_size, 1, nCols, raft::distance::DistanceType::L2SqrtUnexpanded);

    // Mean within cluster distance
    rmm::device_uvector<DataT> Si(1 , stream);
    raft::stats::mean(Si.data(), clusterdistanceMatrix.data(), 1, class_size, false, false, stream);
    h_sArray[l] = Si.element(0,stream);

  }

  // Between cluster distances
  rmm::device_uvector<DataT> betweenclusterdistanceMatrix(nLabels * nLabels, stream);
  ML::Metrics::pairwise_distance(
      handle, d_centroids.data(), d_centroids.data(), betweenclusterdistanceMatrix.data(), nLabels, nLabels, nCols, raft::distance::DistanceType::L2SqrtUnexpanded);

  // Davies Bouldin Score computation
  DataT dbmatrix[nLabels][nLabels];
  DataT rowmax[nLabels] = {0};
  for(int i=0; i<nLabels; ++i){
      for(int j=0; j<nLabels; ++j){
        // compute dbmatrix
        if(i==j)
          dbmatrix[i][j] = 0;
        else
          dbmatrix[i][j] = (h_sArray[i] + h_sArray[j])/betweenclusterdistanceMatrix.element(i * nLabels + j, stream);
        
        // track the row max
        if(rowmax[i] < dbmatrix[i][j])
          rowmax[i] = dbmatrix[i][j];
      }
    }
  
  // mean of dbmatrix is the Davies Bouldin Score
  DataT dbscore =0;
  for (int e =0; e<nLabels; e++)
    dbscore += rowmax[e]/nLabels;

  return dbscore;
}

};  // namespace Metrics
};  // namespace MLCommon
