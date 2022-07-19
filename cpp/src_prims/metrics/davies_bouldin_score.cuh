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

#include <algorithm>
#include <cub/cub.cuh>
#include <cuml/common/logger.hpp>
#include <cuml/metrics/metrics.hpp>
#include <iostream>
#include <math.h>
#include <numeric>
#include <raft/cuda_utils.cuh>
#include <raft/distance/distance.hpp>
#include <raft/distance/distance_type.hpp>
#include <raft/linalg/add.hpp>
#include <raft/linalg/eltwise.hpp>
#include <raft/linalg/map_then_reduce.hpp>
#include <raft/linalg/matrix_vector_op.hpp>
#include <raft/linalg/reduce.hpp>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/matrix/math.hpp>
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

  rmm::device_uvector<int> countArray(nUniqueLabels, stream); // where are we using this?
  // Are we expecting an error while using HistogramEven??
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

  // auto handle = raft::handle_t{};
  // auto stream = handle.get_stream();
  rmm::device_uvector<char> workspace(1, stream);
  rmm::device_uvector<LabelT> d_labels(nRows, stream);
  raft::update_device(d_labels.data(), labels, nRows, stream);
  rmm::device_uvector<DataT> d_Xin(nRows * nCols, stream);
  raft::update_device(d_Xin.data(), X_in, nRows*nCols, stream);

  // the bincountarray to get the sample counts for each class
  rmm::device_uvector<LabelT> binCountArray(nLabels, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(binCountArray.data(), 0, nLabels * sizeof(LabelT), stream));
  countLabels(d_labels.data(), binCountArray.data(), nRows, nLabels, workspace, stream);

  // To check if the countLabels works as expected 
  for (int i = 0; i < nLabels; i++)
    std:: cout << "The number of elements in the class " << i << " are: " << binCountArray.element(i, stream) << std::endl;

  DataT h_sArray[nLabels];
  DataT h_centroids[nLabels][nCols];

  for (int l = 0; l< nLabels; l++){
    // Can exploit kernels
    LabelT sampleIndex = l;
    LabelT class_size = binCountArray.element(sampleIndex, stream);
    LabelT h_Xid[class_size];
    rmm::device_uvector<DataT> X_cluster(class_size * nCols, stream);
    // handle.sync_stream(stream);

    for (int i = 0, j = 0; i < nRows && j < class_size; ++i){
      if (d_labels.element(i, stream) == sampleIndex){
        // std::cout << "The indices of class 1 are: " << i << std::endl;
        h_Xid[j] = i;
        std::cout << "The indices of class" << sampleIndex << " are: " << h_Xid[j] << std::endl;
        j++;
      }
    }
  
    rmm::device_uvector<LabelT> class_labels(class_size, stream);
    raft::update_device(class_labels.data(), &h_Xid[0], (int)class_size, stream);

    // Copy data points for the identified indices to X_cluster
    raft::matrix::copyRows(d_Xin.data(), class_size, nCols, X_cluster.data(), class_labels.data(), class_size, stream, true);

    // print the elements of the class index
    for (int e = 0; e < class_size; e++){
      for (int c = 0; c < nCols; c++)
        std::cout << X_cluster.element(e*nCols+c,stream) << "  ";
      std::cout << std::endl;
    }
  // std::cout << "The element of X_cluster data: " << X_cluster.element(e*nCols,stream) << " , " << X_cluster.element(e*nCols+1,stream) << std::endl;
  
    rmm::device_uvector<DataT> centroid(1 * nCols, stream);
    raft::stats::mean(centroid.data(), X_cluster.data(), nCols, class_size, false, true, stream);

    std::cout << "The centroid of given data: " << std::endl;
    for (int c =0; c<nCols; c++){
      h_centroids[l][c] = centroid.element(c,stream);
      std::cout << h_centroids[l][c] << "  " ;
    }
    std::cout << std::endl;
  
    rmm::device_uvector<DataT> clusterdistanceMatrix(class_size * 1, stream);
    ML::Metrics::pairwise_distance(
        handle, X_cluster.data(), centroid.data(), clusterdistanceMatrix.data(), class_size, 1, nCols, raft::distance::DistanceType::L2SqrtUnexpanded);

    std::cout << "The pairwise distances are: " << std::endl;
    for (int e =0; e< class_size; e++)
      std::cout << clusterdistanceMatrix.element(e,stream) << "   ";
    std::cout << std::endl;

    rmm::device_uvector<DataT> Si(1 , stream);
    raft::stats::mean(Si.data(), clusterdistanceMatrix.data(), 1, class_size, false, false, stream);
    
    h_sArray[l] = Si.element(0,stream);
    std::cout << "The S_i for cluster: " << h_sArray[l]  << std::endl;

  }

  rmm::device_uvector<DataT> d_centroids(nLabels * nCols, stream);
  raft::update_device(d_centroids.data(), &h_centroids[0][0], (int)nLabels*nCols, stream);
  rmm::device_uvector<DataT> betweenclusterdistanceMatrix(nLabels * nLabels, stream);
  ML::Metrics::pairwise_distance(
      handle, d_centroids.data(), d_centroids.data(), betweenclusterdistanceMatrix.data(), nLabels, nLabels, nCols, raft::distance::DistanceType::L2SqrtUnexpanded);

  // check between cluster distances
  std::cout << "Between cluster distances are:" << std::endl;
  for (int e =0; e<nLabels; e++){
    for (int c =0; c<nLabels; c++)
      std::cout << betweenclusterdistanceMatrix.element(e*nLabels+c,stream) << "  ";
  std::cout << std::endl;
  }

  std::cout << "Combined sArray - numerator:" << std::endl;
  float combined_sArray[nLabels][nLabels];
  for(int i=0; i<nLabels; ++i){
      for(int j=0; j<nLabels; ++j){
        combined_sArray[i][j] = h_sArray[i] + h_sArray[j];
        std::cout << combined_sArray[i][j] << "  " ;
      }
      std::cout  << std::endl;
    }

  std::cout << "updated between cluster distance matrix - denominator:" << std::endl;
  float denominator_array[nLabels][nLabels];
  for(int i=0; i<nLabels; ++i){
      for(int j=0; j<nLabels; ++j){
        if(i==j){
          denominator_array[i][i] = 100000;
        }
        else 
        denominator_array[i][j] = betweenclusterdistanceMatrix.element(i * nLabels + j, stream);
      std::cout << denominator_array[i][j] << "    " ;
      }
      std::cout << std::endl;  
    }

  float dbmatrix[nLabels][nLabels];
  float rowmax[nLabels];
  float rmax =0;
  std::cout << "The DB matrix" << std:: endl;
  for(int i=0; i<nLabels; ++i){
      for(int j=0; j<nLabels; ++j){
        dbmatrix[i][j] = combined_sArray[i][j]/denominator_array[i][j];
        if(rmax < dbmatrix[i][j])
          rmax = dbmatrix[i][j];
      std::cout << dbmatrix[i][j] << "   " ;
      }
      rowmax[i] = rmax;
      rmax =0;
      std::cout << std:: endl;
    }
    // mean of dbmatrix is the Davies Bouldin Score
    for (int e =0; e<nLabels; e++)
      rmax += rowmax[e];
    std::cout << "The Davies Bouldin Score for the given data is: " << rmax/nLabels << std::endl;

  return rmax/nLabels;

}


};  // namespace Metrics
};  // namespace MLCommon

// step1: Return 0 as output and see why I get the following error:
// error: no instance of function template "MLCommon::Metrics::davies_bouldin_score" matches the argument list
//             argument types are: (raft::handle_t, float *, int, int, float *, int, float *, rmm::cuda_stream_view, raft::distance::DistanceType)

// Data - differences in data on host vs device. I have seen a lot of bugs in the data handling.
// auto stream = handle.get_stream(); should this be a parameter, how will it impact - if we dont use stream - there is no order/synchronize in the output

// creating the X_cluster array, has n_items * nCols
//   rmm::device_uvector<DataT> X_cluster(n_items * nCols, stream);
//   RAFT_CUDA_TRY(cudaMemsetAsync(X_cluster.data(), 0, n_items * nCols * sizeof(DataT), stream));
//   double* X_cluster = (double*)malloc(nRows * nCols * sizeof(double*));

// algorithm
// step-1: compute number of elements in the cluster [use binCountArray] use the toy example to validate...
  // Toy Example
  // X_in = {{1., 1.}, {1., 2.}, {1., 3.}, {1., 4.}, {1., 5.}};
  // labels = {0., 0., 0., 1., 1.};
  // nRows = 5;
  // nCols = 2;
  // nLabels = 2;
  // DataT - double;
  // LabelT - double;
// step-2: copy the data samples of sampleIndex cluster into X_cluster. [Get indices of cluster and copy corresponding rows of X_in to X_cluster]
// step-3: compute the centroid of the cluster [mean([X_cluster])]
// step-4: compute pairwise distances of cluster and the centroid, [X_cluster, Centroid] 
// step-5: compute mean [s0] to get [s0, s1, s2, ...]
// step-6: compute the numerator matrix 
  // [s0+s0, s0+s1, s0+s2,
  //  s1+s0, s1+s1, s1+s2,
  //  s2+s0, s2+s1, s2+s2]
// step-7: compute the denominator matrix
  // [d00, d01, d02
  //  d10, d11, d12
  //  d20, d21, d22]
// step-8: Elementwise division op 
// step-9: Max on columns
// step-10: Mean on the Maxelements