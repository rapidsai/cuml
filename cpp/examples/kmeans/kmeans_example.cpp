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
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#ifdef HAVE_CUB
#include <cub/util_allocator.cuh>
#endif  //HAVE_CUB

#ifdef HAVE_RMM
#include <rmm/rmm.h>
#include <common/rmmAllocatorAdapter.hpp>
#endif  //HAVE_RMM

#include <cuML.hpp>
#include <kmeans/kmeans.hpp>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
              cudaStatus);                                                    \
  }
#endif  //CUDA_RT_CALL

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg,
             const T default_val) {
  T argval = default_val;
  char** itr = std::find(begin, end, arg);
  if (itr != end && ++itr != end) {
    std::istringstream inbuf(*itr);
    inbuf >> argval;
  }
  return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
  char** itr = std::find(begin, end, arg);
  if (itr != end) {
    return true;
  }
  return false;
}

class cachingDeviceAllocator : public ML::deviceAllocator {
 public:
  cachingDeviceAllocator()
#ifdef HAVE_CUB
    ,
    _allocator(8, 3, cub::CachingDeviceAllocator::INVALID_BIN,
               cub::CachingDeviceAllocator::INVALID_SIZE)
#endif  //HAVE_CUB
  {
  }

  virtual void* allocate(std::size_t n, cudaStream_t stream) {
    void* ptr = 0;
#ifdef HAVE_CUB
    _allocator.DeviceAllocate(&ptr, n, stream);
#else   //!HAVE_CUB
    CUDA_RT_CALL(cudaMalloc(&ptr, n));
#endif  //HAVE_CUB
    return ptr;
  }

  virtual void deallocate(void* p, std::size_t, cudaStream_t) {
#ifdef HAVE_CUB
    _allocator.DeviceFree(p);
#else   //!HAVE_CUB
    CUDA_RT_CALL(cudaFree(p));
#endif  //HAVE_CUB
  }

#ifdef HAVE_CUB
 private:
  cub::CachingDeviceAllocator _allocator;
#endif  //HAVE_CUB
};

int main(int argc, char* argv[]) {
  const int dev_id = get_argval<int>(argv, argv + argc, "-dev_id", 0);
  const size_t num_rows = get_argval<size_t>(argv, argv + argc, "-num_rows", 0);
  const size_t num_cols = get_argval<size_t>(argv, argv + argc, "-num_cols", 0);
  const std::string input =
    get_argval<std::string>(argv, argv + argc, "-input", std::string(""));
  //Default values for k and max_iterations are taken from
  //https://github.com/h2oai/h2o4gpu/blob/master/examples/py/demos/H2O4GPU_KMeans_Homesite.ipynb
  int k = get_argval<int>(argv, argv + argc, "-k", 10);
  int max_iterations =
    get_argval<int>(argv, argv + argc, "-max_iterations", 300);
  {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaSetDevice(dev_id);
    if (cudaSuccess != cudaStatus) {
      std::cerr << "ERROR: Could not select CUDA device with the id: " << dev_id
                << "(" << cudaGetErrorString(cudaStatus) << ")" << std::endl;
      return 1;
    }
    cudaStatus = cudaFree(0);
    if (cudaSuccess != cudaStatus) {
      std::cerr << "ERROR: Could not initialize CUDA on device: " << dev_id
                << "(" << cudaGetErrorString(cudaStatus) << ")" << std::endl;
      return 1;
    }
#ifdef HAVE_RMM
    rmmOptions_t rmmOptions;
    rmmOptions.allocation_mode = PoolAllocation;
    rmmOptions.initial_pool_size = 0;
    rmmOptions.enable_logging = false;
    rmmError_t rmmStatus = rmmInitialize(&rmmOptions);
    if (RMM_SUCCESS != rmmStatus) {
      std::cerr << "WARN: Could not initialize RMM: "
                << rmmGetErrorString(rmmStatus) << std::endl;
    }
#endif  //HAVE_RMM
  }

  std::vector<double> h_srcdata;
  if ("" != input) {
    std::ifstream input_stream(input, std::ios::in);
    if (!input_stream.is_open()) {
      std::cerr << "ERROR: Could not open input file " << input << std::endl;
      return 1;
    }
    std::cout << "Reading input with " << num_rows << " rows and " << num_cols
              << " columns from " << input << "." << std::endl;
    h_srcdata.reserve(num_rows * num_cols);
    double val = 0.0;
    while (input_stream >> val) {
      h_srcdata.push_back(val);
    }
  }
  bool results_correct = true;
  if (0 == h_srcdata.size() || (num_rows * num_cols) == h_srcdata.size()) {
    double threshold = 1.0E-4;  //Scikit-Learn default

    //Input parameters copied from kmeans_test.cu
    if (0 == h_srcdata.size()) {
      k = 2;
      max_iterations = 300;
      threshold = 0.05;
    }
    int seed = 0;
    int metric = 1;
    ML::kmeans::InitMethod method = ML::kmeans::InitMethod::Random;

    //Inputs copied from kmeans_test.cu
    size_t n_samples = 4;
    size_t n_features = 2;
    if (0 == h_srcdata.size()) {
      h_srcdata = {1.0, 1.0, 3.0, 4.0, 1.0, 2.0, 2.0, 3.0};
    } else {
      n_samples = num_rows;
      n_features = num_cols;
    }
    std::cout << "Run KMeans with k=" << k
              << ", max_iterations=" << max_iterations << std::endl;

    ML::cumlHandle cumlHandle;
#ifdef HAVE_RMM
    std::shared_ptr<ML::rmmAllocatorAdapter> allocator(
      new ML::rmmAllocatorAdapter());
#else   //!HAVE_RMM
    std::shared_ptr<cachingDeviceAllocator> allocator(
      new cachingDeviceAllocator());
#endif  //HAVE_RMM
    cumlHandle.setDeviceAllocator(allocator);

    cudaStream_t stream;
    CUDA_RT_CALL(cudaStreamCreate(&stream));
    cumlHandle.setStream(stream);

    //srcdata size n_samples * n_features
    double* d_srcdata = nullptr;
    CUDA_RT_CALL(
      cudaMalloc(&d_srcdata, n_samples * n_features * sizeof(double)));
    CUDA_RT_CALL(cudaMemcpyAsync(d_srcdata, h_srcdata.data(),
                                 n_samples * n_features * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));

    //output pred_centroids size k * n_features
    double* d_pred_centroids = nullptr;
    CUDA_RT_CALL(
      cudaMalloc(&d_pred_centroids, k * n_features * sizeof(double)));
    //output pred_labels size n_samples
    int* d_pred_labels = nullptr;
    CUDA_RT_CALL(cudaMalloc(&d_pred_labels, n_samples * sizeof(int)));

    ML::kmeans::fit_predict(cumlHandle, k, metric, method, max_iterations,
                            threshold, seed, d_srcdata, n_samples, n_features,
                            d_pred_centroids, d_pred_labels);

    std::vector<int> h_pred_labels(n_samples);
    CUDA_RT_CALL(cudaMemcpyAsync(h_pred_labels.data(), d_pred_labels,
                                 n_samples * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
    std::vector<double> h_pred_centroids(k * n_features);
    CUDA_RT_CALL(cudaMemcpyAsync(h_pred_centroids.data(), d_pred_centroids,
                                 k * n_features * sizeof(double),
                                 cudaMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    if (8 == h_srcdata.size()) {
      int h_labels_ref_fit[n_samples] = {0, 1, 0, 1};
      for (int i = 0; i < n_samples; ++i) {
        if (h_labels_ref_fit[i] != h_pred_labels[i]) {
          std::cerr << "ERROR: h_labels_ref_fit[" << i
                    << "] = " << h_labels_ref_fit[i]
                    << " != " << h_pred_labels[i] << " = h_pred_labels[" << i
                    << "]!" << std::endl;
          results_correct = false;
        }
      }

      double h_centroids_ref[k * n_features] = {1.0, 1.5, 2.5, 3.5};
      for (int i = 0; i < k * n_features; ++i) {
        if (std::abs(h_centroids_ref[i] - h_pred_centroids[i]) /
              std::abs(h_centroids_ref[i]) >
            std::numeric_limits<double>::epsilon()) {
          std::cerr << "ERROR: h_centroids_ref[" << i
                    << "] = " << h_centroids_ref[i]
                    << " !~= " << h_pred_centroids[i] << " = h_pred_centroids["
                    << i << "]!" << std::endl;
          results_correct = false;
        }
      }
    } else {
      std::vector<std::pair<size_t, double>> cluster_stats(
        k, std::make_pair(static_cast<size_t>(0), 0.0));
      double global_inertia = 0.0;
      size_t max_points = 0;
      for (size_t i = 0; i < n_samples; ++i) {
        int label = h_pred_labels[i];
        cluster_stats[label].first += 1;
        max_points = std::max(cluster_stats[label].first, max_points);

        double sd = 0.0;
        for (int j = 0; j < n_features; ++j) {
          const double cluster_centroid_comp =
            h_pred_centroids[label * n_features + j];
          const double point_comp = h_srcdata[i * n_features + j];
          sd += (cluster_centroid_comp - point_comp) *
                (cluster_centroid_comp - point_comp);
        }
        cluster_stats[label].second += sd;
        global_inertia += sd;
      }
      int lable_widht = 0;
      int max_label = (k - 1);
      do {
        lable_widht += 1;
        max_label /= 10;
      } while (max_label > 0);
      int num_pts_width = 0;
      do {
        num_pts_width += 1;
        max_points /= 10;
      } while (max_points > 0);
      num_pts_width = std::max(num_pts_width, 7);

      for (int c = 0; c < lable_widht; ++c) std::cout << " ";
      std::cout << "  num_pts       inertia" << std::endl;
      for (int l = 0; l < k; ++l) {
        std::cout << std::setw(lable_widht) << l << "  "
                  << std::setw(num_pts_width) << cluster_stats[l].first << "  "
                  << std::scientific << std::setprecision(6)
                  << cluster_stats[l].second << std::endl;
      }
      std::cout << "Global inertia = " << global_inertia << std::endl;
    }

    CUDA_RT_CALL(cudaFree(d_pred_labels));
    d_pred_labels = nullptr;
    CUDA_RT_CALL(cudaFree(d_pred_centroids));
    d_pred_centroids = nullptr;
    CUDA_RT_CALL(cudaFree(d_srcdata));
    d_srcdata = nullptr;
    CUDA_RT_CALL(cudaStreamDestroy(stream));
  } else {
    std::cerr << "ERROR: Number of input values = " << h_srcdata.size()
              << " != " << num_rows * num_cols << " = " << num_rows << "*"
              << num_cols << " !" << std::endl;
    return 1;
  }
#ifdef HAVE_RMM
  if (rmmIsInitialized(NULL)) {
    rmmFinalize();
  }
#endif  //HAVE_RMM
  CUDA_RT_CALL(cudaDeviceReset());
  return results_correct ? 0 : 1;
}
