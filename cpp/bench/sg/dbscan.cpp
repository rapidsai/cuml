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

#include <cuML.hpp>
#include <dbscan/dbscan.hpp>
#include "argparse.hpp"
#include "dataset.h"

namespace ML {
namespace Bench {

// void printUsage() {
//   std::cout
//     << "To run default example use:" << std::endl
//     << "    dbscan_example [-dev_id <GPU id>]" << std::endl
//     << "For other cases:" << std::endl
//     << "    dbscan_example [-dev_id <GPU id>] -input <samples-file> "
//     << "-num_samples <number of samples> -num_features <number of features> "
//     << "[-min_pts <minimum number of samples in a cluster>] "
//     << "[-eps <maximum distance between any two samples of a cluster>] "
//     << "[-max_bytes_per_batch <maximum memory to use (in bytes) for batch size "
//        "calculation>] "
//     << std::endl;
//   return;
// }

// int main(int argc, char* argv[]) {
//   int devId = get_argval<int>(argv, argv + argc, "-dev_id", 0);
//   size_t nRows = get_argval<size_t>(argv, argv + argc, "-num_samples", 0);
//   size_t nCols = get_argval<size_t>(argv, argv + argc, "-num_features", 0);
//   std::string input =
//     get_argval<std::string>(argv, argv + argc, "-input", std::string(""));
//   int minPts = get_argval<int>(argv, argv + argc, "-min_pts", 3);
//   float eps = get_argval<float>(argv, argv + argc, "-eps", 1.0f);
//   size_t max_bytes_per_batch =
//     get_argval<size_t>(argv, argv + argc, "-max_bytes_per_batch", (size_t)2e7);

//   {
//     cudaError_t cudaStatus = cudaSuccess;
//     cudaStatus = cudaSetDevice(devId);
//     if (cudaSuccess != cudaStatus) {
//       std::cerr << "ERROR: Could not select CUDA device with the id: " << devId
//                 << "(" << cudaGetErrorString(cudaStatus) << ")" << std::endl;
//       return 1;
//     }
//     cudaStatus = cudaFree(0);
//     if (cudaSuccess != cudaStatus) {
//       std::cerr << "ERROR: Could not initialize CUDA on device: " << devId
//                 << "(" << cudaGetErrorString(cudaStatus) << ")" << std::endl;
//       return 1;
//     }
//   }

//   ML::cumlHandle cumlHandle;

// #ifdef HAVE_RMM
//   rmmOptions_t rmmOptions;
//   rmmOptions.allocation_mode = PoolAllocation;
//   rmmOptions.initial_pool_size = 0;
//   rmmOptions.enable_logging = false;
//   rmmError_t rmmStatus = rmmInitialize(&rmmOptions);
//   if (RMM_SUCCESS != rmmStatus) {
//     std::cerr << "WARN: Could not initialize RMM: "
//               << rmmGetErrorString(rmmStatus) << std::endl;
//   }
// #endif  //HAVE_RMM
// #ifdef HAVE_RMM
//   std::shared_ptr<ML::rmmAllocatorAdapter> allocator(
//     new ML::rmmAllocatorAdapter());
// #else   //!HAVE_RMM
//   std::shared_ptr<cachingDeviceAllocator> allocator(
//     new cachingDeviceAllocator());
// #endif  //HAVE_RMM
//   cumlHandle.setDeviceAllocator(allocator);

//   std::vector<float> h_inputData;

//   if (input == "") {
//     // Samples file not specified, run with defaults
//     std::cout << "Samples file not specified. (-input option)" << std::endl;
//     std::cout << "Running with default dataset:" << std::endl;
//     loadDefaultDataset(h_inputData, nRows, nCols, minPts, eps,
//                        max_bytes_per_batch);
//   } else if (nRows == 0 || nCols == 0) {
//     // Samples file specified but nRows and nCols is not specified
//     // Print usage and quit
//     std::cerr << "Samples file: " << input << std::endl;
//     std::cerr << "Incorrect value for (num_samples x num_features): (" << nRows
//               << " x " << nCols << ")" << std::endl;
//     printUsage();
//     return 1;
//   } else {
//     // All options are correctly specified
//     // Try to read input file now
//     std::ifstream input_stream(input, std::ios::in);
//     if (!input_stream.is_open()) {
//       std::cerr << "ERROR: Could not open input file " << input << std::endl;
//       return 1;
//     }
//     std::cout << "Trying to read samples from " << input << std::endl;
//     h_inputData.reserve(nRows * nCols);
//     float val = 0.0;
//     while (input_stream >> val) {
//       h_inputData.push_back(val);
//     }
//     if (h_inputData.size() != nRows * nCols) {
//       std::cerr << "ERROR: Read " << h_inputData.size() << " from " << input
//                 << ", while expecting to read: " << nRows * nCols
//                 << " (num_samples*num_features)" << std::endl;
//       return 1;
//     }
//   }

//   cudaStream_t stream;
//   CUDA_RT_CALL(cudaStreamCreate(&stream));
//   cumlHandle.setStream(stream);

//   std::vector<int> h_labels(nRows);
//   int* d_labels = nullptr;
//   float* d_inputData = nullptr;

//   CUDA_RT_CALL(cudaMalloc(&d_labels, nRows * sizeof(int)));
//   CUDA_RT_CALL(cudaMalloc(&d_inputData, nRows * nCols * sizeof(float)));
//   CUDA_RT_CALL(cudaMemcpyAsync(d_inputData, h_inputData.data(),
//                                nRows * nCols * sizeof(float),
//                                cudaMemcpyHostToDevice, stream));

//   std::cout << "Running DBSCAN with following parameters:" << std::endl
//             << "Number of samples - " << nRows << std::endl
//             << "Number of features - " << nCols << std::endl
//             << "min_pts - " << minPts << std::endl
//             << "eps - " << eps << std::endl
//             << "max_bytes_per_batch - " << max_bytes_per_batch << std::endl;

//   ML::dbscanFit(cumlHandle, d_inputData, nRows, nCols, eps, minPts, d_labels,
//                 max_bytes_per_batch);
//   CUDA_RT_CALL(cudaMemcpyAsync(h_labels.data(), d_labels, nRows * sizeof(int),
//                                cudaMemcpyDeviceToHost, stream));
//   CUDA_RT_CALL(cudaStreamSynchronize(stream));

//   std::map<int, size_t> histogram;
//   for (int row = 0; row < nRows; row++) {
//     if (histogram.find(h_labels[row]) == histogram.end()) {
//       histogram[h_labels[row]] = 1;
//     } else {
//       histogram[h_labels[row]]++;
//     }
//   }

//   size_t nClusters = 0;
//   size_t noise = 0;
//   std::cout << "Histogram of samples" << std::endl;
//   std::cout << "Cluster id, Number samples" << std::endl;
//   for (auto it = histogram.begin(); it != histogram.end(); it++) {
//     if (it->first != -1) {
//       std::cout << std::setw(10) << it->first << ", " << it->second
//                 << std::endl;
//       nClusters++;
//     } else {
//       noise += it->second;
//     }
//   }

//   std::cout << "Total number of clusters: " << nClusters << std::endl;
//   std::cout << "Noise samples: " << noise << std::endl;

//   CUDA_RT_CALL(cudaFree(d_labels));
//   CUDA_RT_CALL(cudaFree(d_inputData));
//   CUDA_RT_CALL(cudaStreamDestroy(stream));
//   CUDA_RT_CALL(cudaDeviceSynchronize());
//   return 0;
// }

bool dbscan(const Dataset& ret, const cumlHandle& handle, int argc,
            char** argv) {
  bool help = get_argval(argv, argv + argc, "-h");
  if (help) {
    printf(
      "USAGE:\n"
      "bench dbscan [options]\n"
      "  Run dbscan algo on the input dataset.\n"
      "OPTIONS:\n"
      "  -min-pts <pts>   Min number of points in a cluster. [3]\n"
      "  -eps <eps>       Max distance between any 2 points of a cluster.\n"
      "                   [1.f]\n"
      "  -max-bytes-per-batch <mem>  Max memory to use for the batch size\n"
      "                              calculation. [0] 0 means use up all the\n"
      "                              available free memory.\n");
    return false;
  }
  printf("Running dbscan...\n");
  // int main(int argc, char* argv[]) {
  //   int devId = get_argval<int>(argv, argv + argc, "-dev_id", 0);
  //   size_t nRows = get_argval<size_t>(argv, argv + argc, "-num_samples", 0);
  //   size_t nCols = get_argval<size_t>(argv, argv + argc, "-num_features", 0);
  //   std::string input =
  //     get_argval<std::string>(argv, argv + argc, "-input", std::string(""));
  //   int minPts = get_argval<int>(argv, argv + argc, "-min_pts", 3);
  //   float eps = get_argval<float>(argv, argv + argc, "-eps", 1.0f);
  //   size_t max_bytes_per_batch =
  //     get_argval<size_t>(argv, argv + argc, "-max_bytes_per_batch", (size_t)2e7);
  return true;
}

}  // end namespace Bench
}  // end namespace ML
