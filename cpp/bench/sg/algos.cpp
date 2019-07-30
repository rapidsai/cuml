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

#include "algos.h"
#include <cstring>
#include <cuML.hpp>
#include <dbscan/dbscan.hpp>
#include <map>
#include "argparse.hpp"
#include "dataset.h"

namespace ML {
namespace Bench {

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
  int minPts = get_argval(argv, argv + argc, "-min-pts", 3);
  float eps = get_argval(argv, argv + argc, "-eps", 1.f);
  size_t maxBytesPerBatch =
    get_argval(argv, argv + argc, "-max-bytes-per-batch", 0);
  printf(
    "With params:\n"
    "  min-pts              = %d\n"
    "  eps                  = %f\n"
    "  max-bytes-per-launch = %lu\n",
    minPts, eps, maxBytesPerBatch);
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  int* labels = (int*)allocator->allocate(ret.nrows * sizeof(int), stream);
  {
    struct timeval start;
    TIC(start);
    dbscanFit(handle, ret.X, ret.nrows, ret.ncols, eps, minPts, labels,
              maxBytesPerBatch);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOC(start, "dbscanFit");
  }
  ///@todo: add some clustering metrics for verification
  allocator->deallocate(labels, ret.nrows * sizeof(int), stream);
  return true;
}

typedef bool (*algoRunner)(const Dataset&, const cumlHandle&, int, char**);
class Runner : public std::map<std::string, algoRunner> {
 public:
  Runner() : std::map<std::string, algoRunner>() { (*this)["dbscan"] = dbscan; }
};

/// Do NOT touch anything below this line! ///
/// Only add new loaders above this line ///

Runner& runner() {
  static Runner map;
  return map;
}

std::string allAlgoNames() {
  const auto& run = runner();
  std::string ret;
  for (const auto& itr : run) {
    ret += itr.first + "|";
  }
  ret.pop_back();
  return ret;
}

int findAlgoStart(int argc, char** argv) {
  const auto& run = runner();
  for (int i = 0; i < argc; ++i) {
    for (const auto& itr : run) {
      if (!std::strcmp(itr.first.c_str(), argv[i])) return i;
    }
  }
  return argc;
}

bool runAlgo(const Dataset& ret, const cumlHandle& handle, int argc,
             char** argv) {
  std::string type = argc > 0 ? argv[0] : "dbscan";
  auto& run = runner();
  const auto& itr = run.find(type);
  ASSERT(itr != run.end(), "runAlgo: invalid algo name '%s'", type.c_str());
  struct timeval start;
  TIC(start);
  bool status = itr->second(ret, handle, argc, argv);
  if (status) {
    TOC(start, "total algo time");
  }
  return status;
}

}  // end namespace Bench
}  // end namespace ML
