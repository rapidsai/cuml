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
#include <kmeans/kmeans.hpp>
#include <map>
#include <randomforest/randomforest.hpp>
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
    "  max-bytes-per-batch  = %lu\n",
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

bool kmeans(const Dataset& ret, const cumlHandle& handle, int argc,
            char** argv) {
  bool help = get_argval(argv, argv + argc, "-h");
  if (help) {
    printf(
      "USAGE:\n"
      "bench kmeans [options]\n"
      "  Run kmeans clustering algo on the input dataset.\n"
      "OPTIONS:\n"
      "  -init <i>        Kmeans initialization method. [0]\n"
      "                     0 = kmeans++\n"
      "                     1 = Random\n"
      "  -max-iter <mi>   Max Llyod iterations. [300]\n"
      "  -tol <tol>       Tolerance criteria for convergence. [1e-4]\n"
      "  -v               Enable verbose output.\n"
      "  -seed <s>        Seed for RNG.\n"
      "  -metric <m>      Distance to use. [0]\n"
      "                     0 = L2\n"
      "  -oversample <o>  Oversampling factor in kmeans||. [2]\n"
      "  -batch-size <b>  batch size. [2^15]\n"
      "  -inertia-check   Check inertia.\n");
    return false;
  }
  printf("Running kmeans...\n");
  kmeans::KMeansParams params;
  params.n_clusters = ret.nclasses;
  params.init =
    (kmeans::KMeansParams::InitMethod)get_argval(argv, argv + argc, "-init", 0);
  params.max_iter = get_argval(argv, argv + argc, "-max-iter", 300);
  params.tol = get_argval(argv, argv + argc, "-tol", 1e-4);
  params.verbose = get_argval(argv, argv + argc, "-v");
  params.seed = get_argval(argv, argv + argc, "-seed", 0);
  params.metric = get_argval(argv, argv + argc, "-metric", 0);
  params.oversampling_factor = get_argval(argv, argv + argc, "-oversample", 2);
  params.batch_size = get_argval(argv, argv + argc, "-batch-size", 1 << 15);
  params.inertia_check = get_argval(argv, argv + argc, "-inertia-check");
  printf(
    "With params:\n"
    "  nclusters          = %d\n"
    "  initMethod         = %d\n"
    "  maxIter            = %d\n"
    "  tolerance          = %lf\n"
    "  verbose            = %d\n"
    "  seed               = %d\n"
    "  metric             = %d\n"
    "  oversample         = %d\n"
    "  batchSize          = %d\n"
    "  inertia-check      = %d\n",
    params.n_clusters, params.init, params.max_iter, params.tol, params.verbose,
    params.seed, params.metric, params.oversampling_factor, params.batch_size,
    params.inertia_check);
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  int* labels = (int*)allocator->allocate(ret.nrows * sizeof(int), stream);
  float* centroids = (float*)allocator->allocate(
    ret.nclasses * ret.ncols * sizeof(float), stream);
  {
    struct timeval start;
    TIC(start);
    int nIter;
    float inertia;
    kmeans::fit_predict(handle, params, ret.X, ret.nrows, ret.ncols, centroids,
                        labels, inertia, nIter);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("TotalIter=%d Inertia=%f\n", nIter, inertia);
    TOC(start, "kmeansFitPredict");
  }
  ///@todo: add some metrics for verification
  allocator->deallocate(labels, ret.nrows * sizeof(int), stream);
  allocator->deallocate(centroids, ret.nclasses * ret.ncols * sizeof(float),
                        stream);
  return true;
}

bool rfClassifier(const Dataset& ret, const cumlHandle& handle, int argc,
                  char** argv) {
  bool help = get_argval(argv, argv + argc, "-h");
  if (help) {
    printf(
      "USAGE:\n"
      "bench rfClassifier [options]\n"
      "  Run RF Classifier algo on the input dataset.\n"
      "OPTIONS:\n"
      "  -ntrees <nt>       Number of trees to build. [100]\n"
      "  -bootstrap         Whether to bootstrap the input data.\n"
      "  -row-subsample <frac> Row subsample ratio. [1.f]\n"
      "  -nstreams <ns>     Number of cuda streams to use. [4]\n"
      "  -max-depth <md>    Max tree depth. [8]\n"
      "  -max-leaves <ml>   Max leaves in each trees. [-1]\n"
      "  -max-features <mf> Max features to consider per split. [1.f]\n"
      "  -nbins <nb>        Number of bins used by split algo. [8]\n"
      "  -split-algo <algo> Split algo. 0 = HIST, 1 = GLOBAL_QUANTILE [1]\n"
      "  -min-rows-split <mrs>  Min number of rows needed for split. [2]\n"
      "  -bootstrap-features    Whether to bootstrap features of input data.\n"
      "  -quantile-per-tree     Compute quantile per tree. (default per RF)\n"
      "  -nclasses              Number of unique classes. [2]\n");
    return false;
  }
  printf("Running RF...\n");
  int nTrees = get_argval(argv, argv + argc, "-ntrees", 100);
  bool bootstrap = get_argval(argv, argv + argc, "-bootstrap");
  float rowSample = get_argval(argv, argv + argc, "-row-subsample", 1.f);
  int nStreams = get_argval(argv, argv + argc, "-nstreams", 4);
  int maxDepth = get_argval(argv, argv + argc, "-max-depth", 8);
  int maxLeaves = get_argval(argv, argv + argc, "-max-leaves", -1);
  float maxFeatures = get_argval(argv, argv + argc, "-max-features", 1.f);
  int nBins = get_argval(argv, argv + argc, "-nbins", 8);
  int algo = get_argval(argv, argv + argc, "-split-algo", 1);
  int minRowsSplit = get_argval(argv, argv + argc, "-min-rows-split", 2);
  bool bootstrapCols = get_argval(argv, argv + argc, "-bootstrap-features");
  bool quantilePerTree = get_argval(argv, argv + argc, "-quantile-per-tree");
  int nClasses = get_argval(argv, argv + argc, "-nclasses", 2);
  printf(
    "With params:\n"
    "  num-trees          = %d\n"
    "  bootstrap          = %d\n"
    "  row-subsample      = %f\n"
    "  num-streams        = %d\n"
    "  max-depth          = %d\n"
    "  max-leaves         = %d\n"
    "  max-features       = %f\n"
    "  num-bins           = %d\n"
    "  split-algo         = %d\n"
    "  min-rows-split     = %d\n"
    "  bootstrap-features = %d\n"
    "  quantile-per-tree  = %d\n",
    nTrees, bootstrap, rowSample, nStreams, maxDepth, maxLeaves, maxFeatures,
    nBins, algo, minRowsSplit, bootstrapCols, quantilePerTree);
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  int* labels = (int*)allocator->allocate(ret.nrows * sizeof(int), stream);
  {
    struct timeval start;
    TIC(start);
    RandomForestClassifierF model;
    model.trees = nullptr;
    auto* mPtr = &model;
    RF_params params = {nTrees,   bootstrap,    rowSample,     nStreams,
                        maxDepth, maxLeaves,    maxFeatures,   nBins,
                        algo,     minRowsSplit, bootstrapCols, quantilePerTree};
    fit(handle, mPtr, ret.X, ret.nrows, ret.ncols, labels, nClasses, params);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOC(start, "rfClassifierFit");
  }
  ///@todo: add some metrics for verification
  allocator->deallocate(labels, ret.nrows * sizeof(int), stream);
  return true;
}

typedef bool (*algoRunner)(const Dataset&, const cumlHandle&, int, char**);
class Runner : public std::map<std::string, algoRunner> {
 public:
  Runner() : std::map<std::string, algoRunner>() {
    (*this)["dbscan"] = dbscan;
    (*this)["kmeans"] = kmeans;
    (*this)["rfClassifier"] = rfClassifier;
  }
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
