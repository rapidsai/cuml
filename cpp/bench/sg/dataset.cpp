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

#include "dataset.h"
#include <cstdio>
#include <cstring>
#include <cuML.hpp>
#include <datasets/make_blobs.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "argparse.hpp"
#include "utils.h"

namespace ML {
namespace Bench {

void Dataset::allocate(const cumlHandle& handle) {
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  X = (float*)allocator->allocate(nrows * ncols * sizeof(float), stream);
  y = (int*)allocator->allocate(nrows * sizeof(int), stream);
}

void Dataset::deallocate(const cumlHandle& handle) {
  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  allocator->deallocate(X, nrows * ncols * sizeof(float), stream);
  allocator->deallocate(y, nrows * sizeof(int), stream);
}

void dumpDataset(const cumlHandle& handle, const Dataset& dataset,
                 const std::string& file) {
  printf("Dumping generated dataset to '%s'\n", file.c_str());
  FILE* fp = std::fopen(file.c_str(), "w");
  ASSERT(fp != nullptr, "Failed to open file '%s' for writing", file.c_str());
  auto stream = handle.getStream();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<float> X(dataset.nrows * dataset.ncols);
  std::vector<int> y(dataset.nrows);
  MLCommon::updateHost(X.data(), dataset.X, dataset.nrows * dataset.ncols,
                       stream);
  MLCommon::updateHost(y.data(), dataset.y, dataset.nrows, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  fprintf(fp, "%d %d %d\n", dataset.nrows, dataset.ncols, dataset.nclasses);
  for (int i = 0, k = 0; i < dataset.nrows; ++i) {
    for (int j = 0; j < dataset.ncols; ++j, ++k) fprintf(fp, "%f ", X[k]);
    fprintf(fp, "%d\n", y[i]);
  }
  fclose(fp);
}

bool blobs(Dataset& ret, const cumlHandle& handle, int argc, char** argv) {
  bool help = get_argval(argv, argv + argc, "-h");
  if (help) {
    printf(
      "USAGE:\n"
      "bench blobs [options]\n"
      "  Generate a random dataset similar to sklearn's make_blobs.\n"
      "OPTIONS:\n"
      "  -center-box-max <max>   max bounding box for the centers of the\n"
      "                          clusters [10.f].\n"
      "  -center-box-min <min>   min bounding box for the centers of the\n"
      "                          clusters [-10.f].\n"
      "  -cluster-std <std>      cluster std-deviation [1.f].\n"
      "  -dump <file>            dump the generated dataset.\n"
      "  -h                      print this help and exit.\n"
      "  -nclusters <nclusters>  number of clusters to generate [2].\n"
      "  -ncols <ncols>          number of cols in the dataset [81].\n"
      "  -nrows <nrows>          number of rows in the dataset [10001].\n"
      "  -seed <seed>            random seed for reproducibility [1234].\n"
      "  -shuffle                whether to shuffle the dataset.\n");
    return false;
  }
  printf("Generating blobs...\n");
  float centerBoxMax = get_argval(argv, argv + argc, "-center-box-max", 10.f);
  float centerBoxMin = get_argval(argv, argv + argc, "-center-box-min", -10.f);
  float clusterStd = get_argval(argv, argv + argc, "-cluster-std", 1.f);
  std::string dump = get_argval(argv, argv + argc, "-dump", std::string());
  ret.nclasses = get_argval(argv, argv + argc, "-nclusters", 2);
  ret.ncols = get_argval(argv, argv + argc, "-ncols", 81);
  ret.nrows = get_argval(argv, argv + argc, "-nrows", 10001);
  ret.allocate(handle);
  uint64_t seed = get_argval(argv, argv + argc, "-seed", 1234ULL);
  bool shuffle = get_argval(argv, argv + argc, "-shuffle");
  printf(
    "With params:\n"
    "  dimension    = %d,%d\n"
    "  center-box   = %f,%f\n"
    "  cluster-std  = %f\n"
    "  num-clusters = %d\n"
    "  seed         = %lu\n"
    "  shuffle      = %d\n",
    ret.nrows, ret.ncols, centerBoxMin, centerBoxMax, clusterStd, ret.nclasses,
    seed, shuffle);
  Datasets::make_blobs(handle, ret.X, ret.y, ret.nrows, ret.ncols, ret.nclasses,
                       nullptr, nullptr, clusterStd, shuffle, centerBoxMin,
                       centerBoxMax, seed);
  if (dump != "") dumpDataset(handle, ret, dump);
  return true;
}

bool load(Dataset& ret, const cumlHandle& handle, int argc, char** argv) {
  bool help = get_argval(argv, argv + argc, "-h");
  if (help) {
    printf(
      "USAGE:\n"
      "bench load [options]\n"
      "  Load the dataset from the input text file.\n"
      "OPTIONS:\n"
      "  -file <file>   file containing the dataset. Mandatory. File format\n"
      "                 is the same as generated by the '-dump' option.\n"
      "  -h             print this help and exit.\n");
    return false;
  }
  std::string file = get_argval(argv, argv + argc, "-file", std::string());
  ASSERT(!file.empty(), "'-file' is a mandatory option");
  printf("Loading dataset from file '%s'...\n", file.c_str());
  FILE* fp = fopen(file.c_str(), "r");
  ASSERT(fscanf(fp, "%d%d%d", &(ret.nrows), &(ret.ncols), &(ret.nclasses)) == 3,
         "Input dataset file is incorrect! No 'rows cols classes' info found");
  std::vector<float> X(ret.nrows * ret.ncols);
  std::vector<int> y(ret.nrows);
  for (int i = 0, k = 0; i < ret.nrows; ++i) {
    for (int j = 0; j < ret.ncols; ++j, ++k) {
      ASSERT(fscanf(fp, "%f", &(X[k])) == 1,
             "Failed to read input at row,col=%d,%d", i, j);
    }
    ASSERT(fscanf(fp, "%d", &(y[i])) == 1, "Failed to read the label at row=%d",
           i);
  }
  fclose(fp);
  ret.allocate(handle);
  auto stream = handle.getStream();
  MLCommon::copy(ret.X, &(X[0]), ret.nrows * ret.ncols, stream);
  MLCommon::copy(ret.y, &(y[0]), ret.nrows, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return true;
}

bool rf_csv(Dataset& ret, const cumlHandle& handle, int argc, char** argv) {
  bool help = get_argval(argv, argv + argc, "-h");
  if (help) {
    printf(
      "USAGE:\n"
      "bench rf_csv [options]\n"
      "  Load the dataset from the input csv file.\n"
      "  The current implementation only supports datasets used in random "
      "forest.\n"
      "OPTIONS:\n"
      "  -file <file>     file containing the dataset. Mandatory.\n"
      "  -dataset <name>  must be one of the following: higgs, year, airline,"
      "                   airline_regression\n"
      "  -row             number of rows to read. Mandatory.\n"
      "  -col             number of cols to read. Default to the cols in "
      "                   the dataset.\n"
      "  -h               print this help and exit.\n");
    return false;
  }
  std::string file = get_argval(argv, argv + argc, "-file", std::string());
  ASSERT(!file.empty(), "'-file' is a mandatory option");
  std::string dataset =
    get_argval(argv, argv + argc, "-dataset", std::string());
  ASSERT(!dataset.empty(), "'-dataset' is a mandatory option");
  ret.ncols = get_argval(argv, argv + argc, "-col", -1);
  ret.nrows = get_argval(argv, argv + argc, "-row", -1);
  ASSERT(ret.nrows != -1, "'-row' is a mandatory option");
  int col_offset = 0;
  int label_id = 0;  // column that is the label (i.e., target feature)
  if (dataset == "higgs") {
    if (ret.ncols == -1) ret.ncols = 28;
    col_offset = 1;  // because the first column in higgs is the label
  } else if (dataset == "year") {
    if (ret.ncols == -1) ret.ncols = 90;
    col_offset = 1;  // because the first column in year is the label
    label_id = 0;
  } else if ((dataset == "airline_regression") || (dataset == "airline")) {
    if (ret.ncols == -1) ret.ncols = 13;
    label_id = 13;
  }
  printf("Loading %s dataset from file '%s'...\n", dataset.c_str(),
         file.c_str());
  std::vector<float> X(ret.nrows * ret.ncols);
  std::vector<int> y(ret.nrows);
  std::ifstream myfile;
  myfile.open(file);
  std::string line;
  int counter = 0;
  int break_cnt = ret.nrows;
  while (getline(myfile, line) && (counter < ret.nrows)) {
    std::stringstream str(line);
    std::vector<float> row;
    float i;
    while (str >> i) {
      row.push_back(i);
      if (str.peek() == ',') str.ignore();
    }
    for (int col = 0; col < ret.ncols; col++) {
      X[counter + col * ret.nrows] =
        row[col + col_offset];  //train data should be col major
    }
    y[counter] =
      (dataset == "airline") ? (int)(row[label_id] > 0) : row[label_id];
    counter++;
  }
  std::cout << "Lines processed " << counter << std::endl;
  myfile.close();
  ret.allocate(handle);
  auto stream = handle.getStream();
  MLCommon::copy(ret.X, &(X[0]), ret.nrows * ret.ncols, stream);
  MLCommon::copy(ret.y, &(y[0]), ret.nrows, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return true;
}

typedef bool (*dataGenerator)(Dataset&, const cumlHandle&, int, char**);
class Generator : public std::map<std::string, dataGenerator> {
 public:
  Generator() : std::map<std::string, dataGenerator>() {
    (*this)["blobs"] = blobs;
    (*this)["load"] = load;
    (*this)["rf_csv"] = rf_csv;
  }
};

/// Do NOT touch anything below this line! ///
/// Only add new loaders above this line ///

Generator& generator() {
  static Generator map;
  return map;
}

std::string allGeneratorNames() {
  const auto& gen = generator();
  std::string ret;
  for (const auto& itr : gen) {
    ret += itr.first + "|";
  }
  ret.pop_back();
  return ret;
}

int findGeneratorStart(int argc, char** argv) {
  const auto& gen = generator();
  for (int i = 0; i < argc; ++i) {
    for (const auto& itr : gen) {
      if (!std::strcmp(itr.first.c_str(), argv[i])) return i;
    }
  }
  return argc;
}

bool loadDataset(Dataset& ret, const cumlHandle& handle, int argc,
                 char** argv) {
  std::string type = argc > 0 ? argv[0] : "blobs";
  auto& gen = generator();
  const auto& itr = gen.find(type);
  ASSERT(itr != gen.end(), "loadDataset: invalid generator name '%s'",
         type.c_str());
  struct timeval start;
  TIC(start);
  auto status = itr->second(ret, handle, argc, argv);
  if (status) {
    printf("dataset dimension: %d x %d\n", ret.nrows, ret.ncols);
    TOC(start, "dataset generation time");
  }
  return status;
}

}  // end namespace Bench
}  // end namespace ML
