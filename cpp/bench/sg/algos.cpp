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
#include <map>
#include "dataset.h"

namespace ML {
namespace Bench {

bool dbscan(const Dataset& ret, const cumlHandle& handle, int argc,
            char** argv);

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
  return itr->second(ret, handle, argc, argv);
}

}  // end namespace Bench
}  // end namespace ML
