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

#include "harness.h"
#include <utils.h>

namespace ML {
namespace Bench {

void RunInfo::printRunInfo() const {
  printf("%s,%s,%f,%f,%f,%f,\"%s\",\"%s\"\n", name.c_str(),
         passed ? "OK" : "FAIL", runtimes.at("run"), runtimes.at("setup"),
         runtimes.at("teardown"), runtimes.at("metrics"), errMsg.c_str(),
         params.c_str());
}

void RunInfo::printHeader() {
  printf(
    "name,"
    "status,"
    "run time (ms),"
    "setup time (ms),"
    "teardown time (ms),"
    "metrics time (ms),"
    "error,"
    "params\n");
}

Harness &Harness::get() {
  static Harness bench;
  return bench;
}

void Harness::Init(int argc, char **argv) {
  auto &b = get();
  ASSERT(!b.initialized, "Harness::Init: Already initialized!");
  for (int i = 1; i < argc; ++i) {
    b.toRun.push_back(argv[i]);
  }
  b.initialized = true;
}

void Harness::RunAll() {
  auto &b = get();
  ASSERT(b.initialized, "Harness::RunAll: Not yet initialized!");
  for (auto itr : b.runners) {
    auto ret = itr.second->run(itr.first, b.toRun);
    b.info.push_back(ret);
  }
  if (b.info.empty()) {
    printf("No benchmarks ran!\n");
    return;
  }
  RunInfo::printHeader();
  for (const auto &itr : b.info)
    for (const auto &ritr : itr) ritr.printRunInfo();
}

void Harness::RegisterRunner(const std::string &name,
                             std::shared_ptr<Runner> r) {
  auto &b = get();
  ASSERT(!b.initialized, "Harness::RegisterRunner: Already initialized!");
  ASSERT(b.runners.find(name) == b.runners.end(),
         "Harness::RegisterRunner: benchmark named '%s' already registered!",
         name.c_str());
  b.runners[name] = r;
}

}  // end namespace Bench
}  // end namespace ML
