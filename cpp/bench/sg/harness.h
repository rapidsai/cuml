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

#include <stdio.h>
#include <utils.h>
#include <chrono>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace ML {
namespace Bench {

/** a simple struct to store achieved run information of a benchmark */
struct RunInfo {
  /** name of the test */
  std::string name;
  /** params for this run as a string */
  std::string params;
  /** whether the test ran successfully */
  bool passed;
  /** error message in case of failed test */
  std::string errMsg;
  /** metrics */
  std::unordered_map<std::string, float> metrics;
  /** runtimes (in ms) */
  std::unordered_map<std::string, float> runtimes;

  RunInfo()
    : name(),
      params(),
      passed(false),
      errMsg("Not run"),
      metrics(),
      runtimes() {}

  /** prints the current bench's runtime info */
  void printRunInfo() const;

  /** print test results header */
  static void printHeader();
};
typedef std::vector<RunInfo> RunInfos;

/**
 * @brief The base benchmark class
 * @tparam Params the parameters for this run
 */
template <typename Params>
struct Benchmark {
  /** params to be used for this run */
  void setParams(const Params &_p) { params = _p; }

  const Params &getParams() const { return params; }

  /** setup method. To be typically used to set the current run */
  void setup() {}

  /** teardown method. To be typically used to clean up after */
  void teardown() {}

  /** running the main test */
  void run() {}

  /** compute any metrics after the training is over */
  void metrics(RunInfo &ri) {}

 protected:
  /** params for this benchmark run */
  Params params;
};

/** Helper class to run the benchmark with all params */
class Runner {
 public:
  virtual ~Runner() {}
  virtual RunInfos run(const std::string &name,
                       const std::vector<std::string> &toRun) = 0;

 protected:
  /**
   * @brief Main function to run a benchmark with different params
   * @tparam BenchType benchmark to be run
   * @tparam Params params struct used by this benchmark
   * @param name name of the benchmark to be run
   * @param all list of all params to be benchmarked
   * @param toRun list of strings, only those test names matching, will be run
   * @return list of perf numbers for each test run
   */
  template <typename BenchType, typename Params>
  RunInfos runImpl(const std::string &name, const std::vector<Params> &all,
                   const std::vector<std::string> &toRun) const {
    RunInfos ret;
    int idx = 0;
    for (const auto &p : all) {
      auto totalStart = std::chrono::high_resolution_clock::now();
      std::ostringstream oss;
      oss << name << "/" << idx;
      ++idx;
      auto tName = oss.str();
      bool match = toRun.empty();
      for (const auto &name : toRun) {
        if (tName.find(name) != std::string::npos) {
          match = true;
          break;
        }
      }
      if (!match) continue;
      std::shared_ptr<BenchType> test(new BenchType);
      RunInfo ri;
      ri.name = tName;
      ri.params = p.str();
      // setup
      {
        auto start = std::chrono::high_resolution_clock::now();
        test->setParams(p);
        test->setup();
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = end - start;
        ri.runtimes["setup"] =
          std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
      }
      printf("%s: ", tName.c_str());
      fflush(stdout);
      // main training run
      try {
        auto start = std::chrono::high_resolution_clock::now();
        test->run();
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = end - start;
        ri.runtimes["run"] =
          std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
        ri.passed = true;
        ri.errMsg.clear();
      } catch (const std::runtime_error &re) {
        ri.errMsg = re.what();
      } catch (const MLCommon::Exception &mle) {
        ri.errMsg = mle.what();
      } catch (...) {
        ri.errMsg = "Unknown exception!";
      }
      // compute trained model metrics
      {
        auto start = std::chrono::high_resolution_clock::now();
        test->metrics(ri);
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = end - start;
        ri.runtimes["metrics"] =
          std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
      }
      // teardown
      {
        auto start = std::chrono::high_resolution_clock::now();
        test->teardown();
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = end - start;
        ri.runtimes["teardown"] =
          std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
      }
      ret.push_back(ri);
      auto totalEnd = std::chrono::high_resolution_clock::now();
      auto totalD = totalEnd - totalStart;
      double totalTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(totalD).count();
      printf("%s [in %lf ms]", ri.passed ? "OK" : "FAIL", totalTime);
      if (!ri.passed && !ri.errMsg.empty()) {
        printf(" (%s)", ri.errMsg.c_str());
      }
      printf("\n");
      fflush(stdout);
    }
    return ret;
  }
};

/** Main Harnessing suite */
class Harness {
 public:
  /** singleton-like getter interface */
  static Harness &get();

  /** initialize the benchmarking suite */
  static void Init(int argc, char **argv);

  /** run the benchmarking tests */
  static void RunAll();

  /** print csv formatted output */
  static void PrintResultsInCsv();

  /** total benchmark tests run */
  static size_t TotalTestsRun();

  /** register a benchmark runner to be run later */
  static void RegisterRunner(const std::string &name,
                             std::shared_ptr<Runner> r);

 private:
  Harness() : initialized(false), info(), runners(), toRun() {}
  ~Harness() {}

  /** flag to check initialization */
  bool initialized;
  /** info about all the benchmarks run */
  std::vector<RunInfos> info;
  /** list of all benchmarks to be run */
  std::map<std::string, std::shared_ptr<Runner>> runners;
  /** list of benchmarks to be filtered before running */
  std::vector<std::string> toRun;
};

template <typename BType, typename PType>
class RunnerImpl : public Runner {
 public:
  RunnerImpl(const std::vector<PType> &a) : allP(a) {}

  RunInfos run(const std::string &n, const std::vector<std::string> &toRun) {
    return runImpl<BType, PType>(n, allP, toRun);
  }

 private:
  std::vector<PType> allP;
};

template <typename BType, typename PType>
class Registrar {
 public:
  Registrar(const std::string &n, const std::vector<PType> &a) {
    auto *r = new RunnerImpl<BType, PType>(a);
    Harness::RegisterRunner(n, std::shared_ptr<Runner>(r));
  }
};

/** helper macro for registering a runner into the Harness */
#define REGISTER_BENCH(Bench, Params, name, allPs) \
  static Registrar<Bench, Params> tmpvar_##name(#name, allPs)

};  // end namespace Bench
};  // end namespace ML
