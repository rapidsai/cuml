/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <unordered_map>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <stdio.h>
#include "cuda_utils.h"


namespace MLCommon {


/** Data holder for high-level details of a GPU, useful for computing SOL */
struct GpuDetails {
    /** cuda device ID */
    int id;
    /** num SMs */
    int nSMs;
    /** mem bus width (in B) */
    int busWidth;
    /** SM clock (MHz) */
    int smClk;
    /** Mem clock (MHz) */
    int memClk;

    /** DRAM BW (in GBps) */
    float getMemBW() const {
        // 2x because of DDR
        return memClk * 2.f * busWidth / 1000.f;
    }
};


/** Defines the possible limiters in achieving perfect SOL */
enum SolLimiter {
    /** compute is the limiter */
    Limiter_Compute = 0,
    /** memory is the limiter */
    Limiter_Memory
};


/**
 * Represents an ideal runtime of a given workload. Info here will subsequently
 * be used to construct a feeds-and-speeds model to estimate achieved SOL%
 */
struct IdealTime {
    /** ideal time spent on computing (in ms) */
    float compute;
    /** ideal time spent on transferring in-and-out of memory (in ms) */
    float memory;
    /** expected overall runtime (in ms) */
    float total;
    /** limiter type */
    SolLimiter limiter;

    /** ctor */
    IdealTime(float com, float mem);
};


/** a simple struct to store achieved SOL information for a test */
struct BenchInfo {
    /** name of the test */
    std::string name;
    /** params for this run as a string */
    std::string params;
    /** ideal/estimated runtime */
    IdealTime ideal;
    /** actual runtime (in ms) */
    float runtime;
    /** whether the test ran successfully */
    bool passed;
    /** achieved SOL% */
    float achieved;
    /** error message in case of failed test */
    std::string errMsg;

    /** default ctor */
    BenchInfo(): name(), params(), ideal(0.f,0.f), runtime(0.f), passed(false),
                 achieved(0.f), errMsg("Not run") {}

    /** prints the current bench's runtime info */
    void printInfo() const;

    /** print test results header */
    static void printHeader();
};
typedef std::vector<BenchInfo> BenchInfos;


/** 
 * @brief The base benchmark class
 * @tparam Params the parameters for this run
 */
template <typename Params>
struct Benchmark {
public:
    /** params to be used for this run */
    void setParams(const Params& _p) { params = _p; }

    /** setup method. To be typically used to set the current run */
    void setup() {}

    /** teardown method. To be typically used to clean up after */
    void teardown() {}

    /** running the main test */
    void run() {}

    /** Compute the expected compute/memory times */
    IdealTime getIdealTime() const { return IdealTime(0.f,0.f); }

protected:
    /** params for this benchmark run */
    Params params;
};


/** Helper class to run the benchmark with all params */
class Runner {
public:
    virtual ~Runner() {}
    virtual BenchInfos run(const std::string& name) = 0;

protected:
    /**
     * @brief Main function to run a benchmark with different params
     * @tparam BenchType benchmark to be run
     * @tparam Params params struct used by this benchmark
     * @param name name of the benchmark to be run
     * @param all list of all params to be benchmarked
     * @return list of perf numbers for each test run
     */
    template <typename BenchType, typename Params>
    BenchInfos runImpl(const std::string& name,
                       const std::vector<Params>& all) const {
        BenchInfos ret;
        int idx = 0;
        for(const auto& p : all) {
            std::shared_ptr<BenchType> test(new BenchType);
            test->setParams(p);
            test->setup();
            std::ostringstream oss;
            oss << name << "/" << idx;
            printf("%s: ", oss.str().c_str());
            fflush(stdout);
            BenchInfo bi;
            bi.name = oss.str();
            bi.params = p.str();
            bi.ideal = test->getIdealTime();
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            try {
                CUDA_CHECK(cudaEventRecord(start));
                test->run();
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&bi.runtime, start, stop));
                bi.passed = true;
                bi.achieved = bi.ideal.total / bi.runtime * 100.f;
                bi.errMsg.clear();
            } catch(const std::runtime_error& re) {
                bi.errMsg = re.what();
            } catch(...) {
                bi.errMsg = "Unknown exception!";
            }
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            ret.push_back(bi);
            test->teardown();
            ++idx;
            printf("%s", bi.passed? "OK" : "FAIL");
            if(!bi.passed && !bi.errMsg.empty())
                printf(" (%s)", bi.errMsg.c_str());
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
    static Harness& get();

    /** initialize the benchmarking suite */
    static void Init(int argc, char** argv);

    /** run the benchmarking tests */
    static void RunAll();

    /** register a benchmark runner to be run later */
    static void RegisterRunner(const std::string& name,
                               std::shared_ptr<Runner> r);

    /* get details of the given GPU for SOL computation */
    static GpuDetails GetDetails(int devId=-1);

private:
    Harness();
    ~Harness();
    void generateMapping();

    /** cuda device to nvml device mapping */
    std::unordered_map<int,int> cuda2nvml;
    /** flag to check initialization */
    bool initialized;
    /** info about all the benchmarks run */
    std::vector<BenchInfos> info;
    /** list of all benchmarks to be run */
    std::map<std::string, std::shared_ptr<Runner> > runners;
};


/** helper macro for registering a runner into the Harness */
#define REGISTER_BENCH(Bench, Params, name, allPs)                      \
    template <typename BType, typename PType>                           \
    class MyRunner : public Runner {                                    \
    public:                                                             \
        MyRunner(const std::vector<PType>& a): allP(a) {}               \
        BenchInfos run(const std::string& n) {                          \
            return runImpl<BType,PType>(n, allP);                       \
        }                                                               \
    private:                                                            \
        std::vector<PType> allP;                                        \
    };                                                                  \
                                                                        \
    template <typename BType, typename PType>                           \
    class Registrar {                                                   \
    public:                                                             \
        Registrar(const std::string& n, const std::vector<PType>& a) {  \
            Runner* r = new MyRunner<BType,PType>(a);                   \
            Harness::RegisterRunner(n, std::shared_ptr<Runner>(r));     \
        }                                                               \
    };                                                                  \
                                                                        \
    static Registrar<Bench,Params> tmpvar(name, allPs)


}; // end namespace MLCommon
