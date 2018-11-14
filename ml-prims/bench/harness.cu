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


#include "cuda_utils.h"
#include "harness.h"
#include <nvml.h>
#include <cuda_runtime_api.h>
#include <vector>


/** check for nvml runtime API errors */
#define NVML_CHECK(call)                        \
    do {                                        \
        nvmlReturn_t status = call;             \
        ASSERT(status == NVML_SUCCESS,          \
               "FAIL: call='%s'. Reason:%s\n",  \
               #call, nvmlErrorString(status)); \
    } while(0)


namespace MLCommon {

IdealTime::IdealTime(float com, float mem):
    compute(com), memory(mem), total(0.f), limiter(Limiter_Compute) {
    if(compute > memory) {
        total = compute;
        limiter = Limiter_Compute;
    } else {
        total = memory;
        limiter = Limiter_Memory;
    }
}


void BenchInfo::printInfo() const {
    printf("%s,\"%s\",%s,%f,%f,%f,%s,%f,%f,\"%s\"\n",
           name.c_str(),
           params.c_str(),
           passed? "OK" : "FAIL",
           ideal.compute,
           ideal.memory,
           ideal.total,
           ideal.limiter == Limiter_Compute? "Compute" : "Memory",
           runtime,
           achieved,
           errMsg.c_str());
}

void BenchInfo::printHeader() {
    printf("name,"
           "params,"
           "status,"
           "ideal compute time (ms),"
           "ideal memory time (ms),"
           "ideal runtime (ms),"
           "SOL limiter,"
           "measured time (ms),"
           "SOL%%,"
           "error\n");
}


Harness::Harness(): cuda2nvml(), initialized(false), info(), runners() {
    NVML_CHECK(nvmlInit());
    generateMapping();
}

Harness::~Harness() {
    NVML_CHECK(nvmlShutdown());
}

void Harness::generateMapping() {
    // get device info from cuda runtime API
    int cudaCount;
    CUDA_CHECK(cudaGetDeviceCount(&cudaCount));
    std::vector<struct cudaDeviceProp> cudaProps;
    cudaProps.resize(cudaCount);
    for(int i=0;i<cudaCount;++i)
        CUDA_CHECK(cudaGetDeviceProperties(&(cudaProps[i]), i));
    // get device info from nvml API
    unsigned nvmlCount;
    NVML_CHECK(nvmlDeviceGetCount(&nvmlCount));
    std::vector<nvmlPciInfo_t> nvmlProps;
    nvmlProps.resize(nvmlCount);
    for(unsigned i=0;i<nvmlCount;++i) {
        nvmlDevice_t dev;
        NVML_CHECK(nvmlDeviceGetHandleByIndex(i, &dev));
        NVML_CHECK(nvmlDeviceGetPciInfo(dev, &(nvmlProps[i])));
    }
    // map these two
    for(unsigned i=0;i<nvmlCount;++i) {
        const auto& ni = nvmlProps[i];
        for(int j=0;j<cudaCount;++j) {
            const auto& cj = cudaProps[j];
            if(ni.bus == cj.pciBusID && ni.device == cj.pciDeviceID &&
               ni.domain == cj.pciDomainID) {
                cuda2nvml[j] = i;
                break;
            }
        }
    }
}

Harness& Harness::get() {
    static Harness bench;
    return bench;
}

void Harness::Init(int argc, char** argv) {
    auto& b = get();
    ASSERT(!b.initialized, "Harness::Init: Already initialized!");
    b.initialized = true;
}

void Harness::RunAll() {
    auto& b = get();
    ASSERT(b.initialized, "Harness::Run: Not yet initialized!");
    for(auto itr : b.runners) {
        auto ret = itr.second->run(itr.first);
        b.info.push_back(ret);
    }
    if(b.info.empty()) {
        printf("No tests ran!\n");
        return;
    }
    BenchInfo::printHeader();
    for(const auto& itr : b.info)
        for(const auto& bitr : itr)
            bitr.printInfo();
}

void Harness::RegisterRunner(const std::string& name,
                             std::shared_ptr<Runner> r) {
    auto& b = get();
    ASSERT(!b.initialized, "Harness::RegisterRunner: Already initialized!");
    ASSERT(b.runners.find(name) == b.runners.end(),
           "Harness::RegisterRunner: benchmark named '%s' already registered!",
           name.c_str());
    b.runners[name] = r;
}

GpuDetails Harness::GetDetails(int devId) {
    auto& b = get();
    if(devId < 0)
        CUDA_CHECK(cudaGetDevice(&devId));
    const auto itr = b.cuda2nvml.find(devId);
    ASSERT(itr != b.cuda2nvml.end(),
           "getClocks: Failed to find nvml-device for cuda devID=%d!", devId);
    struct cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, devId));
    GpuDetails res;
    // device id
    res.id = devId;
    // num sm's
    res.nSMs = props.multiProcessorCount;
    // bus width
    res.busWidth = props.memoryBusWidth / 8;
    nvmlDevice_t dev;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(itr->second, &dev));
    unsigned clock;
    // sm clk
    NVML_CHECK(nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &clock));
    res.smClk = (int)clock;
    // mem clk
    NVML_CHECK(nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &clock));
    res.memClk = (int)clock;
    return res;
}

} // end namespace MLCommon
