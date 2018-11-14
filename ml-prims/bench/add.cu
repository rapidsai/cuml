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


#include "linalg/add.h"
#include "harness.h"


namespace MLCommon {

struct AddParams {
    int len;

    std::string str() const {
        std::ostringstream oss;
        oss << "len=" << len;
        return oss.str();
    }
};

struct AddBenchmark: Benchmark<AddParams> {
    void setup() {
        CUDA_CHECK(cudaMalloc((void**)&ptr0, sizeof(float)*params.len));
        CUDA_CHECK(cudaMalloc((void**)&ptr1, sizeof(float)*params.len));
    }

    void teardown() {
        CUDA_CHECK(cudaFree(ptr0));
        CUDA_CHECK(cudaFree(ptr1));
    }

    void run() {
        LinAlg::add(ptr0, ptr0, ptr1, params.len);
    }

    IdealTime getIdealTime() const {
        auto gd = Harness::GetDetails();
        float mem = params.len * sizeof(float) * 3.f / gd.getMemBW();
        mem /= 1e6f;   // time in ms
        // ignore compute time, since we are clearly BW bound here!
        return IdealTime(0.f, mem);
    }

    float *ptr0, *ptr1;
};

static std::vector<AddParams> inputs = {
    // results in LDG.128
    {1024*1024},
    {32*1024*1024},
    {1024*1024*1024},
    // results in LDG.64
    {1024*1024+2},
    {32*1024*1024+2},
    {1024*1024*1024+2},
    // results in LDG.32
    {1024*1024+1},
    {32*1024*1024+1},
    {1024*1024*1024+1}
};

REGISTER_BENCH(AddBenchmark, AddParams, "Add", inputs);

} // end namespace MLCommon
