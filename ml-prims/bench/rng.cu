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

#include "random/rng.h"
#include "harness.h"


namespace MLCommon {
namespace Random {

enum RandomType {
    RNG_Normal,
    RNG_LogNormal,
    RNG_Uniform,
    RNG_Gumbel,
    RNG_Logistic,
    RNG_Exp,
    RNG_Rayleigh,
    RNG_Laplace,
    RNG_Fill
};

struct RngParams {
    int len;
    RandomType type;
    unsigned long long seed;
    float start, end;

    std::string str() const {
        std::ostringstream oss;
        oss << "len=" << len << ";type=" << type << ";seed=" << seed
            << ";start=" << start << ";end=" << end;
        return oss.str();
    }
};

class RngBenchmark: public Benchmark<RngParams> {
public:
    void setup() {
        CUDA_CHECK(cudaMalloc((void**)&out, sizeof(float)*params.len));
    }

    void teardown() {
        CUDA_CHECK(cudaFree(out));
    }

    void run() {
        Rng<float> r(params.seed);
        switch(params.type) {
        case RNG_Normal:
            r.normal(out, params.len, params.start, params.end);
            break;
        case RNG_LogNormal:
            r.lognormal(out, params.len, params.start, params.end);
            break;
        case RNG_Uniform:
            r.uniform(out, params.len, params.start, params.end);
            break;
        case RNG_Gumbel:
            r.gumbel(out, params.len, params.start, params.end);
            break;
        case RNG_Logistic:
            r.logistic(out, params.len, params.start, params.end);
            break;
        case RNG_Exp:
            r.exponential(out, params.len, params.start);
            break;
        case RNG_Rayleigh:
            r.rayleigh(out, params.len, params.start);
            break;
        case RNG_Laplace:
            r.laplace(out, params.len, params.start, params.end);
            break;
        case RNG_Fill:
            r.fill(out, params.len, params.start);
            break;
        };
    }

    IdealTime getIdealTime() const {
        auto gd = Harness::GetDetails();
        float mem = params.len * sizeof(float) / gd.getMemBW();
        mem /= 1e6f;   // time in ms
        // ignore compute time, since we are clearly BW bound here!
        return IdealTime(0.f, mem);
    }

    float *out;
};


static std::vector<RngParams> inputsf = {
    // results in STG.128
    {1024*1024,        RNG_Uniform, 12345ULL, -1.f, 1.f},
    {32*1024*1024,     RNG_Uniform, 12345ULL, -1.f, 1.f},
    {1024*1024*1024,   RNG_Uniform, 12345ULL, -1.f, 1.f},
    // results in STG.64
    {1024*1024+2,      RNG_Uniform, 12345ULL, -1.f, 1.f},
    {32*1024*1024+2,   RNG_Uniform, 12345ULL, -1.f, 1.f},
    {1024*1024*1024+2, RNG_Uniform, 12345ULL, -1.f, 1.f},
    // results in STG.32
    {1024*1024+1,      RNG_Uniform, 12345ULL, -1.f, 1.f},
    {32*1024*1024+1,   RNG_Uniform, 12345ULL, -1.f, 1.f},
    {1024*1024*1024+1, RNG_Uniform, 12345ULL, -1.f, 1.f},

    {1024*1024,        RNG_Fill, 12345ULL, -1.f, 1.f},
    {32*1024*1024,     RNG_Fill, 12345ULL, -1.f, 1.f},
    {1024*1024*1024,   RNG_Fill, 12345ULL, -1.f, 1.f},
    {1024*1024+2,      RNG_Fill, 12345ULL, -1.f, 1.f},
    {32*1024*1024+2,   RNG_Fill, 12345ULL, -1.f, 1.f},
    {1024*1024*1024+2, RNG_Fill, 12345ULL, -1.f, 1.f},
    {1024*1024+1,      RNG_Fill, 12345ULL, -1.f, 1.f},
    {32*1024*1024+1,   RNG_Fill, 12345ULL, -1.f, 1.f},
    {1024*1024*1024+1, RNG_Fill, 12345ULL, -1.f, 1.f}
};
REGISTER_BENCH(RngBenchmark, RngParams, RngF, inputsf);

} // end namespace Random
} // end namespace MLCommon
