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

#include "selection/kselection.h"
#include "random/rng.h"
#include <gtest/gtest.h>
#include "harness.h"


namespace MLCommon {


    struct WarpTopKParams {
        int rows; //batch size
        int cols;// N the length of variables 
        int k; // the top-k value
        unsigned long long int seed;//seed to generate data

        std::string str() const {
            std::ostringstream oss;
            oss << "row=" << rows<<" cols="<<cols<<" k="<<k<<" "<<seed;
            return oss.str();
        }
    };

    template <typename T>
    struct WarpTopKBenchmark: Benchmark<WarpTopKParams> {
        void setup() {
            Random::Rng<T> r(params.seed);
            allocate(arr, params.rows*params.cols);
            allocate(outk, params.rows*params.k);
            allocate(outv, params.rows*params.k);
            r.uniform(arr, params.rows*params.cols, T(-1.0), T(1.0));
        }

        void teardown() {
            CUDA_CHECK(cudaFree(outv));
            CUDA_CHECK(cudaFree(outk));
            CUDA_CHECK(cudaFree(arr));
        }

        void run() {
            static const bool Sort=false;
            static const bool Greater=true;
            Selection::warpTopK<T,int,Greater,Sort>(outv, outk, arr, params.k, 
                    params.rows, params.cols);
 
        }

        IdealTime getIdealTime() const {
            auto gd = Harness::GetDetails();
            float mem = params.cols * (params.rows+params.k)* sizeof(T) / gd.getMemBW();// correct the ideal memory time
            mem /= 1e6f;   // time in ms
            // ignore compute time, since we are clearly BW bound here!
            return IdealTime(0.f, mem);
        }

        float *arr, *outv;
        int *outk;
    };

    static std::vector<WarpTopKParams> inputs = {
        // results in batchsize=1
        {2048, 1024, 1,1234ULL},
        {2048, 1024, 32,1234ULL},
        {2048, 1024, 64,1234ULL},
        {2048, 1024, 128,1234ULL},
        {2048, 1024, 256,1234ULL},
        {2048, 1024, 1024,1234ULL}

    };

    REGISTER_BENCH(WarpTopKBenchmark<float>, WarpTopKParams, "WarpTop", inputs);

} // end namespace MLCommon
