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

#include <cuda_runtime.h>
#include <cuML.hpp>
#include "argparse.hpp"
#include "dataset.h"
#include "utils.h"

namespace ML {
namespace Bench {

int main_no_catch(int argc, char** argv) {
  int genStart = findGeneratorStart(argc, argv);
  bool help = get_argval(argv, argv + genStart, "-h");
  if (help) {
    auto gens = allGeneratorNames();
    printf(
      "USAGE:\n"
      "bench [-h] [<genType> [<genTypeOptions>]] ...\n"
      "  cuML c++ benchmark suite\n"
      "OPTIONS:\n"
      "  -h                Print this help and exit\n"
      "  <genType>         Dataset generator. [blobs]. Available types:\n"
      "                      (%s)\n"
      "  <genTypeOptions>  Options for each generator. Use '-h' option to a\n"
      "                    particular generator to know its options\n",
      gens.c_str());
    return 0;
  }
  ML::cumlHandle handle;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  handle.setStream(stream);
  ///@todo: set custom allocator
  Dataset data = {0, 0, nullptr, nullptr};
  ///@todo: findAlgoStart and offset the below call
  if (loadDataset(data, handle, argc - genStart, argv + genStart)) {
    ///@todo: call the algo
  }
  data.deallocate(handle);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // end namespace Bench
}  // end namespace ML

int main(int argc, char** argv) {
  try {
    return ML::Bench::main_no_catch(argc, argv);
  } catch (const MLCommon::Exception& ml_e) {
    printf("%s failed! Reason:\n%s\n", argv[0], ml_e.what());
    return -1;
  } catch (const std::exception& e) {
    printf("%s failed! Reason:\n%s\n", argv[0], e.what());
    return -2;
  } catch (...) {
    printf("%s failed! Unknown exception\n", argv[0]);
    return -3;
  }
}
