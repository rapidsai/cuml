/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cuml/common/lto/load_fatbins.hpp>

#include <cuda.h>

#include <nvJitLink.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#define DEMO_CUDA_TRY(_call)                                                     \
  do {                                                                           \
    CUresult result = (_call);                                                   \
    if (result != CUDA_SUCCESS) {                                                \
      const char* msg;                                                           \
      cuGetErrorName(result, &msg);                                              \
      std::cerr << "\nCUDA Error: " #_call " failed with error " << msg << '\n'; \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

namespace {
// We can make a better RAII wrapper around nvjitlinkhandle
void check_nvjitlink_result(nvJitLinkHandle handle, nvJitLinkResult result)
{
  if (result != NVJITLINK_SUCCESS) {
    std::cerr << "\n nvJITLink failed with error " << result << '\n';
    size_t log_size = 0;
    result          = nvJitLinkGetErrorLogSize(handle, &log_size);
    if (result == NVJITLINK_SUCCESS && log_size > 0) {
      std::unique_ptr<char[]> log{new char[log_size]};
      result = nvJitLinkGetErrorLog(handle, log.get());
      if (result == NVJITLINK_SUCCESS) {
        std::cerr << "nvJITLink error log: " << log.get() << '\n';
      }
    }
    exit(1);
  }
}
}  // namespace

CUlibrary load_fatbins(std::vector<std::string> fatbin_names)
{
  static CUdevice device;
  static CUcontext cuda_context;
  cuInit(0);
  DEMO_CUDA_TRY(cuDeviceGet(&device, 0));
  DEMO_CUDA_TRY(cuCtxCreate(&cuda_context, 0, device));

  int major = 0;
  int minor = 0;
  DEMO_CUDA_TRY(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  DEMO_CUDA_TRY(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

  std::string archs = "-arch=sm_" + std::to_string((major * 10 + minor));

  // Load the generated LTO IR and link them together
  nvJitLinkHandle handle;
  const char* lopts[] = {"-lto", archs.c_str()};
  auto result         = nvJitLinkCreate(&handle, 2, lopts);
  check_nvjitlink_result(handle, result);

  // load any fatbin files
  for (auto name : fatbin_names) {
    // need to compute the path to `name`
    std::cout << "attempting to add " << name << " to the nvJITLink module \n";
    result = nvJitLinkAddFile(handle, NVJITLINK_INPUT_FATBIN, name.c_str());
    check_nvjitlink_result(handle, result);
    std::cout << "\t\tadding " << name << " to the nvJITLink module \n";
  }

  // Call to nvJitLinkComplete causes linker to link together all the LTO-IR
  // modules perform any optimizations and generate cubin from it.
  std::cout << "\tStarted LTO runtime linking \n";
  result = nvJitLinkComplete(handle);
  check_nvjitlink_result(handle, result);
  std::cout << "\tCompleted LTO runtime linking \n";

  // get cubin from nvJitLink
  size_t cubin_size;
  result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
  check_nvjitlink_result(handle, result);

  std::unique_ptr<char[]> cubin{new char[cubin_size]};
  result = nvJitLinkGetLinkedCubin(handle, cubin.get());
  check_nvjitlink_result(handle, result);

  result = nvJitLinkDestroy(&handle);
  check_nvjitlink_result(handle, result);

  // cubin is linked, so now load it
  CUlibrary library;
  DEMO_CUDA_TRY(cuLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  return library;
}
