/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "benchmark.cuh"

#include <cuml/matrix/kernel_params.hpp>
#include <cuml/svm/svc.hpp>
#include <cuml/svm/svm_model.h>
#include <cuml/svm/svm_parameter.h>

#include <cmath>
#include <sstream>
#include <utility>

namespace ML {
namespace Bench {
namespace SVM {

template <typename D>
struct SvcParams {
  DatasetParams data;
  BlobsParams blobs;
  ML::matrix::KernelParams kernel;
  ML::SVM::SvmParameter svm_param;
  ML::SVM::SvmModel<D> model;
};

template <typename D>
class SVC : public BlobsFixture<D, D> {
 public:
  SVC(const std::string& name, const SvcParams<D>& p)
    : BlobsFixture<D, D>(name, p.data, p.blobs),
      kernel(p.kernel),
      model(p.model),
      svm_param(p.svm_param)
  {
    std::vector<std::string> kernel_names{"linear", "poly", "rbf", "tanh"};
    std::ostringstream oss;
    oss << name << "/" << kernel_names[static_cast<int>(kernel.kernel)] << p.data;
    this->SetName(oss.str().c_str());
  }

 protected:
  void runBenchmark(::benchmark::State& state) override
  {
    if (this->params.rowMajor) { state.SkipWithError("SVC only supports col-major inputs"); }
    if (this->svm_param.svmType != ML::SVM::C_SVC) {
      state.SkipWithError("SVC currently only supports C_SVC");
    }
    this->loopOnState(state, [this]() {
      ML::SVM::svcFit(*this->handle,
                      this->data.X.data(),
                      this->params.nrows,
                      this->params.ncols,
                      this->data.y.data(),
                      this->svm_param,
                      this->kernel,
                      this->model,
                      static_cast<D*>(nullptr));
      this->handle->sync_stream(this->stream);
      ML::SVM::svmFreeBuffers(*this->handle, this->model);
    });
  }

 private:
  ML::matrix::KernelParams kernel;
  ML::SVM::SvmParameter svm_param;
  ML::SVM::SvmModel<D> model;
};

template <typename D>
std::vector<SvcParams<D>> getInputs()
{
  struct Triplets {
    int nrows, ncols, nclasses;
  };
  std::vector<SvcParams<D>> out;
  SvcParams<D> p;

  p.data.rowMajor = false;

  p.blobs.cluster_std    = 1.0;
  p.blobs.shuffle        = false;
  p.blobs.center_box_min = -2.0;
  p.blobs.center_box_max = 2.0;
  p.blobs.seed           = 12345ULL;

  // SvmParameter{C, cache_size, max_iter, nochange_steps, tol, verbosity})
  p.svm_param = ML::SVM::SvmParameter{
    1, 200, 100, 100, 1e-3, rapids_logger::level_enum::info, 0, ML::SVM::C_SVC};
  p.model = ML::SVM::SvmModel<D>{0, 0, 0, nullptr, {}, nullptr, 0, nullptr};

  std::vector<Triplets> rowcols = {{50000, 2, 2}, {2048, 100000, 2}, {50000, 1000, 2}};

  std::vector<ML::matrix::KernelParams> kernels{{ML::matrix::KernelType::LINEAR, 3, 1, 0},
                                                {ML::matrix::KernelType::POLYNOMIAL, 3, 1, 0},
                                                {ML::matrix::KernelType::RBF, 3, 1, 0},
                                                {ML::matrix::KernelType::TANH, 3, 0.1, 0}};

  for (auto& rc : rowcols) {
    p.data.nrows    = rc.nrows;
    p.data.ncols    = rc.ncols;
    p.data.nclasses = rc.nclasses;
    // Limit the number of iterations for large tests
    p.svm_param.max_iter = (rc.nrows > 10000) ? 20 : 100;
    for (auto kernel : kernels) {
      p.kernel       = kernel;
      p.kernel.gamma = 1.0 / rc.ncols;
      out.push_back(p);
    }
  }
  return out;
}

ML_BENCH_REGISTER(SvcParams<float>, SVC<float>, "blobs", getInputs<float>());
ML_BENCH_REGISTER(SvcParams<double>, SVC<double>, "blobs", getInputs<double>());

}  // namespace SVM
}  // namespace Bench
}  // end namespace ML
