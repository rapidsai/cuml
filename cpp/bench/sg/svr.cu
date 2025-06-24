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
#include <cuml/svm/svr.hpp>

#include <cmath>
#include <utility>

namespace ML {
namespace Bench {
namespace SVM {

template <typename D>
struct SvrParams {
  DatasetParams data;
  RegressionParams regression;
  ML::matrix::KernelParams kernel;
  ML::SVM::SvmParameter svm_param;
  ML::SVM::SvmModel<D>* model;
};

template <typename D>
class SVR : public RegressionFixture<D> {
 public:
  SVR(const std::string& name, const SvrParams<D>& p)
    : RegressionFixture<D>(name, p.data, p.regression),
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
    if (this->params.rowMajor) { state.SkipWithError("SVR only supports col-major inputs"); }
    if (this->svm_param.svmType != ML::SVM::EPSILON_SVR) {
      state.SkipWithError("SVR currently only supports EPSILON_SVR");
    }
    this->loopOnState(state, [this]() {
      ML::SVM::svrFit(*this->handle,
                      this->data.X.data(),
                      this->params.nrows,
                      this->params.ncols,
                      this->data.y.data(),
                      this->svm_param,
                      this->kernel,
                      *(this->model));
      this->handle->sync_stream(this->stream);
      ML::SVM::svmFreeBuffers(*this->handle, *(this->model));
    });
  }

 private:
  ML::matrix::KernelParams kernel;
  ML::SVM::SvmParameter svm_param;
  ML::SVM::SvmModel<D>* model;
};

template <typename D>
std::vector<SvrParams<D>> getInputs()
{
  struct Triplets {
    int nrows, ncols, n_informative;
  };
  std::vector<SvrParams<D>> out;
  SvrParams<D> p;

  p.data.rowMajor = false;

  p.regression.shuffle        = true;  // better to shuffle when n_informative < ncols
  p.regression.seed           = 1378ULL;
  p.regression.effective_rank = -1;  // dataset generation will be faster
  p.regression.bias           = 0;
  p.regression.tail_strength  = 0.5;  // unused when effective_rank = -1
  p.regression.noise          = 1;

  // SvmParameter{C, cache_size, max_iter, nochange_steps, tol, verbosity,
  //              epsilon, svmType})
  p.svm_param = ML::SVM::SvmParameter{
    1, 200, 200, 100, 1e-3, rapids_logger::level_enum::info, 0.1, ML::SVM::EPSILON_SVR};
  p.model = new ML::SVM::SvmModel<D>{0, 0, 0, 0};

  std::vector<Triplets> rowcols = {{50000, 2, 2}, {1024, 10000, 10}, {3000, 200, 200}};

  std::vector<ML::matrix::KernelParams> kernels{{ML::matrix::KernelType::LINEAR, 3, 1, 0},
                                                {ML::matrix::KernelType::POLYNOMIAL, 3, 1, 0},
                                                {ML::matrix::KernelType::RBF, 3, 1, 0},
                                                {ML::matrix::KernelType::TANH, 3, 0.1, 0}};

  for (auto& rc : rowcols) {
    p.data.nrows               = rc.nrows;
    p.data.ncols               = rc.ncols;
    p.regression.n_informative = rc.n_informative;
    // Limit the number of iterations for large tests
    p.svm_param.max_iter = (rc.nrows > 10000) ? 50 : 200;
    for (auto kernel : kernels) {
      p.kernel       = kernel;
      p.kernel.gamma = 1.0 / rc.ncols;
      out.push_back(p);
    }
  }
  return out;
}

ML_BENCH_REGISTER(SvrParams<float>, SVR<float>, "regression", getInputs<float>());
ML_BENCH_REGISTER(SvrParams<double>, SVR<double>, "regression", getInputs<double>());

}  // namespace SVM
}  // namespace Bench
}  // end namespace ML
