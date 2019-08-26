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

namespace ML {
namespace SVM {

/**
 * Numerical input parameters for an SVM.
 */
struct svmParameter {
    double C;      //!< Penalty term C
    double cache_size;  //!< kernel cache size in MiB
    //! maximum number of outer SMO iterations. Use -1 to let the SMO solver set
    //! a default value (100*n_rows).
    int max_iter;
    double tol;    //!< Tolerance used to stop fitting.
    int verbose;  //!< Print information about traning
};

}; // namespace SVM
}; // namespace ML
