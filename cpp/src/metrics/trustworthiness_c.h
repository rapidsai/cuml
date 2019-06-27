/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

namespace MLCommon {
namespace Distance {
enum DistanceType {
  /** evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk) */
  EucExpandedL2 = 0,
  /** same as above, but inside the epilogue, perform square root operation */
  EucExpandedL2Sqrt,
  /** cosine distance */
  EucExpandedCosine,
  /** L1 distance */
  EucUnexpandedL1,
  /** evaluate as dist_ij += (x_ik - y-jk)^2 */
  EucUnexpandedL2,
  /** same as above, but inside the epilogue, perform square root operation */
  EucUnexpandedL2Sqrt,
};
}
};  // namespace MLCommon

using namespace MLCommon::Distance;

namespace ML {
namespace Metrics {

template <typename math_t, DistanceType distance_type>
double trustworthiness_score(const cumlHandle& h, math_t* X, math_t* X_embedded,
                             int n, int m, int d, int n_neighbors);

}
}  // namespace ML