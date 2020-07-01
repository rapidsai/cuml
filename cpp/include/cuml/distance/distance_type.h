#pragma once

namespace ML {
namespace Distance {

/** enum to tell how to compute euclidean distance */
enum DistanceType : unsigned short {
  /** evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk) */
  EucExpandedL2 = 0,
  /** same as above, but inside the epilogue, perform square root operation */
  EucExpandedL2Sqrt = 1,
  /** cosine distance */
  EucExpandedCosine = 2,
  /** L1 distance */
  EucUnexpandedL1 = 3,
  /** evaluate as dist_ij += (x_ik - y-jk)^2 */
  EucUnexpandedL2 = 4,
  /** same as above, but inside the epilogue, perform square root operation */
  EucUnexpandedL2Sqrt = 5,
};

};  // end namespace Distance
};  // end namespace ML
