cdef extern from "raft/linalg/distance_type.h" namespace "raft::distance":

    ctypedef enum DistanceType:
        EucExpandedL2 "raft::distance::DistanceType::L2Expanded"
        EucExpandedL2Sqrt "raft::distance::DistanceType::L2SqrtExpanded"
        EucExpandedCosine "raft::distance::DistanceType::CosineExpanded"
        EucUnexpandedL1 "raft::distance::DistanceType::L1"
        EucUnexpandedL2 "raft::distance::DistanceType::L2Unexpanded"
        EucUnexpandedL2Sqrt "raft::distance::DistanceType::L2SqrtUnexpanded"
