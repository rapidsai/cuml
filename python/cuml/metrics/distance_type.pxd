cdef extern from "raft/linalg/distance_type.h" namespace "raft::distance":

    ctypedef enum DistanceType:
        EucExpandedL2 "raft::distance::DistanceType::EucExpandedL2"
        EucExpandedL2Sqrt "raft::distance::DistanceType::EucExpandedL2Sqrt"
        EucExpandedCosine "raft::distance::DistanceType::EucExpandedCosine"
        EucUnexpandedL1 "raft::distance::DistanceType::EucUnexpandedL1"
        EucUnexpandedL2 "raft::distance::DistanceType::EucUnexpandedL2"
        EucUnexpandedL2Sqrt "raft::distance::DistanceType::EucUnexpandedL2Sqrt"
