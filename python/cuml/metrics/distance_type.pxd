cdef extern from "raft/linalg/distance_type.h" namespace "raft::distance":

    ctypedef enum DistanceType:
        L2Expanded "raft::distance::DistanceType::L2Expanded"
        L2SqrtExpanded "raft::distance::DistanceType::L2SqrtExpanded"
        CosineExpanded "raft::distance::DistanceType::CosineExpanded"
        L1 "raft::distance::DistanceType::L1"
        L2Unexpanded "raft::distance::DistanceType::L2Unexpanded"
        L2SqrtUnexpanded "raft::distance::DistanceType::L2SqrtUnexpanded"
        InnerProduct "raft::distance::DistanceType::InnerProduct"
        Linf "raft::distance::DistanceType::Linf"
        Canberra "raft::distance::DistanceType::Canberra"
        LpUnexpanded "raft::distance::DistanceType::LpUnexpanded"
        CorrelationExpanded "raft::distance::DistanceType::CorrelationExpanded"
        JaccardExpanded "raft::distance::DistanceType::JaccardExpanded"
        HellingerExpanded "raft::distance::DistanceType::HellingerExpanded"
        Haversine "raft::distance::DistanceType::Haversine"
        BrayCurtis "raft::distance::DistanceType::BrayCurtis"
        JensenShannon "raft::distance::DistanceType::JensenShannon"