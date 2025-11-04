#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

cdef extern from "cuml/common/distance_type.hpp" namespace "ML::distance" nogil:

    ctypedef enum class DistanceType:
        L2Expanded "ML::distance::DistanceType::L2Expanded"
        L2SqrtExpanded "ML::distance::DistanceType::L2SqrtExpanded"
        CosineExpanded "ML::distance::DistanceType::CosineExpanded"
        L1 "ML::distance::DistanceType::L1"
        L2Unexpanded "ML::distance::DistanceType::L2Unexpanded"
        L2SqrtUnexpanded "ML::distance::DistanceType::L2SqrtUnexpanded"
        InnerProduct "ML::distance::DistanceType::InnerProduct"
        Linf "ML::distance::DistanceType::Linf"
        Canberra "ML::distance::DistanceType::Canberra"
        LpUnexpanded "ML::distance::DistanceType::LpUnexpanded"
        CorrelationExpanded "ML::distance::DistanceType::CorrelationExpanded"
        JaccardExpanded "ML::distance::DistanceType::JaccardExpanded"
        HellingerExpanded "ML::distance::DistanceType::HellingerExpanded"
        Haversine "ML::distance::DistanceType::Haversine"
        BrayCurtis "ML::distance::DistanceType::BrayCurtis"
        JensenShannon "ML::distance::DistanceType::JensenShannon"
        HammingUnexpanded "ML::distance::DistanceType::HammingUnexpanded"
        KLDivergence "ML::distance::DistanceType::KLDivergence"
        RusselRaoExpanded "ML::distance::DistanceType::RusselRaoExpanded"
        DiceExpanded "ML::distance::DistanceType::DiceExpanded"
        Precomputed "ML::distance::DistanceType::Precomputed"
