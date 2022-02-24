#
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cdef extern from "raft/distance/distance_type.hpp" namespace "raft::distance":

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
        HammingUnexpanded "raft::distance::DistanceType::HammingUnexpanded"
        KLDivergence "raft::distance::DistanceType::KLDivergence"
        RusselRaoExpanded "raft::distance::DistanceType::RusselRaoExpanded"
        DiceExpanded "raft::distance::DistanceType::DiceExpanded"
        Precomputed "raft::distance::DistanceType::Precomputed"
