#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
