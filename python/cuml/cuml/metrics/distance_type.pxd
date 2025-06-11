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

cdef extern from "cuml/cuvs_stubs/distance_type.hpp" namespace "MLCommon::CuvsStubs" nogil:

    ctypedef enum class DistanceType:
        L2Expanded "MLCommon::CuvsStubs::DistanceType::L2Expanded"
        L2SqrtExpanded "MLCommon::CuvsStubs::DistanceType::L2SqrtExpanded"
        CosineExpanded "MLCommon::CuvsStubs::DistanceType::CosineExpanded"
        L1 "MLCommon::CuvsStubs::DistanceType::L1"
        L2Unexpanded "MLCommon::CuvsStubs::DistanceType::L2Unexpanded"
        L2SqrtUnexpanded "MLCommon::CuvsStubs::DistanceType::L2SqrtUnexpanded"
        InnerProduct "MLCommon::CuvsStubs::DistanceType::InnerProduct"
        Linf "MLCommon::CuvsStubs::DistanceType::Linf"
        Canberra "MLCommon::CuvsStubs::DistanceType::Canberra"
        LpUnexpanded "MLCommon::CuvsStubs::DistanceType::LpUnexpanded"
        CorrelationExpanded "MLCommon::CuvsStubs::DistanceType::CorrelationExpanded"
        JaccardExpanded "MLCommon::CuvsStubs::DistanceType::JaccardExpanded"
        HellingerExpanded "MLCommon::CuvsStubs::DistanceType::HellingerExpanded"
        Haversine "MLCommon::CuvsStubs::DistanceType::Haversine"
        BrayCurtis "MLCommon::CuvsStubs::DistanceType::BrayCurtis"
        JensenShannon "MLCommon::CuvsStubs::DistanceType::JensenShannon"
        HammingUnexpanded "MLCommon::CuvsStubs::DistanceType::HammingUnexpanded"
        KLDivergence "MLCommon::CuvsStubs::DistanceType::KLDivergence"
        RusselRaoExpanded "MLCommon::CuvsStubs::DistanceType::RusselRaoExpanded"
        DiceExpanded "MLCommon::CuvsStubs::DistanceType::DiceExpanded"
        Precomputed "MLCommon::CuvsStubs::DistanceType::Precomputed"
