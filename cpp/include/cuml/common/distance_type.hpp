/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

namespace ML::distance {

enum class DistanceType {
  L2Expanded          = 0,
  L2SqrtExpanded      = 1,
  CosineExpanded      = 2,
  L1                  = 3,
  L2Unexpanded        = 4,
  L2SqrtUnexpanded    = 5,
  InnerProduct        = 6,
  Linf                = 7,
  Canberra            = 8,
  LpUnexpanded        = 9,
  CorrelationExpanded = 10,
  JaccardExpanded     = 11,
  HellingerExpanded   = 12,
  Haversine           = 13,
  BrayCurtis          = 14,
  JensenShannon       = 15,
  HammingUnexpanded   = 16,
  KLDivergence        = 17,
  RusselRaoExpanded   = 18,
  DiceExpanded        = 19,
  BitwiseHamming      = 20,
  Precomputed         = 100
};

}  // end namespace ML::distance
