/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

namespace aion {
class lapack;

class lapack {
 public:
    template <typename Dtype>
    static int geqrf(int m, int n, Dtype* a, int lda, Dtype* tau);

    template <typename Dtype>
    static int orgqr(int m, int n, int k, Dtype* a, int lda, const Dtype* tau);
};

}  // namespace aion
