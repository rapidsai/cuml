/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <utils.h>
#include <vector>

template <class T>
struct GiniQuestion {
  int bootstrapped_column;
  int original_column;
  T value;

  /*
	   delta = (max - min) /nbins
	   base_ques_val = min + delta
	   value = base_ques_val + batch_id * delta.

	We need to ensure the question value is always computed on the GPU. Otherwise, the flag_kernel
	called via make_split would make a split that'd be inconsistent with the one that
	produced the histograms during the gini computation. This issue arises when there is
	a data value close to the question that gets split differently in gini than in
	flag_kernel.
	*/
  int batch_id;
  T min, max;
  int nbins;
  int ncols;

  void set_question_fields(int cfg_bootcolumn, int cfg_column, int cfg_batch_id,
                           int cfg_nbins, int cfg_ncols, T cfg_min, T cfg_max,
                           T cfg_value);
};

struct GiniInfo {
  float best_gini = -1.0f;
  std::vector<int>
    hist;  //Element hist[i] stores # labels with label i for a given node.
};
