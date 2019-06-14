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
/* Return max. possible number of columns that can be processed within avail_shared_memory.
   Expects that requested_shared_memory is a multiple of ncols. */
int get_batch_cols_cnt(const size_t avail_shared_memory, const size_t requested_shared_memory, const int ncols) {
	int ncols_in_batch = ncols;
	int ncols_factor = requested_shared_memory / ncols;
	if (requested_shared_memory > avail_shared_memory) {
		ncols_in_batch = avail_shared_memory / ncols_factor; // floor div.
	}
	return  ncols_in_batch;
}


/* Update batch_ncols (max. possible number of columns that can be processed within avail_shared_memory),
   blocks (for next kernel launch), and shmemsize (requested shared memory for next kernel launch).
   Precondition: requested_shared_memory is a multiple of ncols. */
void update_kernel_config(const size_t avail_shared_memory, const size_t requested_shared_memory, const int ncols,
               const int nrows, const int threads, int & batch_ncols, int & blocks, size_t & shmemsize) {
  batch_ncols = get_batch_cols_cnt(avail_shared_memory, requested_shared_memory, ncols);
  shmemsize = (requested_shared_memory / ncols) * batch_ncols; // requested_shared_memory is a multiple of ncols for all kernels
  blocks = min(MLCommon::ceildiv(batch_ncols * nrows, threads), 65536);
}
