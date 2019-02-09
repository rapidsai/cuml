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

#include "cuda_utils.h"
#include "knn.h"
#include <cuda_runtime.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/Heap.h>

#include <mpi.h>

#include <map>
#include <vector>
#include <sstream>


namespace ML {


	/**
	 * Build a kNN object for training and querying a k-nearest neighbors model.
	 * @param D 	number of features in each vector
	 */
	kNN::kNN(int D): D(D), total_n(0), indices(0){}
	kNN::~kNN() {

		for(faiss::gpu::GpuIndexFlatL2* idx : sub_indices) { delete idx; }
		for(faiss::gpu::GpuResources *r : res) { delete r; }
	}

	/**
	 * Fit a kNN model by creating separate indices for multiple given
	 * instances of kNNParams.
	 * @param input  an array of pointers to data on (possibly different) devices
	 * @param N 	 number of items in input array.
	 */
	void kNN::fit(kNNParams *input, int N) {

		int nDevices;
		cudaGetDeviceCount(&nDevices);

     	std::map<int, std::vector<kNNParams*>*> m;
     	for(int i = 0; i < N; i++) {

			kNNParams *params = &input[i];
			cudaPointerAttributes att;
			cudaError_t err = cudaPointerGetAttributes(&att, params->ptr);

			if(err == 0 && att.device > -1) {
				auto vec = m.find(att.device);
				if(vec == m.end()) {
					std::vector<kNNParams*> *new_vec = new std::vector<kNNParams*>();
					new_vec->push_back(params);
					m.insert(std::make_pair(att.device, new_vec));
				} else {
					vec->second->push_back(params);
				}
			} else {

				std::stringstream ss;
				ss << "Input memory for " << &params << " failed. isDevice?=" << att.devicePointer;
				throw ss.str();
			}
		}

		std::map<int, std::vector<kNNParams*>*>::iterator it = m.begin();
		while(it != m.end()) {

			id_ranges.push_back(total_n);
			auto *gpures = new faiss::gpu::StandardGpuResources();
			res.emplace_back(gpures);

			faiss::gpu::GpuIndexFlatConfig config;
			config.device = it->first;
			config.useFloat16 = false;
			config.storeTransposed = false;

			auto *idx = new faiss::gpu::GpuIndexFlatL2(gpures, D, config);
			sub_indices.emplace_back(idx);
			this->indices += 1;

			std::vector<kNNParams*>::iterator vit = it->second->begin();
			while(vit != it->second->end()) {
				idx->add((*vit)->N, (*vit)->ptr);
				this->total_n += (*vit)->N;
				++vit;
			}
			++it;


		}



		std::cout << "Fit " << this->total_n << " items in " << this->indices << " indices" << std::endl;
	}

	/**
	 * Multi-GPU search for the k-nearest neighbors of a set of query vectors
	 * @param search_items set of vectors to query for neighbors
	 * @param n 		   number of items in search_items
	 * @param res_I 	   pointer to device memory for returning k nearest indices
	 * @param res_D		   pointer to device memory for returning k nearest distances
	 * @param k			   number of neighbors to query
	 */
	void kNN::search(const float *search_items, int n, long *res_I, float *res_D, int k) {
		float *result_D = new float[k*n];
		long *result_I = new long[k*n];

		float *all_D = new float[indices*k*n];
		long *all_I = new long[indices*k*n];

        for(int i = 0; i < indices; i++)
			this->sub_indices[i]->search(n, search_items, k,
					all_D+(i*k*n), all_I+(i*k*n));

		merge_tables<faiss::CMin<float, int>>(n, k, indices,
				result_D, result_I, all_D, all_I, id_ranges.data());

		MLCommon::updateDevice(res_D, result_D, k*n, 0);
		MLCommon::updateDevice(res_I, result_I, k*n, 0);

		delete all_D;
		delete all_I;

		delete result_D;
		delete result_I;
	}


	int kNN::get_index_size() { return this->total_n; }

	/**
	 * Multi-node multi-GPU search for the k-nearest neighbors of a set of query vectors.
	 * One rank from each physical node will perform their own knn::search() and send
	 * their results to the reduce node.
	 *
	 * The reduce rank will be the first node in the "ranks" argument. It will be responsible
	 * for receiving the knn results from all other participating ranks and reducing to the
	 * final set of indices and distances. Only the reduce rank will return a value.
	 */
	void kNN::search_mn(const float *search_items, int n, long *res_I, float *res_D, int k,
						int* ranks, int n_ranks) {

		std::cout << "Inside search_mn" << std::endl;

		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		MPI_Group world_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		MPI_Group prime_group;
		MPI_Group_incl(world_group, n_ranks, ranks, &prime_group);

		MPI_Comm prime_comm;
		MPI_Comm_create_group(MPI_COMM_WORLD, prime_group, 0, &prime_comm);

		std::cout << "C++ Rank: " << rank << std::endl;

		// perform local search
		float *result_D, *tmp_D;
		long *result_I, *tmp_I;

		MLCommon::allocate(result_D, n*k);
		MLCommon::allocate(result_I, n*k);

		tmp_D = (float*)malloc(n*k*sizeof(float));
		tmp_I = (long*)malloc(n*k*sizeof(long));

		this->search(search_items, n, result_I, result_D, k);

		MLCommon::updateHost(tmp_D, result_D, n*k);
		MLCommon::updateHost(tmp_I, result_I, n*k);

		int group_size;			// Will this ever be different than n_ranks?
		int root = ranks[0];	// Always use smallest rank as the root

		float *D_buf;
		long *I_buf;	// buffer of data received from all the ranks (stored in rank order!)
		int *idx_buf;

		MPI_Comm_size(prime_comm, &group_size);

		std::cout << "Group size: " << group_size << std::endl;

		// Only allocate buffers if we are the root rank
		if(rank == ranks[0]) {
			std::cout << "Allocating buffers..." << std::endl;
			I_buf = (long*)malloc(group_size*n*k*sizeof(long));
			D_buf = (float*)malloc(group_size*n*k*sizeof(float));
			idx_buf = (int*)malloc(group_size*sizeof(int));

			std::cout << "Done." << std::endl;
		}

		int size_buf = this->get_index_size();

		// We know how big buffers need to be so we could just do 3 gathers: D, I, & idx_size
		MPI_Gather(tmp_I, n*k, MPI_LONG, I_buf, n*k, MPI_LONG, root, prime_comm);
		MPI_Gather(tmp_D, n*k, MPI_FLOAT, D_buf, n*k, MPI_FLOAT, root, prime_comm);
		MPI_Gather(&size_buf, 1, MPI_INT, idx_buf, 1, MPI_INT, root, prime_comm);

		std::vector<long> rank_id_ranges;
		std::cout << "Rank ID Ranges" << std::endl;
		rank_id_ranges.push_back(0);
		for(int i = 1; i < group_size; i++) {
			std::cout << "Inside" << std::endl;
			std::cout << idx_buf[i]+idx_buf[i-1] << ", ";
			rank_id_ranges.push_back(idx_buf[i]+idx_buf[i-1]);
		}

		std::cout << "Done" << std::endl;

		long *final_I = new long[n*k];
		float *final_D = new float[n*k];

		std::cout << "Running merge tables" << std::endl;
		this->merge_tables<faiss::CMin<float, int>>(n, k, n_ranks,
				final_D, final_I, D_buf, I_buf, rank_id_ranges.data());

		std::cout << "Done." << std::endl;

		std::cout << "Copying results." << std::endl;
		// copy result to res_I and res_D
		MLCommon::updateDevice(res_I, result_I, n*k);
		MLCommon::updateDevice(res_D, result_D,  n*k);

		if(rank == ranks[0]) {
			delete D_buf;
			delete I_buf;
			delete idx_buf;
		}

		CUDA_CHECK(cudaFree(result_I));
		CUDA_CHECK(cudaFree(result_D));

		delete final_I;
		delete final_D;

		MPI_Group_free(&world_group);
		MPI_Group_free(&prime_group);
		MPI_Comm_free(&prime_comm);
	}



	/** Merge results from several shards into a single result set.
	 * @param n
	 * @param k
	 * @param nshard
	 * @param out_distances
	 * @param out_labels
	 * @param all_distances  size nshard * n * k
	 * @param all_labels     size nshard * n * k
	 * @param translations  label translations to apply, size nshard
	 */
	template <class C>
	void kNN::merge_tables (long n, long k, long nshard,
					   float *out_distances, long *out_labels,
					   float *all_distances,
					   long *all_labels,
					   long *translations) {
		if(k == 0) {
			return;
		}

		long stride = n * k;
		#pragma omp parallel
		{
			std::vector<int> buf (2 * nshard);
			int * pointer = buf.data();
			int * shard_ids = pointer + nshard;
			std::vector<float> buf2 (nshard);
			float * heap_vals = buf2.data();
			#pragma omp for
			for (long i = 0; i < n; i++) {
				// the heap maps values to the shard where they are
				// produced.
				const float *D_in = all_distances + i * k;
				const long *I_in = all_labels + i * k;
				int heap_size = 0;

				for (long s = 0; s < nshard; s++) {
					pointer[s] = 0;
					if (I_in[stride * s] >= 0)
						heap_push<C> (++heap_size, heap_vals, shard_ids,
									 D_in[stride * s], s);
				}

				float *D = out_distances + i * k;
				long *I = out_labels + i * k;

				for (int j = 0; j < k; j++) {
					if (heap_size == 0) {
						I[j] = -1;
						D[j] = C::neutral();
					} else {
						// pop best element
						int s = shard_ids[0];
						int & p = pointer[s];
						D[j] = heap_vals[0];
						I[j] = I_in[stride * s + p] + translations[s];

						heap_pop<C> (heap_size--, heap_vals, shard_ids);
						p++;
						if (p < k && I_in[stride * s + p] >= 0)
							heap_push<C> (++heap_size, heap_vals, shard_ids,
										 D_in[stride * s + p], s);
					}
				}
			}
		}

	};

};


// end namespace ML
