/*
 * knn_multi.h
 *
 *  Created on: Jan 28, 2019
 *      Author: cjnolet
 */

#ifndef KNN_MULTI_H_
#define KNN_MULTI_H_

#include "cuda_utils.h"
#include "knn_c.h"
#include <mpi.h>
#include <faiss/Heap.h>


namespace ML {
void search_MGMN(kNN *knn, const float *search_items, int n, long *res_I, float *res_D, int k,
					int* ranks, int n_ranks) {

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	MPI_Group prime_group;
	MPI_Group_incl(world_group, n_ranks, ranks, &prime_group);

	MPI_Comm prime_comm;
	MPI_Comm_create_group(MPI_COMM_WORLD, prime_group, 0, &prime_comm);

	std::cout << "Rank: " << rank << std::endl;

	// perform local search
	float *result_D = new float[k*n];
	long *result_I = new long[k*n];

	knn->search(search_items, n, result_I, result_D, k);

	int group_size;			// Will this ever be different than n_ranks?
	int root = ranks[0];	// Always use smallest rank as the root

	float *D_buf;
	long *I_buf;	// buffer of data received from all the ranks (stored in rank order!)
	int *idx_buf;

	MPI_Comm_size(prime_comm, &group_size);

	if(rank == ranks[0]) {
		D_buf = (float*)malloc(group_size*n*k*sizeof(float));
		I_buf = (long*)malloc(group_size*n*k*sizeof(long));
		idx_buf = (int*)malloc(group_size*sizeof(int));
	}


	int size_buf = knn->get_index_size();


	// We know how big buffers need to be so we could just do 3 gathers: D, I, & idx_size
	MPI_Gather(result_I, n*k, MPI_LONG, I_buf, n*k, MPI_LONG, root, prime_comm);
	MPI_Gather(result_D, n*k, MPI_FLOAT, D_buf, n*k, MPI_FLOAT, root, prime_comm);
	MPI_Gather(&size_buf, 1, MPI_INT, idx_buf, n_ranks, MPI_INT, root, prime_comm);

	std::vector<long> rank_id_ranges;
	rank_id_ranges.push_back(0);
	for(int i = 1; i < n_ranks; i++)
		rank_id_ranges.push_back(idx_buf[i]+idx_buf[i-1]);

	// Perform reduce
	knn->merge_tables<faiss::CMin<float, int>>(n, k, n_ranks,
			result_D, result_I, D_buf, I_buf, rank_id_ranges.data());

	// copy result to res_I and res_D
	MLCommon::updateDevice(res_D, result_D, k*n, 0);
	MLCommon::updateDevice(res_I, result_I, k*n, 0);

	delete D_buf;
	delete I_buf;
	delete idx_buf;

	delete result_D;
	delete result_I;
}

};



#endif /* KNN_MULTI_H_ */
