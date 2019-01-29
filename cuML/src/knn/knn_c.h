
#include <sstream>
#include <iostream>

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>

#ifndef _KNN_H
#define _KNN_H
namespace ML {

	using namespace faiss;

	class kNNParams {
	public:
		float *ptr;
		int N;
		friend std::ostream & operator<<(std::ostream &str, kNNParams &v) {
			str << "kNNParams {ptr=" << v.ptr << ", N=" << v.N << "}";
			return str;
		}
	};

	class MPI_Search_payload {
	public:
		float *d;
		float *i;
		int d_len;
		int i_len;
		int idx_size;
	};

    class kNN {

		std::vector<long> id_ranges;

		std::vector<faiss::gpu::GpuResources* > res;
		std::vector<faiss::gpu::GpuIndexFlatL2* > sub_indices;

		int total_n;
		int indices;
		int D;


    public:
		kNN(int D);
		~kNN();
		void search(const float *search_items, int search_items_size, long *res_I, float *res_D, int k);
		void fit(kNNParams *input, int N);
		void search_mn(const float *search_items, int n, long *res_I, float *res_D, int k, int* ranks, int n_ranks);

		int get_index_size();

		template <class C>
		void merge_tables(long n, long k, long nshard,
							   float *distances, long *labels,
							   float *all_distances,
							   long *all_labels,
							   long *translations);


    };
}

#endif
