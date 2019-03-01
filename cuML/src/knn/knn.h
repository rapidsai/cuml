
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>

#pragma once

namespace ML {

    enum IndexType { flat, ivfpq };

	class kNNParams {
	public:
		const float *ptr;
		int N;
		friend std::ostream & operator<<(std::ostream &str, kNNParams &v) {
			str << "kNNParams {ptr=" << v.ptr << ", N=" << v.N << "}";
			return str;
		}
	};

    class kNN {

		std::vector<long> id_ranges;

		std::vector<faiss::gpu::GpuResources* > res;
		std::vector<faiss::gpu::GpuIndex* > sub_indices;

		int total_n;
		int indices;
		int D;

    private:
		template <class C>
		void merge_tables(long n, long k, long nshard,
							   float *distances, long *labels,
							   float *all_distances,
							   long *all_labels,
							   long *translations);

    public:
		kNN(int D);
		~kNN();
		void search(const float *search_items, int search_items_size, long *res_I, float *res_D, int k);
		void fit(kNNParams *input, int N);

    };
}

