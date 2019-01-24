
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/IndexProxy.h>
#include <iostream>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/MetaIndexes.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/Heap.h>
#include <faiss/FaissException.h>
#include <faiss/MetaIndexes.h>

#ifndef _KNN_H
#define _KNN_H
namespace ML {

	using namespace faiss;

	struct kNNParams {
		float *ptr;
		int N;
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
			void search(float *search_items, int search_items_size, long *res_I, float *res_D, int k);
			void fit(kNNParams *input, int N);

    };


    template<typename Type>
    void updateDevice(Type* dPtr, const Type* hPtr, size_t len,
    		cudaStream_t stream = 0) {
    			cudaMemcpy(dPtr, hPtr, len * sizeof(Type), cudaMemcpyHostToDevice);
    }

    /** @} */

    	/** merge result tables from several shards.
    	 * @param all_distances  size nshard * n * k
    	 * @param all_labels     idem
    	 * @param translartions  label translations to apply, size nshard
    	 */

    	template <class C>
    	void merge_tables (long n, long k, long nshard,
    	                   float *distances, long *labels,
    	                   float *all_distances,
    	                   long *all_labels,
    	                   long *translations)
    	{
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

    	            float *D = distances + i * k;
    	            long *I = labels + i * k;

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
    	                    std::cout << "I_in=" << I[j] << " ";

    	                    heap_pop<C> (heap_size--, heap_vals, shard_ids);
    	                    p++;
    	                    if (p < k && I_in[stride * s + p] >= 0)
    	                        heap_push<C> (++heap_size, heap_vals, shard_ids,
    	                                     D_in[stride * s + p], s);
    	                }
    	            }

    	            std::cout << std::endl;
    	        }
    	    }
    	}
}



#endif
