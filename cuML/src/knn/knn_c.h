#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>

namespace ML {


    class kNN {
        
        faiss::gpu::GpuIndexFlatL2 index_flat;
        faiss::gpu::StandardGpuResources res;
        kNN(int D);
        ~kNN();
        void search(float *search_items, int search_items_size, long *res_I, float *res_D, int k);
	void fit(float *input, int N);

    };
}
