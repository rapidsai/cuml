
#include <faiss/gpu/GpuIndexFlatL2.h>

namespace ML {

    using namespace MLCommon;

    class kNN {
        
        faiss:gpu::GpuIndexFlatL2 index_flat;
        kNN();
        ~kNN();
        void search(float *search_items, int search_items_size, long *res_I, float *res_D, int k);
	void fit(float *input, int N, int D);

    };
}
