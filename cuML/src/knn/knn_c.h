
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/IndexProxy.h>
#include <iostream>
#include <faiss/gpu/GpuIndexFlat.h>

#ifndef _KNN_H
#define _KNN_H
namespace ML {


    class kNN {

    	faiss::gpu::IndexProxy indexProxy;
     	int D;

    public:
			kNN(int D);
			~kNN();
			void search(float *search_items, int search_items_size, long *res_I, float *res_D, int k);
			void fit(float *input, int N, int n_gpus);

    };
}

#endif
