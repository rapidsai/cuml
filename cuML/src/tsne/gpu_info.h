
#include "cuda_utils.h"

// From Random/rng.h
// Gets CUDA device details [core count, warp size etc]
namespace GPU_Info_ {

void gpuInfo(	int * __restrict__ BLOCKS,
				int * __restrict__ TPB_X,
			    int * __restrict__ integration_kernel_threads,
			    int * __restrict__ integration_kernel_factor,
			    int * __restrict__ repulsive_kernel_threads,
			    int * __restrict__ repulsive_kernel_factor,
			    int * __restrict__ bounding_kernel_threads,
			    int * __restrict__ bounding_kernel_factor, 
			    int * __restrict__ tree_kernel_threads,
			    int * __restrict__ tree_kernel_factor,
			    int * __restrict__ sort_kernel_threads,
			    int * __restrict__ sort_kernel_factor,
			    int * __restrict__ summary_kernel_threads,
			    int * __restrict__ summary_kernel_factor)
{
	int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp DEVICE;

    CUDA_CHECK(cudaGetDeviceProperties(&DEVICE, dev));
    *BLOCKS = DEVICE.multiProcessorCount;

    const int GPU_TYPE = DEVICE.major;

    // Only 32 TPB_X is supported
    *TPB_X = DEVICE.warpSize;
    assert(*TPB_X == 32);


    // From cannylab/options.h
    if (GPU_TYPE >= 5) {  // MAXWELL
	    *integration_kernel_threads    = 1024;
	    *integration_kernel_factor     = 1;
	    *repulsive_kernel_threads      = 256;
	    *repulsive_kernel_factor       = 5;
	    *bounding_kernel_threads       = 512;
	    *bounding_kernel_factor        = 3;
	    *tree_kernel_threads           = 1024;
	    *tree_kernel_factor            = 2;
	    *sort_kernel_threads           = 64;
	    *sort_kernel_factor            = 6;
	    *summary_kernel_threads        = 128;
	    *summary_kernel_factor         = 6;
	}
	else if (GPU_TYPE >= 3) {  // KEPLER
	    *integration_kernel_threads    = 1024;
	    *integration_kernel_factor     = 2;
	    *repulsive_kernel_threads      = 1024;
	    *repulsive_kernel_factor       = 2;
	    *bounding_kernel_threads       = 1024;
	    *bounding_kernel_factor        = 2;
	    *tree_kernel_threads           = 1024;
	    *tree_kernel_factor            = 2;
	    *sort_kernel_threads           = 128;
	    *sort_kernel_factor            = 4;
	    *summary_kernel_threads        = 768;
	    *summary_kernel_factor         = 1;
	}
	else {  // DEFAULT
	    *integration_kernel_threads    = 512;
	    *integration_kernel_factor     = 3;
	    *repulsive_kernel_threads      = 256;
	    *repulsive_kernel_factor       = 5;
	    *bounding_kernel_threads       = 512;
	    *bounding_kernel_factor        = 3;
	    *tree_kernel_threads           = 512;
	    *tree_kernel_factor            = 3;
	    *sort_kernel_threads           = 64;
	    *sort_kernel_factor            = 6;
	    *summary_kernel_threads        = 128;
	    *summary_kernel_factor         = 6;
	}

}

// end namespace
}