#include <cuda_runtime.h>


namespace ML {
    int get_device(const void *ptr) {
        cudaPointerAttributes att;
        cudaPointerGetAttributes(&att, ptr);
        return att.device;
    }
}
