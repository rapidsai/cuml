template <typename T>
__device__
T sum(T array, int len){
        T sum = 0;
        for (size_t i = 0; i < len; i++) {
                sum += array[i];
        }
        return sum;
}

template <typename T>
__device__
void elementwise(T out, T in_a, T in_b, int len){
        for (size_t i = 0; i < len; i++) {
                out[i] = in_a[i] * in_b[i];
        }
}

template <typename T>
__device__
int arg_max(T array, int len){
        T maxVal = array[0];
        int max_idx = 0;
        for (size_t i = 0; i < len; i++) {
                if (array[i] > maxVal) {
                        maxVal = array[i];
                        max_idx = i;
                }
        }
        return max_idx;
}

// template <typename T>
// __global__
// void oneHotKernel(T* dO, int lddO, int* dIdxArray, int len_array,
//                   int nThreads_x, int nThreads_y){
//         int i_start = threadIdx.x + blockDim.x * blockIdx.x;
//         int j_start = threadIdx.y + blockDim.y * blockIdx.y;
//
//         for (size_t tau = j_start; tau < len_array; tau+=nThreads_y) {
//                 for (size_t stateId = i_start; stateId < nStates; stateId+=nThreads_x) {
//                         if (stateId == dIdxArray[tau]) {
//                                 dO[IDX(stateId, tau, lddO)] = 1.;
//                         }
//                         else{
//                                 dO[IDX(stateId, tau, lddO)] = 0;
//                         }
//                 }
//         }
// }
//
// template <typename T>
// void _one_hot(T* dO, int lddO, int* dIdxArray, int len_array){
//
//
//         dim3 block(32,32);
//         dim3 grid(ceildiv(lddo, (int)block.x),
//                   ceildiv(len_array, (int)block.y),
//                   1);
//         int nThreads_x = grid.x * block.x;
//         int nThreads_y = grid.y * block.y;
//
//         oneHotKernel<T> <<< grid, block >>>(dO, lddO, dIdxArray, len_array, nThreads_x, nThreads_y);
//         cudaDeviceSynchronize();
//         CUDA_CHECK(cudaPeekAtLastError());
//
// }

template <typename T>
__device__
T _prod(T x, T y){
        return std::exp(std::log(x) + std::log(y));
}
