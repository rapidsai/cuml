#pragma once

#include <cub/cub.cuh>
#include "cuda_utils.h"
#include "decoupled_lookback.h"
#include "vectorized.h"


namespace MLCommon {
namespace Random {

template <typename IntType, int NumItems>
DI void getIndices(IntType (&idx)[NumItems], IntType a, IntType b, int k,
                   IntType n) {
    IntType tid = NumItems * (threadIdx.x + (blockIdx.x * blockDim.x));
    IntType mask = (1 << k) - 1;
#pragma unroll
    for (int i = 0; i < NumItems; ++i)
        idx[i] = ((a * (tid + i)) + b) & mask;
}

template <typename IntType, int NumItems, int TPB>
DI void compaction(const IntType (&idx)[NumItems], IntType (&outIdx)[NumItems],
                   DecoupledLookBack<int>& dlb, IntType N) {
#pragma unroll
    for (int i = 0; i < NumItems; ++i)
        outIdx[i] = idx[i] < N;
    typedef cub::BlockScan<IntType, TPB> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage).ExclusiveSum(outIdx, outIdx);
    IntType thisBlockSum = 0;
    if(threadIdx.x == blockDim.x - 1)
        thisBlockSum = outIdx[NumItems-1] + (idx[NumItems - 1] < N);
    auto prefix = dlb(thisBlockSum);
#pragma unroll
    for (int i = 0; i < NumItems; ++i)
        outIdx[i] += prefix;
}

///@todo: fix shared mem bank conflicts
template <typename IntType, int NumItems, int TPB>
DI void writePermIndices(IntType* perms, const IntType (&idx)[NumItems],
                         const IntType (&outIdx)[NumItems], IntType* s_perms,
                         IntType N, int k) {
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < NumItems; ++i) {
        int pos = tid * NumItems + i;
        s_perms[pos] = outIdx[i];
        s_perms[pos + NumItems * TPB] = idx[i];
    }
    __syncthreads();
    IntType maxVal = 1 << k;
    IntType gid = blockIdx.x * blockDim.x * NumItems;
#pragma unroll
    for (int i = 0; i < NumItems; ++i) {
        int pos = tid + i * TPB;
        auto loc = s_perms[pos];
        auto myIdx = s_perms[pos + NumItems * TPB];
        // It's possible that for smaller N values we are launching more threads
        // than needed!
        IntType offset = gid + pos;
        if(myIdx < N && offset < maxVal) perms[loc] = myIdx;
    }
}

///@todo: fix shared mem bank conflicts
///@todo: support for column-major layout
template <typename Type, typename IntType, int VecLen, int NumItems, int TPB>
DI void shuffleArray(Type *out, const Type *in, IntType *s_perms, bool rowMajor,
                     const IntType (&idx)[NumItems],
                     const IntType (&outIdx)[NumItems], IntType N, IntType D) {
    if(rowMajor) {
        typedef TxN_t<Type, VecLen> VecType;
        constexpr int WarpSize = 32;
        constexpr int nWarps = TPB / WarpSize;
        constexpr int colStride = WarpSize * VecType::Ratio;
        int warpId = threadIdx.x / WarpSize;
        int laneId = threadIdx.x % WarpSize;
        for (int row = warpId; row < TPB * NumItems; row += nWarps) {
            IntType outRowStart = s_perms[row] * D;
            IntType inRow = s_perms[row + NumItems * TPB];
            IntType inRowStart = inRow * D;
            if (inRow < N) {
                int startCol = laneId * VecType::Ratio;
                for (int col = startCol; col < D; col += colStride) {
                    VecType data;
                    data.load(in, inRowStart + col);
                    data.store(out, outRowStart + col);
                }
            }
        }
    } else {
    }
}

template <typename Type, typename IntType, int VecLen, int NumItems, int TPB>
__global__ void permuteKernel(IntType* perms, Type* out, const Type* in,
                              IntType a, IntType b, int k, IntType N, IntType D,
                              bool rowMajor, void* workspace) {
    DecoupledLookBack<IntType> dlb(workspace);
    IntType idx[NumItems], outIdx[NumItems];
    getIndices<IntType, NumItems>(idx, a, b, k, N);
    compaction<IntType, NumItems, TPB>(idx, outIdx, dlb, N);
    __shared__ IntType s_perms[2 * NumItems * TPB];
    if(perms != nullptr) {
        writePermIndices<IntType, NumItems, TPB>(
            perms, idx, outIdx, s_perms, N, k);
    }
    if(out != nullptr && in != nullptr) {
        shuffleArray<Type, IntType, VecLen, NumItems, TPB>(
            out, in, s_perms, rowMajor, idx, outIdx, N, D);
    }
}

template <typename Type, typename IntType, int VecLen, int NumItems, int TPB>
void permuteImpl(IntType* perms, Type* out, const Type* in, IntType N,
                 IntType D, bool rowMajor, void *workspace,
                 size_t workspaceSize, int nblks, IntType twoPowK, int k,
                 cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, workspaceSize, stream));
    // always keep 'a' to be coprime to 2^k
    IntType a = rand() % twoPowK;
    if(a % 2 == 0)
        a = (a + 1) % twoPowK;
    IntType b = rand() % twoPowK;
    permuteKernel<Type, IntType, VecLen, NumItems, TPB>
        <<<nblks, TPB, 0, stream>>>(perms, out, in, a, b, k, N, D, rowMajor,
                                    workspace);
    CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Generate permutations of the input array. Pretty useful primitive for
 * shuffling the input datasets in ML algos. See note at the end for some of its
 * limitations!
 * @tparam Type Data type of the array to be shuffled
 * @tparam IntType Integer type used for addressing indices
 * @tparam TPB threads per block
 * @param perms the output permutation indices. Typically useful only when
 * one wants to refer back. If you don't need this, pass a nullptr
 * @param out the output shuffled array. Pass nullptr if you don't want this to
 * be written. For eg: when you only want the perms array to be filled.
 * @param in input array (in-place is not supported due to race conditions!)
 * @param D number of columns of the input array
 * @param N length of the input array (or number of rows)
 * @param rowMajor whether the input/output matrices are row or col major
 * @param workspace temporary workspace needed for computations. If you pass
 * nullptr, then the size needed would be computed and returned in the
 * workspaceSize argument
 * @param workspaceSize if workspace is passed as nullptr, this will contain
 * the workspace size in bytes.
 * @param stream cuda stream where to launch the work
 *
 * @note This is NOT a uniform permutation generator! In fact, it only generates
 * very small percentage of permutations. If your application really requires a
 * high quality permutation generator, it is recommended that you pick
 * Knuth Shuffle.
 */
template <typename Type, typename IntType = int, int TPB = 256>
void permute(IntType* perms, Type* out, const Type* in, IntType D, IntType N,
             bool rowMajor, void *workspace, size_t& workspaceSize,
             cudaStream_t stream = 0) {
    ///@todo: support col-major layout
    ASSERT(rowMajor, "permute: Currently only rowMajor layout is supported!");
    ///@todo: figure out this number based on input matrix dimensions
    constexpr int NumItems = 4;
    // get the next highest po2 for N
    int k = (int)log2(N, (IntType)0);
    if(N > (1 << k)) ++k;
    IntType twoPowK = 1 << k;
    auto nblks = ceildiv(twoPowK, TPB * NumItems);
    if(workspace == nullptr) {
        workspaceSize = DecoupledLookBack<IntType>::computeWorkspaceSize(nblks);
        return;
    }
    size_t bytes = D * sizeof(Type);
    uint64_t inAddr = uint64_t(in);
    uint64_t outAddr = uint64_t(out);
    if (16 / sizeof(Type) && bytes % 16 == 0 && inAddr % 16 == 0 &&
        outAddr % 16 == 0) {
        permuteImpl<Type, IntType, 16 / sizeof(Type), NumItems, TPB>
            (perms, out, in, N, D, rowMajor, workspace, workspaceSize, nblks,
             twoPowK, k, stream);
    } else if (8 / sizeof(Type) && bytes % 8 == 0 && inAddr % 8 == 0 &&
               outAddr % 8 == 0) {
        permuteImpl<Type, IntType, 8 / sizeof(Type), NumItems, TPB>
            (perms, out, in, N, D, rowMajor, workspace, workspaceSize, nblks,
             twoPowK, k, stream);
    } else if (4 / sizeof(Type) && bytes % 4 == 0 && inAddr % 4 == 0 &&
               outAddr % 4 == 0) {
        permuteImpl<Type, IntType, 4 / sizeof(Type), NumItems, TPB>
            (perms, out, in, N, D, rowMajor, workspace, workspaceSize, nblks,
             twoPowK, k, stream);
    } else if (2 / sizeof(Type) && bytes % 2 == 0 && inAddr % 2 == 0 &&
               outAddr % 2 == 0) {
        permuteImpl<Type, IntType, 2 / sizeof(Type), NumItems, TPB>
            (perms, out, in, N, D, rowMajor, workspace, workspaceSize, nblks,
             twoPowK, k, stream);
    } else if (1 / sizeof(Type)) {
        permuteImpl<Type, IntType, 1 / sizeof(Type), NumItems, TPB>
            (perms, out, in, N, D, rowMajor, workspace, workspaceSize, nblks,
             twoPowK, k, stream);
    } else {
        permuteImpl<Type, IntType, 1, NumItems, TPB>
            (perms, out, in, N, D, rowMajor, workspace, workspaceSize, nblks,
             twoPowK, k, stream);
    }
}

}; // end namespace Random
}; // end namespace MLCommon
