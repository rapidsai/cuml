#pragma once

#include "cuda_utils.h"
#include <limits>


namespace MLCommon {
namespace Selection {

/**
 * @brief The comparator
 * @tparam Greater whether to apply greater or lesser than comparison
 * @tparam T data type
 */
template <bool Greater, typename T>
struct Compare {
    /** compare the two input operands */
    static DI bool op(T a, T b) {
        return Greater? a > b : a < b;
    }
};


/**
 * @brief Struct to abstract compare-and-swap operation
 * @tparam TypeV value type
 * @tparam TypeK key type
 */
template <typename TypeV, typename TypeK>
struct KVPair {
    /** the value used to compare and decide for swap */
    TypeV val;
    /** key associated with the value */
    TypeK key;
    typedef KVPair<TypeV,TypeK> Pair;

    /**
     * @brief Compare and swap the current with the other pair
     * @tparam Greater when to perform a swap operation
     * @param other the other pair
     * @param small whether the comparison is being done by warp with smaller laneid
     */
    template <bool Greater>
    DI void cas(Pair& other, bool small) {
        bool swap_ = compare<Greater>(other, small);
        if(swap_)
            swap(other);
    }

    /** assign the contents of other pair to the current */
    DI void operator=(Pair& other) {
        val = other.val;
        key = other.key;
    }

    /** equality comparison */
    DI bool operator==(const Pair& other) {
        return val == other.val && key  == other.key;
    }

    /** greater than operator */
    DI bool operator>(const Pair& other) {
        ///@todo: should we also consider the key when values are the same?
        return val > other.val;
    }

    /** lesser than operator */
    DI bool operator<(const Pair& other) {
        ///@todo: should we also consider the key when values are the same?
        return val < other.val;
    }

    /**
     * @brief shuffle the current value with the src laneId
     * @param srcLane the source lane
     * @param width lane width
     * @param mask mask of participating threads (Volta+)
     * @return the shuffled value
     */
    DI Pair shfl(int srcLane, int width=WarpSize, uint32_t mask=0xffffffffu) {
        Pair ret = *this;
        ret.val = MLCommon::shfl(ret.val, srcLane, width, mask);
        ret.key = MLCommon::shfl(ret.key, srcLane, width, mask);
        return ret;
    }

    /**
     * @brief XOR-shuffle the current value with the src laneId
     * @param laneMask mask to be applied in order to get the destination lane id
     * @param width lane width
     * @param mask mask of participating threads (Volta+)
     * @return the shuffled value
     */
    DI Pair shfl_xor(int laneMask, int width=WarpSize, uint32_t mask=0xffffffffu) {
        Pair ret = *this;
        ret.val = MLCommon::shfl_xor(ret.val, laneMask, width, mask);
        ret.key = MLCommon::shfl_xor(ret.key, laneMask, width, mask);
        return ret;
    }

    /** store the data to global memory */
    DI void store(TypeV* vptr, TypeK* kptr) const {
        if(vptr != nullptr)
            *vptr = val;
        if(kptr != nullptr)
            *kptr = key;
    }

private:
    template <bool Greater>
    DI bool compare(const Pair& other, bool small) {
        return small?
            Compare<Greater,TypeV>::op(val, other.val) :
            Compare<!Greater,TypeV>::op(val, other.val);
    }

    DI void swap(Pair& other) {
        auto tmp = *this;
        *this = other;
        other = tmp;
    }
};


/**
 * @brief perform a warp-wide parallel one-pass bitonic kind of network traversal
 * @tparam TypeV value type
 * @tparam TypeK key type
 * @tparam Greater when to perform swap operation
 * @param current current thread's value
 */
template <typename TypeV, typename TypeK, bool Greater>
DI void warpSort(KVPair<TypeV,TypeK>& current) {
    int lid = laneId();
    #pragma unroll
    for(int stride=WarpSize/2;stride>=1;stride/=2) {
        bool small = !(lid & stride);
        auto other = current.shfl_xor(stride);
        current.cas<Greater>(other, small);
    }
}


/**
 * @brief Struct to abstract an array of key-val pairs.
 * It is assumed to be strided across warp. Meaning, this array is assumed to be
 * actually of length N*32, in row-major order. In other words, all of
 * arr[0] across all threads will come first, followed by arr[1] and so on.
 * @tparam TypeV value type
 * @tparam TypeK key type
 * @tparam N number of elements in the array
 * @tparam Greater whether to do a greater than comparison
 */
template <typename TypeV, typename TypeK, int N, bool Greater>
struct KVArray {
    typedef KVPair<TypeV,TypeK> Pair;
    /** the array of pairs */
    Pair arr[N];
    /** bit-mask representing all valid indices of the array */
    constexpr static int ArrMask = N - 1;
    /** mask representing all threads in a warp */
    constexpr static int WarpMask = WarpSize - 1;

    /** reset the contents of the array */
    DI void reset(TypeV iV, TypeK iK) {
        #pragma unroll
        for(int i=0;i<N;++i) {
            arr[i].val = iV;
            arr[i].key = iK;
        }
    }

    DI void topkUpdate(Pair& other) {
        #pragma unroll
        for(int i=0;i<N;++i) {
            // perform the sort in the reverse order as to minimize the
            // amount of shfl's needed during the merge phase
            warpSort<TypeV,TypeK,!Greater>(other);
            arr[i].cas<Greater>(other, true);
            warpSort<TypeV,TypeK,Greater>(arr[i]);
        }
    }

    ///@todo: this fails for N=8 onwards!!
    ///@todo: it also generates "stack frame" for N>=8
    /** sort the elements in this array */
    DI void sort() {
        // start by sorting along the warp, first
        warpWideSort();
        // iteratively merge each of these "warp-wide" sorted arrays
        #pragma unroll
        for(int stride=1;stride<N;stride*=2) {
            const int s2 = 2 * stride;
            #pragma unroll
            for(int start=0;start<N;start+=s2)
                mergeHalves(stride, start);
            #pragma unroll
            for(int start=0;start<N;start+=stride)
                postMergeSort(stride, start);
            warpWideSort();
        }
    }

private:
    DI void mergeHalves(int stride, int start) {
        const int mask = 2 * stride - 1;
        #pragma unroll
        for(int i=0;i<stride;++i) {
            int src = i + start;
            int dst = (i + start) ^ mask;
            auto srcOtherPair = arr[src].shfl_xor(WarpMask);
            auto dstOtherPair = arr[dst].shfl_xor(WarpMask);
            arr[src].cas<Greater>(dstOtherPair, true);
            arr[dst].cas<Greater>(srcOtherPair, false);
        }
    }

    DI void postMergeSort(int stride, int start) {
        #pragma unroll
        for(int s=stride/2;s>=1;s/=2) {
            #pragma unroll
            for(int j=0;j<s;++j) {
                int ij = start + j;
                arr[ij].cas<Greater>(arr[ij+s], true);
            }
        }
    }

    DI void warpWideSort() {
        #pragma unroll
        for(int i=0;i<N;++i)
            warpSort<TypeV,TypeK,Greater>(arr[i]);
    }
};

///@todo: specialize this for k=1
template <typename TypeV, typename TypeK, int N, int TPB, bool Greater, bool Sort>
__global__ void warpTopKkernel(TypeV* outV, TypeK* outK, const TypeV* arr,
                               int k, int rows, int cols, TypeV iV, TypeK iK) {
    static_assert(Sort==false, "warpTopK: Sort=true is not yet supported!");
    constexpr int RowsPerBlk = TPB / WarpSize;
    const int warpId = threadIdx.x / WarpSize;
    const int rowId = blockIdx.x * RowsPerBlk + warpId;
    if(rowId >= rows)
        return;
    const int maxCols = alignTo(cols, WarpSize);
    KVArray<TypeV,TypeK,N,Greater> topk;
    KVPair<TypeV,TypeK> other;
    topk.reset(iV, iK);
    int colId = threadIdx.x;
    for(;colId<maxCols;colId+=WarpSize) {
        auto idx = rowId * cols + colId;
        other.val = colId < cols? arr[idx] : iV;
        other.key = idx;
        warpFence();
        topk.topkUpdate(other);
    }
    int lid = laneId();
    #pragma unroll
    for(int i=0;i<N;++i) {
        int col = i * WarpSize + lid;
        if(outV != nullptr && col < k)
            outV[rowId*k+col] = topk.arr[i].val;
        if(outK != nullptr && col < k)
            outK[rowId*k+col] = topk.arr[i].key;
    }
}

#define CASE_K(kval)                                                    \
    case kval:                                                          \
        warpTopKkernel<TypeV,TypeK,kval,TPB,Greater,Sort>               \
            <<<nblks,TPB>>>(outV, outK, arr, k, rows, cols, iV, iK);    \
        break
/**
 * @brief Perform warp-wide top-k selection on the input matrix
 * @tparam TypeV value type
 * @tparam TypeK key type
 * @tparam Greater whether to do a greater than comparison
 * @tparam Sort whether to sort the final topK values before writing
 * @note the input matrix is assumed to be row-major!
 * @todo verify and extend support to k <= 1024
 */
template <typename TypeV, typename TypeK, bool Greater, bool Sort>
void warpTopK(TypeV* outV, TypeK* outK, const TypeV* arr,
              int k, int rows, int cols) {
    constexpr int TPB = 256;
    constexpr int RowsPerBlk = TPB / WarpSize;
    const int nblks = ceildiv(rows, RowsPerBlk);
    const int kAligned = alignTo(k, WarpSize) / WarpSize;
    const TypeV iV = Greater? std::numeric_limits<TypeV>::max() :
        std::numeric_limits<TypeV>::min();
    const TypeK iK = Greater? std::numeric_limits<TypeK>::max() :
        std::numeric_limits<TypeK>::min();
    switch(kAligned) {
        CASE_K(1);
        CASE_K(2);
        CASE_K(3);
        CASE_K(4);
    default:
        ASSERT(false, "TopK kernels only support k <= 128 [%d]", k);
    };
}
#undef CASE_K

}; // end namespace Selection
}; // end namespace MLCommon
