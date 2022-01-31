#include <cudf/types.hpp>
#include <cstdint>
#include "internal.cuh"

template<typename SrcT>
__device__ float convert_maybe_check(SrcT* in, std::size_t idx, cudf::scale_type scale, int* lossy = nullptr) {
  SrcT orig = in[idx];
  float res = static_cast<double>(orig) * scale;
  if(lossy != nullptr) {
    SrcT back_cast = static_cast<double>(res) / scale;
    if(back_cast != orig) atomicExch(lossy, 1);
  }
  return res;
}

__device__ float cast_scalar(void* in, std::size_t idx, cudf::data_type from, int* lossy) {
  float val = 0.0f;
  switch(from.id()) {
    // below, we're checking for rounding. _typecast_will_lose_information() only checks for out-of-range,
    // which could only happen in FLOAT64 here.
    case INT8:                   return convert_maybe_check<int8_t>(in, idx, from.scale());
    case INT16:                  return convert_maybe_check<int16_t>(in, idx, from.scale());
    case INT32:                  return convert_maybe_check<int32_t>(in, idx, from.scale(), lossy);
    case INT64:                  return convert_maybe_check<int64_t>(in, idx, from.scale(), lossy);
    case UINT8:                  return convert_maybe_check<uint8_t>(in, idx, from.scale());
    case UINT16:                 return convert_maybe_check<uint16_t>(in, idx, from.scale());
    case UINT32:                 return convert_maybe_check<uint32_t>(in, idx, from.scale(), lossy);
    case UINT64:                 return convert_maybe_check<uint64_t>(in, idx, from.scale(), lossy);
    case FLOAT32:                return convert_maybe_check<float>(in, idx, from.scale());
    case FLOAT64:                return convert_maybe_check<double>(in, idx, from.scale(), lossy);
    case BOOL8:                  return reinterpret_cast<int8_t*>(in)[idx] != 0 ? 1.0f : 0.0f;
    
    /* these 3 would work when cudf/fixed_point/fixed_point.hpp is included.
    However, back-converting to check for loss in precision creates a scaleless number.
    Checking for equality causes to rescale the original number to scale-0 (scale == 1?) instead
    of the other way. This would create false positives for loss detection.
    The best way really is to detect rounding/overflow in double precision multiplication and cast to double,
    or, alternatively, int64_t multiplication and cast to double.

    If we only check for out-of-range condition, the checks could be optimized to extracting the
    exponent of a scale-1 number.
    */
    case DECIMAL32:
    case DECIMAL64:
    case DECIMAL128:             //return ((__int128_t*)in)[idx] * from.scale();

    case TIMESTAMP_DAYS:         //return ((int64_t*)in)[idx] * from.scale();
    case TIMESTAMP_SECONDS:      //return ((int64_t*)in)[idx] * from.scale();
    case TIMESTAMP_MILLISECONDS: //return ((int64_t*)in)[idx] * from.scale();
    case TIMESTAMP_MICROSECONDS: //return ((int64_t*)in)[idx] * from.scale();
    case TIMESTAMP_NANOSECONDS:  //return ((int64_t*)in)[idx] * from.scale();
    case DURATION_DAYS:          //return ((int64_t*)in)[idx] * from.scale();
    case DURATION_SECONDS:       //return ((int64_t*)in)[idx] * from.scale();
    case DURATION_MILLISECONDS:  //return ((int64_t*)in)[idx] * from.scale();
    case DURATION_MICROSECONDS:  //return ((int64_t*)in)[idx] * from.scale();
    case DURATION_NANOSECONDS:   //return ((int64_t*)in)[idx] * from.scale();
    case EMPTY:                   ///< Always null with no underlying data
    case DICTIONARY32:            ///< Dictionary type using int32 indices
    case STRING:                  ///< String elements
    case LIST:                    ///< List elements
    case STRUCT:                  ///< Struct elements
    case NUM_TYPE_IDS:  ///< Total number of type ids
    default:
      // bad code, return NAN for now
      return NAN;
  }
}

// cudf::column_view contains more info than we really need, and requires an extra pointer dereference in each GPU thread
struct device_col {
  // contains scale for fixed-point columns
  cudf::data_type dtype;
  const void *value;
  const uint8_t *valid_mask; // bit N set to 1 if Nth value is valid (cudf convention)
};

// blocks are square tiles. If last column reached, waste the remainder of tile.
#define TILE_SIZE 32

// is lossy a scalar or vector (1 per column)?
__global__ void cast_and_transpose(float* row_major, const device_col* cols, std::size_t n_cols, std::size_t n_rows, int* lossy) {
  #define SHMEM_TILE_COLS = (TILE_SIZE + 1)
  __shared__ float shmem_row_major[TILE_SIZE * SHMEM_TILE_COLS];
  // thread index preamble
  std::size_t tile_row0 = (std::size_t)TILE_SIZE * blockIdx.x;
  std::size_t tile_col0 = (std::size_t)TILE_SIZE * blockIdx.y;

  char row1_in_tile = threadIdx.x, col1_in_tile = threadIdx.y;
  std::size_t row1 = tile_row0 + row1_in_tile;
  std::size_t col1 = tile_col0 + col1_in_tile;
  if(row1 < n_rows && col1 < n_cols) {
    device_col column = cols[col1];
    // load from gmem. check if valid and convert
    float val = cast_scalar(column.value, row1, column.dtype, lossy);
    if(column.valid_mask != nullptr && !fetch_bit(column.valid_mask, row)) val = NAN;
    // shmem write index will be the non-contiguous one
    shmem_row_major[row1_in_tile * SHMEM_TILE_COLS + col1_in_tile] = val;
  }
  // move shmem -> gmem
  __syncthreads();
  // same tile, difference only within tile
  char row2_in_tile = threadIdx.y, col2_in_tile = threadIdx.x;
  std::size_t row2 = tile_row0 + row2_in_tile;
  std::size_t col2 = tile_col0 + col2_in_tile;
  if(row2 < n_rows && col2 < n_cols)
    row_major[row2 * n_cols + col2] = shmem_row_major[row2_in_tile * SHMEM_TILE_COLS + col2_in_tile];
}

// similar to interleave_columns()

void cudf_to_row_major(const raft::handle_t& h, float** row_major, std::size_t* n_cols, std::size_t* n_rows,
                       const std::vector<column_view>& cols) {
  *n_cols = cols.size();
  *n_rows = cols[0].size();

  thrust::host_vector<device_col> h_cols;
  for(column_view& cv : cols)
    h_cols.emplace_back(cv.dtype, cv.data, cv.mask);
  
  CUDA_CHECK(cudaSetDevice(h.get_device()));
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();
  rmm::device_uvector d_cols = h_cols;
  *row_major = mr.allocate(*n_cols * *n_rows * sizeof(float));
  
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid(div_up(*n_rows, TILE_SIZE), div_up(*n_cols, TILE_SIZE));
  cast_and_transpose<<<grid, block>>>(*row_major, d_cols.data(), d_cols.size(), *n_rows, nullptr);
}
