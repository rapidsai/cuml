#include "internal.cuh"
#include <cstdint>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <raft/handle.hpp>
#include <thrust/host_vector.h>

template <typename SrcT, typename ScaleT>
__device__ float convert_maybe_check(const void* in,
                                     std::size_t idx,
                                     ScaleT scale,
                                     int* lossy = nullptr)
{
  SrcT orig = reinterpret_cast<const SrcT*>(in)[idx];
  float res = static_cast<double>(orig) * scale;
  if (lossy != nullptr) {
    SrcT back_cast = static_cast<double>(res) / scale;
    if (back_cast != orig) atomicExch(lossy, 1);
  }
  return res;
}

__device__ float cast_scalar(const void* in, std::size_t idx, cudf::data_type from, int* lossy)
{
  switch (from.id()) {
    // below, we're checking for rounding. _typecast_will_lose_information() only checks for
    // out-of-range, which could only happen in FLOAT64 here.
    case cudf::type_id::INT8: return convert_maybe_check<int8_t>(in, idx, from.scale());
    case cudf::type_id::INT16: return convert_maybe_check<int16_t>(in, idx, from.scale());
    case cudf::type_id::INT32: return convert_maybe_check<int32_t>(in, idx, from.scale(), lossy);
    case cudf::type_id::INT64: return convert_maybe_check<int64_t>(in, idx, from.scale(), lossy);
    case cudf::type_id::UINT8: return convert_maybe_check<uint8_t>(in, idx, from.scale());
    case cudf::type_id::UINT16: return convert_maybe_check<uint16_t>(in, idx, from.scale());
    case cudf::type_id::UINT32: return convert_maybe_check<uint32_t>(in, idx, from.scale(), lossy);
    case cudf::type_id::UINT64: return convert_maybe_check<uint64_t>(in, idx, from.scale(), lossy);
    case cudf::type_id::FLOAT32: return convert_maybe_check<float>(in, idx, from.scale());
    case cudf::type_id::FLOAT64: return convert_maybe_check<double>(in, idx, from.scale(), lossy);
    case cudf::type_id::BOOL8: return reinterpret_cast<const int8_t*>(in)[idx] != 0 ? 1.0f : 0.0f;

    /* these 3 would work when cudf/fixed_point/fixed_point.hpp is included.
    However, back-converting to check for loss in precision creates a scaleless number.
    Checking for equality causes to rescale the original number to scale-0 (scale == 1?) instead
    of the other way. This would create false positives for loss detection.
    The best way really is to detect rounding/overflow in double precision multiplication and cast
    to double, or, alternatively, int64_t multiplication and cast to double.

    If we only check for out-of-range condition, the checks could be optimized to extracting the
    exponent of a scale-1 number.
    */
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128:  // return ((__int128_t*)in)[idx] * from.scale();

    case cudf::type_id::TIMESTAMP_DAYS:          // return ((int64_t*)in)[idx] * from.scale();
    case cudf::type_id::TIMESTAMP_SECONDS:       // return ((int64_t*)in)[idx] * from.scale();
    case cudf::type_id::TIMESTAMP_MILLISECONDS:  // return ((int64_t*)in)[idx] * from.scale();
    case cudf::type_id::TIMESTAMP_MICROSECONDS:  // return ((int64_t*)in)[idx] * from.scale();
    case cudf::type_id::TIMESTAMP_NANOSECONDS:   // return ((int64_t*)in)[idx] * from.scale();
    case cudf::type_id::DURATION_DAYS:           // return ((int64_t*)in)[idx] * from.scale();
    case cudf::type_id::DURATION_SECONDS:        // return ((int64_t*)in)[idx] * from.scale();
    case cudf::type_id::DURATION_MILLISECONDS:   // return ((int64_t*)in)[idx] * from.scale();
    case cudf::type_id::DURATION_MICROSECONDS:   // return ((int64_t*)in)[idx] * from.scale();
    case cudf::type_id::DURATION_NANOSECONDS:    // return ((int64_t*)in)[idx] * from.scale();
    case cudf::type_id::EMPTY:                   ///< Always null with no underlying data
    case cudf::type_id::DICTIONARY32:            ///< Dictionary type using int32 indices
    case cudf::type_id::STRING:                  ///< String elements
    case cudf::type_id::LIST:                    ///< List elements
    case cudf::type_id::STRUCT:                  ///< Struct elements
    case cudf::type_id::NUM_TYPE_IDS:            ///< Total number of type ids
    default:
      // bad code, return NAN for now
      return NAN;
  }
}

// cudf::column_view contains more info than we really need, and requires an extra pointer
// dereference in each GPU thread
struct device_col {
  // contains scale for fixed-point columns
  cudf::data_type dtype;
  const void* value;
  const uint8_t* valid_mask;  // bit N set to 1 if Nth value is valid (cudf convention)
};

// blocks are square tiles. If last column reached, waste the remainder of tile.
#define TILE_SIZE std::size_t(32)

// is lossy a scalar or vector (1 per column)?
__global__ void cast_and_transpose(
  float* row_major, const device_col* cols, std::size_t n_cols, std::size_t n_rows, int* lossy)
{
#define SHMEM_TILE_COLS (TILE_SIZE + std::size_t(1))
  __shared__ float shmem_row_major[TILE_SIZE * SHMEM_TILE_COLS];
  // thread index preamble
  std::size_t tile_row0 = (std::size_t)TILE_SIZE * blockIdx.x;
  std::size_t tile_col0 = (std::size_t)TILE_SIZE * blockIdx.y;

  char row1_in_tile = threadIdx.x, col1_in_tile = threadIdx.y;
  std::size_t row1 = tile_row0 + row1_in_tile;
  std::size_t col1 = tile_col0 + col1_in_tile;
  if (row1 < n_rows && col1 < n_cols) {
    device_col column = cols[col1];
    // load from gmem. check if valid and convert
    float val = cast_scalar(column.value, row1, column.dtype, lossy);
    if (column.valid_mask != nullptr && !ML::fil::fetch_bit(column.valid_mask, row1)) val = NAN;
    // shmem write index will be the non-contiguous one
    shmem_row_major[row1_in_tile * SHMEM_TILE_COLS + col1_in_tile] = val;
  }
  // move shmem -> gmem
  __syncthreads();
  // same tile, difference only within tile
  char row2_in_tile = threadIdx.y, col2_in_tile = threadIdx.x;
  std::size_t row2 = tile_row0 + row2_in_tile;
  std::size_t col2 = tile_col0 + col2_in_tile;
  if (row2 < n_rows && col2 < n_cols)
    row_major[row2 * n_cols + col2] =
      shmem_row_major[row2_in_tile * SHMEM_TILE_COLS + col2_in_tile];
}

// similar to interleave_columns()

namespace ML {

void cudf_to_row_major(const raft::handle_t& h,
                       float* row_major,
                       const std::vector<cudf::column_view>& cols)
{
  std::size_t n_cols = cols.size();
  std::size_t n_rows = cols[0].size();

  std::vector<device_col> h_cols;
  for (const cudf::column_view& cv : cols) {
    h_cols.push_back({cv.type(), cv.head(), reinterpret_cast<const uint8_t*>(cv.null_mask())});
    ASSERT(static_cast<std::size_t>(cv.size()) == n_rows,
           "internal error: creating matrix out of unequally sized columns");
  }

  CUDA_CHECK(cudaSetDevice(h.get_device()));
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();
  device_col* d_cols                  = (device_col*)mr->allocate(h_cols.size() * sizeof *d_cols);
  CUDA_CHECK(cudaMemcpyAsync(
    d_cols, h_cols.data(), h_cols.size() * sizeof *d_cols, cudaMemcpyHostToDevice, 0));

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid(raft::ceildiv(n_rows, TILE_SIZE), raft::ceildiv(n_cols, TILE_SIZE));
  cast_and_transpose<<<grid, block>>>(row_major, d_cols, h_cols.size(), n_rows, nullptr);

  mr->deallocate(d_cols, h_cols.size() * sizeof *d_cols);
}

} // namespace ML
