/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/prims/opg/comm_utils.h>
#include <cuml/prims/opg/matrix/math.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/divide.cuh>
#include <raft/linalg/matrix_vector.cuh>

namespace MLCommon {
namespace Matrix {
namespace opg {

template <bool rowMajor, bool bcastAlongRows, typename T, int TPB = 256>
void matrixVectorBinaryDivSkipZero_impl(std::vector<Matrix::Data<T>*>& data,
                                        const Matrix::PartDescriptor& dataDesc,
                                        const Matrix::Data<T>& vec,
                                        bool return_zero,
                                        const raft::comms::comms_t& comm,
                                        cudaStream_t* streams,
                                        int n_streams)
{
  int rank = comm.get_rank();

  std::vector<Matrix::RankSizePair*> localBlocks = dataDesc.blocksOwnedBy(rank);

  using layout_t       = std::conditional_t<rowMajor, raft::row_major, raft::col_major>;
  constexpr auto apply = bcastAlongRows ? raft::Apply::ALONG_ROWS : raft::Apply::ALONG_COLUMNS;

  for (size_t i = 0; i < localBlocks.size(); i++) {
    auto n_rows = static_cast<int>(localBlocks[i]->size);
    auto n_cols = static_cast<int>(dataDesc.N);

    auto matrix_view =
      raft::make_device_matrix_view<T, int, layout_t>(data[i]->ptr, n_rows, n_cols);

    auto vec_size = bcastAlongRows ? n_cols : n_rows;
    auto vec_view = raft::make_device_vector_view<const T, int>(vec.ptr, vec_size);

    raft::resources handle;
    raft::resource::set_cuda_stream(handle, streams[i]);

    raft::linalg::binary_div_skip_zero<apply>(handle, matrix_view, vec_view, return_zero);
  }

  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamSynchronize(streams[i]));
  }
}

template <bool rowMajor, bool bcastAlongRows, typename T, int TPB = 256>
void matrixVectorBinaryMult_impl(std::vector<Matrix::Data<T>*>& data,
                                 const Matrix::PartDescriptor& dataDesc,
                                 const Matrix::Data<T>& vec,
                                 const raft::comms::comms_t& comm,
                                 cudaStream_t* streams,
                                 int n_streams)
{
  int rank = comm.get_rank();

  std::vector<Matrix::RankSizePair*> localBlocks = dataDesc.blocksOwnedBy(rank);

  using layout_t       = std::conditional_t<rowMajor, raft::row_major, raft::col_major>;
  constexpr auto apply = bcastAlongRows ? raft::Apply::ALONG_ROWS : raft::Apply::ALONG_COLUMNS;

  for (size_t i = 0; i < localBlocks.size(); i++) {
    auto n_rows = static_cast<int>(localBlocks[i]->size);
    auto n_cols = static_cast<int>(dataDesc.N);

    auto matrix_view =
      raft::make_device_matrix_view<T, int, layout_t>(data[i]->ptr, n_rows, n_cols);

    auto vec_size = bcastAlongRows ? n_cols : n_rows;
    auto vec_view = raft::make_device_vector_view<const T, int>(vec.ptr, vec_size);

    raft::resources handle;
    raft::resource::set_cuda_stream(handle, streams[i]);

    raft::linalg::binary_mult<apply>(handle, matrix_view, vec_view);
  }

  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamSynchronize(streams[i]));
  }
}

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryDivSkipZero(std::vector<Matrix::Data<double>*>& data,
                                   const Matrix::PartDescriptor& dataDesc,
                                   const Matrix::Data<double>& vec,
                                   bool return_zero,
                                   const raft::comms::comms_t& comm,
                                   cudaStream_t* streams,
                                   int n_streams)
{
  matrixVectorBinaryDivSkipZero_impl<rowMajor, bcastAlongRows, double>(
    data, dataDesc, vec, return_zero, comm, streams, n_streams);
}

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryDivSkipZero(std::vector<Matrix::Data<float>*>& data,
                                   const Matrix::PartDescriptor& dataDesc,
                                   const Matrix::Data<float>& vec,
                                   bool return_zero,
                                   const raft::comms::comms_t& comm,
                                   cudaStream_t* streams,
                                   int n_streams)
{
  matrixVectorBinaryDivSkipZero_impl<rowMajor, bcastAlongRows, float>(
    data, dataDesc, vec, return_zero, comm, streams, n_streams);
}

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryMult(std::vector<Matrix::Data<double>*>& data,
                            const Matrix::PartDescriptor& dataDesc,
                            const Matrix::Data<double>& vec,
                            const raft::comms::comms_t& comm,
                            cudaStream_t* streams,
                            int n_streams)
{
  matrixVectorBinaryMult_impl<rowMajor, bcastAlongRows, double>(
    data, dataDesc, vec, comm, streams, n_streams);
}

template <bool rowMajor, bool bcastAlongRows>
void matrixVectorBinaryMult(std::vector<Matrix::Data<float>*>& data,
                            const Matrix::PartDescriptor& dataDesc,
                            const Matrix::Data<float>& vec,
                            const raft::comms::comms_t& comm,
                            cudaStream_t* streams,
                            int n_streams)
{
  matrixVectorBinaryMult_impl<rowMajor, bcastAlongRows, float>(
    data, dataDesc, vec, comm, streams, n_streams);
}

template void matrixVectorBinaryDivSkipZero<false, true>(std::vector<Matrix::Data<double>*>&,
                                                         const Matrix::PartDescriptor&,
                                                         const Matrix::Data<double>&,
                                                         bool,
                                                         const raft::comms::comms_t&,
                                                         cudaStream_t*,
                                                         int);

template void matrixVectorBinaryDivSkipZero<false, true>(std::vector<Matrix::Data<float>*>&,
                                                         const Matrix::PartDescriptor&,
                                                         const Matrix::Data<float>&,
                                                         bool,
                                                         const raft::comms::comms_t&,
                                                         cudaStream_t*,
                                                         int);

template void matrixVectorBinaryMult<false, true>(std::vector<Matrix::Data<double>*>&,
                                                  const Matrix::PartDescriptor&,
                                                  const Matrix::Data<double>&,
                                                  const raft::comms::comms_t&,
                                                  cudaStream_t*,
                                                  int);

template void matrixVectorBinaryMult<false, true>(std::vector<Matrix::Data<float>*>&,
                                                  const Matrix::PartDescriptor&,
                                                  const Matrix::Data<float>&,
                                                  const raft::comms::comms_t&,
                                                  cudaStream_t*,
                                                  int);

};  // namespace opg
};  // namespace Matrix
};  // namespace MLCommon
