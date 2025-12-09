/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <opg/matrix/data.hpp>
#include <opg/matrix/part_descriptor.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

namespace ML {
namespace Matrix {
namespace opg {

void gatherPart(const raft::handle_t& h,
                float* gatheredPart,
                std::vector<Matrix::Data<float>*>& parts,
                Matrix::PartDescriptor& desc,
                int partIndex,
                int rootRank,
                int myRank,
                cudaStream_t stream);

void allGatherPart(const raft::handle_t& h,
                   float* gatheredPart,
                   std::vector<Matrix::Data<float>*>& parts,
                   Matrix::PartDescriptor& desc,
                   int partIndex,
                   int myRank,
                   cudaStream_t stream);
void gather(const raft::handle_t& h,
            float* gatheredMatrix,
            std::vector<Matrix::Data<float>*>& parts,
            Matrix::PartDescriptor& desc,
            int rootRank,
            int myRank,
            cudaStream_t stream);

void allGather(const raft::handle_t& h,
               float* gatheredMatrix,
               std::vector<Matrix::Data<float>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream);

void allocate(const raft::handle_t& h,
              std::vector<Matrix::Data<float>*>& parts,
              Matrix::PartDescriptor& desc,
              int myRank,
              cudaStream_t stream);

void deallocate(const raft::handle_t& h,
                std::vector<Matrix::Data<float>*>& parts,
                Matrix::PartDescriptor& desc,
                int myRank,
                cudaStream_t stream);

void randomize(const raft::handle_t& h,
               raft::random::Rng& r,
               std::vector<Matrix::Data<float>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream,
               float low  = -1.0f,
               float high = 1.0f);

void reset(const raft::handle_t& h,
           std::vector<Matrix::Data<float>*>& parts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream);

void printRaw2D(float* buffer, int rows, int cols, bool isColMajor, cudaStream_t stream);

void print(const raft::handle_t& h,
           std::vector<Matrix::Data<float>*>& parts,
           Matrix::PartDescriptor& desc,
           const char* matrixName,
           int myRank,
           cudaStream_t stream);

//------------------------------------------------------------------------------

void gatherPart(const raft::handle_t& h,
                double* gatheredPart,
                std::vector<Matrix::Data<double>*>& parts,
                Matrix::PartDescriptor& desc,
                int partIndex,
                int rootRank,
                int myRank,
                cudaStream_t stream);

void allGatherPart(const raft::handle_t& h,
                   double* gatheredPart,
                   std::vector<Matrix::Data<double>*>& parts,
                   Matrix::PartDescriptor& desc,
                   int partIndex,
                   int myRank,
                   cudaStream_t stream);

void gather(const raft::handle_t& h,
            double* gatheredMatrix,
            std::vector<Matrix::Data<double>*>& parts,
            Matrix::PartDescriptor& desc,
            int rootRank,
            int myRank,
            cudaStream_t stream);

void allGather(const raft::handle_t& h,
               double* gatheredMatrix,
               std::vector<Matrix::Data<double>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream);

void allocate(const raft::handle_t& h,
              std::vector<Matrix::Data<double>*>& parts,
              Matrix::PartDescriptor& desc,
              int myRank,
              cudaStream_t stream);

void deallocate(const raft::handle_t& h,
                std::vector<Matrix::Data<double>*>& parts,
                Matrix::PartDescriptor& desc,
                int myRank,
                cudaStream_t stream);

void randomize(const raft::handle_t& h,
               raft::random::Rng& r,
               std::vector<Matrix::Data<double>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream,
               double low  = -1.0,
               double high = 1.0);

void reset(const raft::handle_t& h,
           std::vector<Matrix::Data<double>*>& parts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream);

void printRaw2D(double* buffer, int rows, int cols, bool isColMajor, cudaStream_t stream);

void print(const raft::handle_t& h,
           std::vector<Matrix::Data<double>*>& parts,
           Matrix::PartDescriptor& desc,
           const char* matrixName,
           int myRank,
           cudaStream_t stream);
}  // end namespace opg
}  // namespace Matrix
}  // end namespace ML

