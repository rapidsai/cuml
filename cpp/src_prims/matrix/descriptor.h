/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#pragma once

#include <stdint.h>
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include "cuda_utils.h"


namespace MLCommon {
namespace Matrix {

/** Describes the data layout */
enum Layout {
  /** row major layout */
  LayoutRowMajor = 0,
  /** column major layout */
  LayoutColMajor
};


/**
 * @brief Descriptor data structure to provide info about the way data is
 *  distributed among workers.
 * <code>
 *   // single-node DGX2 with a fixed 2D block cyclic data distribution
 *   Descriptor md(40000, 40000, 1000, 1000, 4, 4);
 *   // print basic info
 *   printf("Total-size=%dx%d block-size=%dx%d\n", md.M, md.N, md.MB, md.NB);
 *   // print ownership of all blocks
 *   for(int i=0;i<md.numRowBlocks();++i) {
 *     for(int j=0;j<md.numColBlocks();++j) {
 *       printf("Device that owns global 2D-block row,col=%d,%d is %d\n",
 *              i, j, md.ranks[md.linearBlockIndex(i, j)]);
 *     }
 *   }
 *
 *   // single-node DGX1 with a custom data distribution
 *   Descriptor md1(40000, 40000, 20000, 10000, {0, 1, 2, 3, 4, 5, 6, 7});
 * </code>
 * @note Following assumptions come along with this structure:
 * <ul>
 *  <li>Block is a contiguous sub-matrix that's stored in one worker</li>
 *  <li>One worker can own multiple blocks</li>
 *  <li>Blocks are organized in column major order</li>
 * </ul>
 */
template<typename SizeType>
class Descriptor {

    public:

      /**
       * @brief for a given matrix with only a single block.
       * @param _M number of rows in matrix
       * @param _N number of columns in matrix
       * @param device the device id of the matrix
       * @param intraBlk are the blocks in row-major or col-major?
       */
      Descriptor(SizeType _M, SizeType _N, int device, Layout intraBlk = LayoutRowMajor):
          M(_M), N(_N), MB(M), NB(N), intraBlockLayout(intaBlk)  {
          blocks2device.resize(1);
          blocks2device[0] = device;
      }

      /**
       * @brief For a given matrix and block-sizes construct the corresponding
       *  descriptor for it. This is useful when we are dealing with standard
       *  row/column-wise block-cyclic data distribution, as seen in other popular
       *  multi-node packages like magma etc.
       * @param _M total number of rows of this matrix
       * @param _N total number of columns
       * @param _MB number of rows in a block
       * @param _NB number of cols in a block
       * @param numDevicesRows number of devices across the rows
       * @param numDevicesCols number of devices across the cols
       * @param intraBlk layout inside of each block
       */
      Descriptor(SizeType _M, SizeType _N, SizeType _MB, SizeType _NB,
              SizeType numDevicesRows, SizeType numDevicesCols,
                 Layout intraBlk = LayoutRowMajor)
        : M(_M), N(_N), MB(_MB), NB(_NB), blocks2device(),
          intraBlockLayout(intraBlk) {
        blocks2device.resize(totalBlocks());

        int myBlockId = 0;
        // get the mapping of deviceId owning each block
        for (int i = 0; i < totalBlocks(); ++i) {
          int ridx = i % numRowBlocks();
          int cidx = i / numRowBlocks();
          int blkRowIdx = ridx % numDevicesRows;
          int blkColIdx = cidx % numDevicesCols;
          int devId = blkRowIdx + blkColIdx * numDevicesRows;
          blocks2device[i] = devId;
        }
      }

      /**
       * @brief For a given matrix and block-sizes construct the corresponding
       *  descriptor using the custom device allocation as specified by the user.
       * @param _M total number of rows of this matrix
       * @param _N total number of columns
       * @param _MB number of rows in a block
       * @param _NB number of cols in a block
       * @param list the custom blocks2devices assignment as done by the caller
       * @param intraBlk layout inside of each block
       */
      Descriptor(SizeType _M, SizeType _N, SizeType _MB, SizeType _NB,
                 const std::vector<int> &b2d, Layout intraBlk = LayoutRowMajor)
        : M(_M), N(_N), MB(_MB), NB(_NB), blocks2device(b2d),
          intraBlockLayout(intraBlk) {
        ASSERT((int)blocks2device.size() == totalBlocks(),
               "Total blocks computed (%d) doesn't match with input list (%d)!",
               totalBlocks(), (int)blocks2device.size());
      }

      /**
       * @brief Mapping from global to local block (negative if block is not owned)
       * @param out array of mappings from global to local blocks of size totalBlocks()
       *            (-1 if block is not lock, block id otherwise).
       * @param myRank the rank of the local process
       */
      void global_to_local(int* out, int myRank) {
       int myBlockId = 0;
        for (int i = 0; i < totalBlocks(); ++i) {
            out[i] = -1;
            if (blocks2device[i] == myRank) {
                out[i] = myBlockId;
                ++myBlockId;
            }
        }
      }


      /** total number of blocks across all workers */
      int totalBlocks() const { return numRowBlocks() * numColBlocks(); }

      /** total number of blocks along the row dimension */
      int numRowBlocks() const { return ceildiv(M, MB); }

      /** total number of blocks along the column dimension */
      int numColBlocks() const { return ceildiv(N, NB); }

      int getM() const {
          return M;
      }

      int getN() const {
          return N;
      }

      /**
       * @brief get the block's linear index from its global 2D-tileId
       * @param ridx row index of this block among all blocks
       * @param cidx col index of this block among all blocks
       * @return the linear index of this block in the `blocks2device` array
       */
      int linearBlockIndex(int ridx, int cidx) const {
        return ridx + cidx * numRowBlocks();
      }

      /**
       * @brief get the block's 2D-tileId (ridx,cidx) from its linear index
       * @param linear index of this block among all blocks
       * @return the pair (ridx,cidx) for the block specfied
       */
      std::pair<int,int> blockPosition(int linearBlockIndex) const {
        return std::make_pair(linearBlockIndex % numRowBlocks(), linearBlockIndex / numRowBlocks());
      }

      /** Count the total number of blocks owned by a given rank */
      int totalBlocksOwnedBy(int rank) const {
        int nBlocks = 0;
        for (const auto &b2d : blocks2device) {
          if (b2d == rank) {
            ++nBlocks;
          }
        }
        return nBlocks;
      }

      /** Returns the vector of blocks (each identified by linearBLockIndex) owned by the given rank */
      std::vector<int> blocksOwnedBy(int rank) const {
        std::vector<int> res;
        for(int i = 0; i < blocks2device.size(); ++i) {
          if (blocks2device[i] == rank) {
            res.push_back(i);
          }
        }
        return res;
      }

      /** Returns the 0-based offset of the block in the local buffer. Increments by 1 for each local block */
      int blockOffset(int linearBlockIndex) const {
        int rankId = blocks2device[linearBlockIndex];
        int offset = 0;
        for (int i = 0; i < linearBlockIndex; ++i) {
          if (blocks2device[i] == rankId) {
            ++offset;
          }
        }
        return offset;
      }

      /** Count the total number of matrix elements owned by a given rank */
      int totalElementsOwnedBy(int rank) const {
          return MB * NB * totalBlocksOwnedBy(rank);
      }

      /** Print matrix descriptor in human readable form */
      friend std::ostream& operator<< (std::ostream& os, const Descriptor &desc) {
        os << std::endl;
        os << "    matrix dimensions: " << desc.M  << " x " << desc.N  << std::endl;
        os << "     block dimensions: " << desc.MB << " x " << desc.NB << std::endl;
        os << "        blocks2device: ";
        for (const auto &b2d : desc.blocks2device)
          os << b2d << " ";
        os << std::endl;
        os << "         global2local: ";
        for (const auto &g2l : desc.global2local)
          os << g2l << " ";
        os << std::endl;
        return os;
      }


    private:
        /** total number of rows */
        SizeType M;
        /** total number of columns */
        SizeType N;
        /** number of contiguous rows per block owned by a worker */
        SizeType MB;
        /** number of contiguous cols per block owned by a worker */
        SizeType NB;
        /** mapping of each block (in col-major order) to the device that owns it */
        std::vector<int> blocks2device;
        /**
         * intra block data layout. Note that inter-block layout is still assumed to
         * be column-major always!
         */
        Layout intraBlockLayout;

  ///@todo: a converter to/from cudaLibDnDistMatrixDescriptor_t??
};


/** check whether the descriptors are the same */
inline bool operator==(const Descriptor &a, const Descriptor &b) {
  auto equal = a.M == b.M && a.N == b.N && a.MB == b.MB && a.NB == b.NB &&
               a.blocks2device.size() == b.blocks2device.size() &&
               a.intraBlockLayout == b.intraBlockLayout;
  if (!equal)
    return false;
  for (size_t i = 0; i < a.blocks2device.size(); ++i) {
    if (a.blocks2device[i] != b.blocks2device[i])
      return false;
  }
  return true;
}

};
}; // end namespace ML
