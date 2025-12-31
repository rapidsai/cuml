/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/prims/opg/matrix/data.hpp>
#include <cuml/prims/opg/matrix/part_descriptor.hpp>

#include <raft/util/cuda_utils.cuh>

namespace MLCommon {
namespace Matrix {

PartDescriptor::PartDescriptor(
  size_t _M, size_t _N, const std::vector<RankSizePair*>& _partsToRanks, int rank, Layout _layout)
  : M(_M), N(_N), partsToRanks(_partsToRanks), rank(rank), layout(_layout)
{
  partsToRanks.resize(totalBlocks());
}

std::set<int> PartDescriptor::uniqueRanks()
{
  std::set<int> r;
  for (size_t i = 0; i < partsToRanks.size(); ++i) {
    r.insert(partsToRanks[i]->rank);
  }
  return r;
}

int PartDescriptor::totalBlocksOwnedBy(int rank) const
{
  int nBlocks = 0;
  for (const auto* p2r : partsToRanks) {
    if (p2r->rank == rank) { ++nBlocks; }
  }
  return nBlocks;
}

std::vector<RankSizePair*> PartDescriptor::blocksOwnedBy(int rank) const
{
  std::vector<RankSizePair*> res;
  for (size_t i = 0; i < partsToRanks.size(); ++i) {
    if (partsToRanks[i]->rank == rank) { res.push_back(partsToRanks[i]); }
  }
  return res;
}

std::vector<size_t> PartDescriptor::startIndices() const
{
  std::vector<size_t> res;
  size_t n_total = 0;
  for (size_t i = 0; i < partsToRanks.size(); ++i) {
    if (i < partsToRanks.size()) res.push_back(n_total);
    n_total += partsToRanks[i]->size;
  }
  return res;
}

std::vector<size_t> PartDescriptor::startIndices(int rank) const
{
  std::vector<size_t> res;
  int64_t n_total = 0;

  for (size_t i = 0; i < partsToRanks.size(); ++i) {
    if (i < partsToRanks.size() and partsToRanks[i]->rank == rank) res.push_back(n_total);
    n_total += partsToRanks[i]->size;
  }
  return res;
}

size_t PartDescriptor::totalElementsOwnedBy(int rank) const
{
  size_t total                   = 0;
  std::vector<RankSizePair*> res = blocksOwnedBy(rank);
  for (size_t i = 0; i < partsToRanks.size(); ++i) {
    if (partsToRanks[i]->rank == rank) { total += partsToRanks[i]->size; }
  }
  return total;
}

std::ostream& operator<<(std::ostream& os, const PartDescriptor& desc) { return os; }

bool operator==(const PartDescriptor& a, const PartDescriptor& b) { return true; }

};  // end namespace Matrix
};  // end namespace MLCommon
