template <typename DataT, typename LabelT, typename IdxT, int TPB, int samples_per_thread>
__global__ void computeSplitRegressionKernel_part1(
  DataT* pred, DataT* pred2, DataT* pred2P, IdxT* count, IdxT nbins,
  IdxT max_depth, IdxT min_samples_split, IdxT min_samples_leaf,
  DataT min_impurity_decrease, IdxT max_leaves,
  Input<DataT, LabelT, IdxT> input, const Node<DataT, LabelT, IdxT>* nodes,
  IdxT colStart, int* done_count, int* mutex, const IdxT* n_leaves,
  volatile Split<DataT, IdxT>* splits, void* workspace, CRITERION splitType,
  IdxT treeid, uint64_t seed, 
  WorkloadInfo<IdxT>* workload_info, bool proportionate_launch) {
  extern __shared__ char smem[];
  // Read workload info for this block 
  WorkloadInfo<IdxT> workload_info_cta = workload_info[blockIdx.x];
  IdxT nid;
  if (proportionate_launch) {
    nid = workload_info_cta.nodeid;
    // nid = blockid_to_nodeid[blockIdx.x];
  } else {
    nid = blockIdx.z;
  }

  auto node = nodes[nid];
  auto range_start = node.start;
  auto range_len = node.count;

  IdxT relative_blockid, num_blocks;
  if (proportionate_launch) {
    // relative_blockid = relative_blockids[blockIdx.x];
    relative_blockid = workload_info_cta.offset_blockid;
    // num_blocks = (range_len / (TPB*samples_per_thread)) == 0 ?
    //            1 : (range_len / (TPB*samples_per_thread));
    // num_blocks = (range_len + (TPB*samples_per_thread - 1)) /
    //              (TPB*samples_per_thread);
    num_blocks = workload_info_cta.num_blocks;

  } else {
    relative_blockid = blockIdx.x;
    num_blocks = gridDim.x;
  }

  // exit if current node is leaf
  if (leafBasedOnParams<DataT, IdxT>(node.depth, max_depth, min_samples_split,
                                     max_leaves, n_leaves, range_len)) {
    return;
  }
  // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
  //   printf("In part1 beginning\n");
  // }
  // variables
  auto end = range_start + range_len;
  // auto len = nbins * 2;
  auto pdf_spred_len = 1 + nbins;
  // auto cdf_spred_len = 2 * nbins;
  IdxT stride = blockDim.x * num_blocks;
  IdxT tid = threadIdx.x + relative_blockid * blockDim.x;
  IdxT col;

  // allocating pointers to shared memory
  // auto* pdf_spred = alignPointer<DataT>(smem);
  // auto* cdf_spred = alignPointer<DataT>(pdf_spred + pdf_spred_len);
  // auto* pdf_scount = alignPointer<int>(cdf_spred + cdf_spred_len);
  // auto* cdf_scount = alignPointer<int>(pdf_scount + nbins);
  // auto* sbins = alignPointer<DataT>(cdf_scount + nbins);
  // auto* spred2 = alignPointer<DataT>(sbins + nbins);
  // auto* spred2P = alignPointer<DataT>(spred2 + len);
  // auto* spredP = alignPointer<DataT>(spred2P + nbins);
  // auto* sDone = alignPointer<int>(spredP + nbins);

  auto* pdf_spred = alignPointer<DataT>(smem);
  auto* pdf_scount = alignPointer<int>(pdf_spred + pdf_spred_len);
  auto* sbins = alignPointer<DataT>(pdf_scount + nbins);


  // select random feature to split-check
  // (if feature-sampling is true)
  if (input.nSampledCols == input.N) {
    col = colStart + blockIdx.y;
  } else {
    int colIndex = colStart + blockIdx.y;
    col = select(colIndex, treeid, node.info.unique_id, seed, input.N);
  }

  // memset smem pointers
  for (IdxT i = threadIdx.x; i < pdf_spred_len; i += blockDim.x) {
    pdf_spred[i] = DataT(0.0);
  }
  // for (IdxT i = threadIdx.x; i < cdf_spred_len; i += blockDim.x) {
  //   cdf_spred[i] = DataT(0.0);
  // }
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    pdf_scount[i] = 0;
    // cdf_scount[i] = 0;
    sbins[i] = input.quantiles[col * nbins + i];
  }
  __syncthreads();
  auto coloffset = col * input.M;

  // compute prediction pdfs and count pdfs
  for (auto i = range_start + tid; i < end; i += stride) {
    auto row = input.rowids[i];
    auto d = input.data[row + coloffset];
    auto label = input.labels[row];
    for (IdxT b = 0; b < nbins; ++b) {
      // if sample is less-than-or-equal to threshold
      if (d <= sbins[b]) {
        atomicAdd(pdf_spred + b, label);
        atomicAdd(pdf_scount + b, 1);
        break;
      }
    }
  }
  __syncthreads();

  // update the corresponding global location for counts
  auto gcOffset = ((nid * gridDim.y) + blockIdx.y) * nbins;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    atomicAdd(count + gcOffset + i, pdf_scount[i]);
  }

  // update the corresponding global location for preds
  auto gOffset = ((nid * gridDim.y) + blockIdx.y) * pdf_spred_len;
  for (IdxT i = threadIdx.x; i < pdf_spred_len; i += blockDim.x) {
    atomicAdd(pred + gOffset + i, pdf_spred[i]);
  }
  // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
  //   printf("In part1 ending\n");
  // }

}
  // __threadfence();  // for commit guarantee
  // __syncthreads();

  // // Wait until all blockIdx.x's are done
  // MLCommon::GridSync gs(workspace, MLCommon::SyncType::ACROSS_X, false);
  // gs.sync();


template <typename DataT, typename LabelT, typename IdxT, int TPB, int samples_per_thread>
__global__ void computeSplitRegressionKernel_part2(
  DataT* pred, DataT* pred2, DataT* pred2P, IdxT* count, IdxT nbins,
  IdxT max_depth, IdxT min_samples_split, IdxT min_samples_leaf,
  DataT min_impurity_decrease, IdxT max_leaves,
  Input<DataT, LabelT, IdxT> input, const Node<DataT, LabelT, IdxT>* nodes,
  IdxT colStart, int* done_count, int* mutex, const IdxT* n_leaves,
  volatile Split<DataT, IdxT>* splits, void* workspace, CRITERION splitType,
  IdxT treeid, uint64_t seed,
  WorkloadInfo<IdxT>* workload_info, bool proportionate_launch) {

  extern __shared__ char smem[];

  // Read workload info for this block 
  WorkloadInfo<IdxT> workload_info_cta = workload_info[blockIdx.x];

  IdxT nid;
  if (proportionate_launch) {
    // nid = blockid_to_nodeid[blockIdx.x];
    nid = workload_info_cta.nodeid;
  } else {
    nid = blockIdx.z;
  }
  auto node = nodes[nid];
  auto range_start = node.start;
  auto range_len = node.count;

  IdxT relative_blockid, num_blocks;
  if (proportionate_launch) {
    // relative_blockid = relative_blockids[blockIdx.x];
    relative_blockid = workload_info_cta.offset_blockid;
    // num_blocks = (range_len / (TPB*samples_per_thread)) == 0 ?
    //            1 : (range_len / (TPB*samples_per_thread));
    // num_blocks = (range_len + (TPB*samples_per_thread - 1)) /
    //              (TPB*samples_per_thread);
    num_blocks = workload_info_cta.num_blocks;
  } else {
    relative_blockid = blockIdx.x;
    num_blocks = gridDim.x;
  }
  
  // exit if current node is leaf
  if (leafBasedOnParams<DataT, IdxT>(node.depth, max_depth, min_samples_split,
                                     max_leaves, n_leaves, range_len)) {
    return;
  }
  // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
  //   printf("In part2 beginning\n");
  // }
  // variables
  auto end = range_start + range_len;
  auto len = nbins * 2;
  auto pdf_spred_len = 1 + nbins;
  auto cdf_spred_len = 2 * nbins;
  IdxT stride = blockDim.x * num_blocks;
  IdxT tid = threadIdx.x + relative_blockid * blockDim.x;
  IdxT col;

  // allocating pointers to shared memory
  auto* pdf_spred = alignPointer<DataT>(smem);
  auto* cdf_spred = alignPointer<DataT>(pdf_spred + pdf_spred_len);
  auto* pdf_scount = alignPointer<int>(cdf_spred + cdf_spred_len);
  auto* cdf_scount = alignPointer<int>(pdf_scount + nbins);
  auto* sbins = alignPointer<DataT>(cdf_scount + nbins);
  auto* spred2 = alignPointer<DataT>(sbins + nbins);
  auto* spred2P = alignPointer<DataT>(spred2 + len);
  auto* spredP = alignPointer<DataT>(spred2P + nbins);
  auto* sDone = alignPointer<int>(spredP + nbins);

  // select random feature to split-check
  // (if feature-sampling is true)
  if (input.nSampledCols == input.N) {
    col = colStart + blockIdx.y;
  } else {
    int colIndex = colStart + blockIdx.y;
    col = select(colIndex, treeid, node.info.unique_id, seed, input.N);
  }

  // for (IdxT i = threadIdx.x; i < pdf_spred_len; i += blockDim.x) {
  //   pdf_spred[i] = DataT(0.0);
  // }
  for (IdxT i = threadIdx.x; i < cdf_spred_len; i += blockDim.x) {
    cdf_spred[i] = DataT(0.0);
  }
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    // pdf_scount[i] = 0;
    cdf_scount[i] = 0;
    sbins[i] = input.quantiles[col * nbins + i];
  }
  __syncthreads();

  auto gcOffset = ((nid * gridDim.y) + blockIdx.y) * nbins;
  // transfer from global to smem
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    pdf_scount[i] = count[gcOffset + i];
    spred2P[i] = DataT(0.0);
  }

  auto gOffset = ((nid * gridDim.y) + blockIdx.y) * pdf_spred_len;
  for (IdxT i = threadIdx.x; i < pdf_spred_len; i += blockDim.x) {
    pdf_spred[i] = pred[gOffset + i];
  }
  // memset spred2
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    spred2[i] = DataT(0.0);
  }
  __syncthreads();

  /** pdf to cdf conversion **/

  /** get cdf of spred from pdf_spred **/
  // cdf of samples lesser-than-equal to threshold
  DataT total_sum = pdf_to_cdf<DataT, IdxT, TPB>(pdf_spred, cdf_spred, nbins);

  // cdf of samples greater than threshold
  // calculated by subtracting lesser-than-equals from total_sum
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    *(cdf_spred + nbins + i) = total_sum - *(cdf_spred + i);
  }

  /** get cdf of scount from pdf_scount **/
  pdf_to_cdf<int, IdxT, TPB>(pdf_scount, cdf_scount, nbins);
  __syncthreads();

  // calcualting prediction average-sums
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    spredP[i] = cdf_spred[i] + cdf_spred[i + nbins];
  }
  __syncthreads();

  // now, compute the mean value to be used for metric update
  auto invlen = DataT(1.0) / range_len;
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    auto cnt_l = DataT(cdf_scount[i]);
    auto cnt_r = DataT(range_len - cdf_scount[i]);
    cdf_spred[i] /= cnt_l;
    cdf_spred[i + nbins] /= cnt_r;
    spredP[i] *= invlen;
  }
  __syncthreads();

  /* Make a second pass over the data to compute gain */
  auto coloffset = col * input.M;
  // 2nd pass over data to compute partial metric across blockIdx.x's
  if (splitType == CRITERION::MAE) {
    for (auto i = range_start + tid; i < end; i += stride) {
      auto row = input.rowids[i];
      auto d = input.data[row + coloffset];
      auto label = input.labels[row];
      for (IdxT b = 0; b < nbins; ++b) {
        auto isRight = d > sbins[b];  // no divergence
        auto offset = isRight * nbins + b;
        auto diff = label - (isRight ? cdf_spred[nbins + b] : cdf_spred[b]);
        atomicAdd(spred2 + offset, raft::myAbs(diff));
        atomicAdd(spred2P + b, raft::myAbs(label - spredP[b]));
      }
    }
  } else {
    for (auto i = range_start + tid; i < end; i += stride) {
      auto row = input.rowids[i];
      auto d = input.data[row + coloffset];
      auto label = input.labels[row];
      for (IdxT b = 0; b < nbins; ++b) {
        auto isRight = d > sbins[b];  // no divergence
        auto offset = isRight * nbins + b;
        auto diff = label - (isRight ? cdf_spred[nbins + b] : cdf_spred[b]);
        auto diff2 = label - spredP[b];
        atomicAdd(spred2 + offset, (diff * diff));
        atomicAdd(spred2P + b, (diff2 * diff2));
      }
    }
  }
  __syncthreads();

  // update the corresponding global location for pred2P
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    atomicAdd(pred2P + gcOffset + i, spred2P[i]);
  }

  // changing gOffset for pred2 from that of pred
  gOffset = ((nid * gridDim.y) + blockIdx.y) * len;
  // update the corresponding global location for pred2
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    atomicAdd(pred2 + gOffset + i, spred2[i]);
  }
  __threadfence();  // for commit guarantee
  __syncthreads();

  // last threadblock will go ahead and compute the best split
  bool last = true;
  if (num_blocks > 1) {
    last = MLCommon::signalDone(done_count + nid * gridDim.y + blockIdx.y,
                                num_blocks, relative_blockid == 0, sDone);
  }
  // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
  //   printf("In part2 ending\n");
  // }
  // exit if not last
  if (!last) return;

  // last block computes the final gain
  // create a split instance to test current feature split
  Split<DataT, IdxT> sp;
  sp.init();

  // store global pred2 and pred2P into shared memory of last x-dim block
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    spred2[i] = pred2[gOffset + i];
  }
  for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
    spred2P[i] = pred2P[gcOffset + i];
  }
  __syncthreads();

  // calculate the best candidate bins (one for each block-thread) in current
  // feature and corresponding regression-metric gain for splitting
  regressionMetricGain(spred2, spred2P, cdf_scount, sbins, sp, col, range_len,
                       nbins, min_samples_leaf, min_impurity_decrease);
  __syncthreads();

  // calculate best bins among candidate bins per feature using warp reduce
  // then atomically update across features to get best split per node (in split[nid])
  sp.evalBestSplit(smem, splits + nid, mutex + nid);
}
