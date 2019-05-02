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

namespace ML {  

#define LOG(verbose, fmt, ...)					\
    do{								\
	if(verbose){						\
	    std::string msg;					\
	    char verboseMsg[2048];				\
	    std::sprintf(verboseMsg, fmt, ##__VA_ARGS__);	\
	    msg += verboseMsg;					\
	    std::cerr << msg;					\
	}							\
    } while (0)

namespace kmeans{
namespace detail{
typedef cutlass::Shape<8, 128, 128> OutputTile_8_128_128;

template<typename DataT>      
struct SamplingOp{
    DataT *rnd;
    int *flag;
    DataT cluster_cost;
    int oversampling_factor;
      
    CUB_RUNTIME_FUNCTION __forceinline__
    SamplingOp(DataT c, int l, DataT *rand, int* ptr)
	: cluster_cost(c),
	  oversampling_factor(l),
	  rnd(rand),
	  flag(ptr) {
    }

    __host__ __device__ __forceinline__
    bool operator()(const cub::KeyValuePair<ptrdiff_t, DataT> &a) const {
	DataT prob_threshold = (DataT) rnd[a.key];
	
	DataT prob_x = (( oversampling_factor * a.value) / cluster_cost);

	return !flag[a.key] && (prob_x > prob_threshold);
    }
};

template<typename IndexT,
	 typename DataT>
struct KeyValueIndexOp{
    __host__ __device__ __forceinline__
    IndexT operator()(const cub::KeyValuePair<IndexT, DataT> &a) const {
	return a.key;
    }    
};


// TODO - decide this based on avail memory and perf analysis..
template<typename CountT>
CountT
getDataBatchSize(CountT n_samples, CountT n_features){
    return std::min((CountT)(1<<16), n_samples);
}

// Computes the intensity histogram from a sequence of labels
template<typename SampleIteratorT,
	 typename CounterT>
void
countLabels(const cumlHandle_impl& handle,
	    SampleIteratorT labels,
	    CounterT *count,
	    int n_samples,
	    int n_clusters,
	    MLCommon::device_buffer<char> &workspace,
	    cudaStream_t stream) {
    int num_levels  = n_clusters + 1;
    int lower_level = 0;
    int upper_level = n_clusters;
      
    size_t temp_storage_bytes = 0;
    CUDA_CHECK( cub::DeviceHistogram::HistogramEven(nullptr,
						    temp_storage_bytes,
						    labels,
						    count,
						    num_levels,
						    lower_level,
						    upper_level,
						    n_samples,
						    stream) );

    workspace.reserve(temp_storage_bytes, stream);

    CUDA_CHECK( cub::DeviceHistogram::HistogramEven(workspace.data(),
						    temp_storage_bytes,
						    labels,
						    count,
						    num_levels,
						    lower_level,
						    upper_level,
						    n_samples,
						    stream) );
}



template<typename DataT,
	 typename IndexT>
Tensor<DataT, 2, IndexT>
sampleCentroids(const cumlHandle_impl &handle,
		Tensor<DataT, 2, IndexT>   &X,
		Tensor<DataT, 1, IndexT>   &minClusterDistance,
		Tensor<int, 1, IndexT>     &isSampleCentroid,
		typename kmeans::detail::SamplingOp<DataT> &select_op,
		MLCommon::device_buffer<char> &workspace,
		cudaStream_t stream){
    ML::detail::streamSyncer _(handle);
    int n_local_samples = X.getSize(0);
    int n_features      = X.getSize(1);
            
    Tensor<int, 1> nSelected({1},
			     handle.getDeviceAllocator(), stream);
      
    cub::ArgIndexInputIterator<DataT*> ip_itr(minClusterDistance.data());
    Tensor< cub::KeyValuePair<ptrdiff_t, DataT>, 1 > sampledMinClusterDistance({n_local_samples},
									       handle.getDeviceAllocator(), stream);
    size_t temp_storage_bytes = 0;	  	  
    CUDA_CHECK( cub::DeviceSelect::If(nullptr,
				      temp_storage_bytes,
				      ip_itr,
				      sampledMinClusterDistance.data(),
				      nSelected.data(),
				      n_local_samples,
				      select_op,
				      stream) );

    workspace.resize(temp_storage_bytes, stream);

    CUDA_CHECK( cub::DeviceSelect::If(workspace.data(),
				      temp_storage_bytes,
				      ip_itr,
				      sampledMinClusterDistance.data(),
				      nSelected.data(),
				      n_local_samples,
				      select_op,
				      stream) );


    int nPtsSampledInRank = 0;
    MLCommon::copy(&nPtsSampledInRank,
		   nSelected.data(),
		   nSelected.numElements(),
		   stream);
    CUDA_CHECK( cudaStreamSynchronize(stream) );
	
    int* rawPtr_isSampleCentroid = isSampleCentroid.data();
    thrust::for_each_n(thrust::cuda::par.on(stream),
		       sampledMinClusterDistance.begin(),
		       nPtsSampledInRank,
		       [=] __device__ (cub::KeyValuePair<ptrdiff_t, DataT> val) { 
			   rawPtr_isSampleCentroid[val.key] = 1;
		       });

    Tensor<DataT, 2, IndexT> inRankCp({nPtsSampledInRank, n_features},
				      handle.getDeviceAllocator(), stream);

					
    Matrix::gather(X.data(),
		   X.getSize(1),
		   X.getSize(0),
		   sampledMinClusterDistance.data(),
		   nPtsSampledInRank,
		   inRankCp.data(),
		   [=] __device__ (cub::KeyValuePair<ptrdiff_t, DataT> val) { // MapTransformOp
		       return val.key;
		   },
		   stream);

    CUDA_CHECK( cudaStreamSynchronize(stream) );
    return inRankCp;
}
    

template<typename DataT,
	 typename IndexT,
	 typename ReductionOpT>
DataT
computeClusterCost(const cumlHandle_impl &handle,
		   Tensor<DataT, 1, IndexT> &minClusterDistance,
		   MLCommon::device_buffer<char> &workspace,
		   ReductionOpT reduction_op,
		   cudaStream_t stream){
    Tensor<DataT, 1, IndexT> clusterCostD({1},
					  handle.getDeviceAllocator(), stream);
      
    size_t   temp_storage_bytes = 0;	  	  
    CUDA_CHECK( cub::DeviceReduce::Reduce(nullptr,
					  temp_storage_bytes,
					  minClusterDistance.data(),
					  clusterCostD.data(),
					  minClusterDistance.numElements(),
					  reduction_op,
					  DataT(),
					  stream) );
      
    workspace.resize(temp_storage_bytes, stream);
      
    CUDA_CHECK( cub::DeviceReduce::Reduce(workspace.data(),
					  temp_storage_bytes,
					  minClusterDistance.data(),
					  clusterCostD.data(),
					  minClusterDistance.numElements(),
					  reduction_op,
					  DataT(),
					  stream) );

    DataT clusterCostInRank;
    MLCommon::copy(&clusterCostInRank,
		   clusterCostD.data(),
		   clusterCostD.numElements(),
		   stream);
	
    CUDA_CHECK( cudaStreamSynchronize(stream) );
    return clusterCostInRank;
}


// calculate pairwise distance between 'dataset[n x d]' and 'centroids[k x d]',
// result will be stored in 'pairwiseDistance[n x k]'
template<typename DataT,
	 typename IndexT,
	 Distance::DistanceType DistanceType>
void
pairwiseDistanceImpl(const cumlHandle_impl& handle,
		     Tensor<DataT, 2, IndexT> &X,
		     Tensor<DataT, 2, IndexT> &centroids,
		     Tensor<DataT, 2, IndexT> &pairwiseDistance,
		     MLCommon::device_buffer<char> &workspace,
		     cudaStream_t stream){
  
    auto n_samples  = X.getSize(0);
    auto n_features = X.getSize(1);
    auto n_clusters  = centroids.getSize(0);

    size_t worksize = Distance::getWorkspaceSize
	<DistanceType, DataT, DataT, DataT>(X.data(),
					    centroids.data(),
					    n_samples,
					    n_clusters,
					    n_features);
  
    workspace.resize(worksize, stream);

      
    Distance::distance
	<DistanceType, DataT, DataT, DataT, kmeans::detail::OutputTile_8_128_128>
	(X.data(),
	 centroids.data(),
	 pairwiseDistance.data(),
	 n_samples,
	 n_clusters,
	 n_features,
	 workspace.data(),
	 worksize,
	 stream);
  
}

// calculate pairwise distance between 'dataset[n x d]' and 'centroids[k x d]',
// result will be stored in 'pairwiseDistance[n x k]'
template<typename DataT,
	 typename IndexT>
void
pairwiseDistance(const cumlHandle_impl& handle,
		 Tensor<DataT, 2, IndexT> &X,
		 Tensor<DataT, 2, IndexT> &centroids,
		 Tensor<DataT, 2, IndexT> &pairwiseDistance,
		 MLCommon::device_buffer<char> &workspace,
		 MLCommon::Distance::DistanceType metric,
		 cudaStream_t stream){
	
    auto n_samples  = X.getSize(0);
    auto n_features = X.getSize(1);
    auto n_clusters  = centroids.getSize(0);
	
    ASSERT(X.getSize(1) == centroids.getSize(1),
	   "# features in dataset and centroids are different (must be same)");

  
    if(metric == Distance::DistanceType::EucExpandedL2){
	pairwiseDistanceImpl<DataT, IndexT, Distance::DistanceType::EucExpandedL2>(handle, X, centroids, pairwiseDistance, workspace, stream);
    }else if(metric == Distance::DistanceType::EucExpandedL2Sqrt){
	pairwiseDistanceImpl<DataT, IndexT, Distance::DistanceType::EucExpandedL2Sqrt>(handle, X, centroids, pairwiseDistance, workspace, stream);
    }else if(metric == Distance::DistanceType::EucExpandedCosine){
	pairwiseDistanceImpl<DataT, IndexT, Distance::DistanceType::EucExpandedCosine>(handle, X, centroids, pairwiseDistance, workspace, stream);
    }else if(metric == Distance::DistanceType::EucUnexpandedL1){
	pairwiseDistanceImpl<DataT, IndexT, Distance::DistanceType::EucUnexpandedL1>(handle, X, centroids, pairwiseDistance, workspace, stream);
    }else{
	THROW("unknown distance metric");
    }
}

// Calculates a <key, value> pair for every sample in input 'X' where key is an index to an sample in 'centroids' (index of the nearest centroid) and 'value' is the distance between the sample and the 'centroid[key]'
template<typename DataT,
	 typename IndexT>
void
minClusterAndDistance(const cumlHandle_impl &handle,
		      Tensor<DataT, 2, IndexT> &X, 
		      Tensor<DataT, 2, IndexT> &centroids,
		      Tensor<DataT, 2, IndexT> &pairwiseDistance,
		      Tensor<cub::KeyValuePair<IndexT, DataT>, 1, IndexT> &minClusterAndDistance,
		      MLCommon::device_buffer<char> &workspace,
		      MLCommon::Distance::DistanceType metric,
		      cudaStream_t stream){
    ML::detail::streamSyncer _(handle);
    auto n_samples       = X.getSize(0);
    auto n_features      = X.getSize(1);
    auto n_clusters      = centroids.getSize(0);      
    auto dataBatchSize   = kmeans::detail::getDataBatchSize(n_samples, n_features);
	
    // tile over the input dataset 
    for (auto dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
	
	// # of samples for the current batch
	auto  ns   = std::min(dataBatchSize, n_samples - dIdx);

	// datasetView [ns x n_features] - view representing the current batch of input dataset 
	auto datasetView  = X.template view<2>({ns, n_features},
					       {dIdx, 0});

	// distanceView [ns x n_clusters]
	auto distanceView = pairwiseDistance.template view<2>({ns, n_clusters},
							      {0, 0});
	
	// minClusterAndDistanceView [ns x n_clusters]
	auto minClusterAndDistanceView = minClusterAndDistance.template view<1>({ns},
										{dIdx});

	
	// calculate pairwise distance between cluster centroids and current batch of input dataset
	kmeans::detail::pairwiseDistance(handle,
					 datasetView,
					 centroids,
					 distanceView,
					 workspace,
					 metric,
					 stream);

	// argmin reduction returning <index, value> pair
	// calculates the closest centroid and the distance to the closent centroid
	cub::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());
	LinAlg::coalescedReduction(minClusterAndDistanceView.data(),
				   distanceView.data(),
				   distanceView.getSize(1), 
				   distanceView.getSize(0), 
				   initial_value,
				   stream,
				   false,
				   [=] __device__ (const DataT val, const IndexT i) {
				       cub::KeyValuePair<IndexT, DataT> pair;
				       pair.key = i;
				       pair.value = val;
				       return pair;
				   },
				   [=] __device__ (cub::KeyValuePair<IndexT, DataT> a,
						   cub::KeyValuePair<IndexT, DataT> b) {
				       return (b.value < a.value) ? b : a;
				   },
				   [=] __device__ (cub::KeyValuePair<IndexT, DataT> pair) {
				       return pair;
				   });

    }
}
      
template<typename DataT,
	 typename IndexT>
void
minClusterDistance(const cumlHandle_impl &handle,
		   Tensor<DataT, 2, IndexT> &X, 
		   Tensor<DataT, 2, IndexT> &centroids,
		   Tensor<DataT, 2, IndexT> &pairwiseDistance,
		   Tensor<DataT, 1, IndexT> &minClusterDistance,
		   MLCommon::device_buffer<char> &workspace,
		   MLCommon::Distance::DistanceType metric,
		   cudaStream_t stream){
    ML::detail::streamSyncer _(handle);
    auto n_samples  = X.getSize(0);
    auto n_features = X.getSize(1);
    auto nc         = centroids.getSize(0);
      
    auto dataBatchSize  = kmeans::detail::getDataBatchSize(n_samples, n_features);

    // tile over the input data and calculate distance matrix [n_samples x n_clusters]
    for (int dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
	  
	// # of samples for the current batch
	int  ns   = std::min(dataBatchSize, X.getSize(0) - dIdx);

	// datasetView [ns x n_features] - view representing the current batch of input dataset 
	auto datasetView   = X.template view<2>({ns, n_features},
						{dIdx, 0});

	// minClusterDistanceView [ns x n_clusters]
	auto minClusterDistanceView = minClusterDistance.template view<1>({ns},
									  {dIdx});

	// calculate pairwise distance between cluster centroids and current batch of input dataset
	kmeans::detail::pairwiseDistance(handle,
					 datasetView,
					 centroids,
					 pairwiseDistance,
					 workspace,
					 metric,
					 stream);
    	
	LinAlg::coalescedReduction(minClusterDistanceView.data(),
				   pairwiseDistance.data(),
				   nc, // leading dimension of pairwiseDistance
				   ns, // second dimension of pairwiseDistance
				   std::numeric_limits<DataT>::max(),
				   stream,
				   false,
				   [=] __device__ (DataT val, int i) { // MainLambda
				       return val;
				   },
				   [=] __device__ (DataT a, DataT b) { // ReduceLambda
				       return (b < a) ? b : a;
				   },
				   [=] __device__ (DataT val) { // FinalLambda
				       return val;
				   });
    
    }
}
      
// shuffle and randomly select 'n_samples_to_gather' from input 'in' and stores in 'out'
// does not modify the input
template <typename DataT,
	  typename IndexT>
void
shuffleAndGather(const cumlHandle_impl &handle,
		 Tensor<DataT, 2, IndexT> &in,
		 Tensor<DataT, 2, IndexT> &out,
		 size_t n_samples_to_gather,
		 int seed,
		 cudaStream_t stream,
		 MLCommon::device_buffer<char> *workspace = nullptr){
    auto         n_samples  = in.getSize(0);
    auto         n_features = in.getSize(1);
	
    Tensor<IndexT, 1> indices({n_samples},
			      handle.getDeviceAllocator(),
			      stream);

    if(workspace){
	// shuffle indices on device using ml-prims
	Random::permute<DataT>(indices.data(),
			       nullptr,
			       nullptr,
			       in.getSize(1),
			       in.getSize(0),
			       true,
			       stream);
    }else{
	// shuffle indices on host and copy to device...
	host_buffer<IndexT> ht_indices(handle.getHostAllocator(), stream, n_samples);
	  
	std::iota(ht_indices.begin(), ht_indices.end(), 0);
	  
	std::mt19937 gen(seed);
	std::shuffle(ht_indices.begin(), ht_indices.end(), gen);

	MLCommon::copy(indices.data(), ht_indices.data(),
		       indices.numElements(), stream);
    }

    Matrix::gather(in.data(),
		   in.getSize(1),
		   in.getSize(0),
		   indices.data(),
		   n_samples_to_gather,
		   out.data(),
		   stream);
}


template<typename DataT,
	 typename IndexT>
void
countSamplesInCluster(const cumlHandle_impl& handle,
		      Tensor<DataT, 2, IndexT> &X,
		      Tensor<DataT, 2, IndexT> &centroids,
		      MLCommon::device_buffer<char> &workspace,
		      MLCommon::Distance::DistanceType metric,
		      Tensor<int, 1, IndexT> &sampleCountInCluster,
		      cudaStream_t stream){
    ML::detail::streamSyncer _(handle);
    auto n_samples  = X.getSize(0);
    auto n_features = X.getSize(1);
    auto n_clusters = centroids.getSize(0);

    int dataBatchSize = kmeans::detail::getDataBatchSize(n_samples, n_features);

    // stores (key, value) pair corresponding to each sample where
    //   - key is the index of nearest cluster 
    //   - value is the distance to the nearest cluster
    Tensor<cub::KeyValuePair<IndexT, DataT>, 1, IndexT> minClusterAndDistance({n_samples},
									      handle.getDeviceAllocator(),
									      stream);

    // temporary buffer to store distance matrix, destructor releases the resource
    Tensor<DataT, 2, IndexT> pairwiseDistance({dataBatchSize, n_clusters},
					      handle.getDeviceAllocator(),
					      stream);

    // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i] is a <key, value> pair where
    //   'key' is index to an sample in 'centroids' (index of the nearest centroid) and
    //   'value' is the distance between the sample 'X[i]' and the 'centroid[key]'
    kmeans::detail::minClusterAndDistance(handle,
					  X,
					  centroids,
					  pairwiseDistance,
					  minClusterAndDistance,
					  workspace,
					  metric,
					  stream);
      
    CUDA_CHECK( cudaStreamSynchronize(stream) );
    // Using TransformInputIteratorT to dereference an array of cub::KeyValuePair and converting them to just return the Key to be used in reduce_rows_by_key prims
    kmeans::detail::KeyValueIndexOp<IndexT, DataT> conversion_op;
    cub::TransformInputIterator<IndexT, kmeans::detail::KeyValueIndexOp<IndexT, DataT>,
				cub::KeyValuePair<IndexT, DataT>*> itr(minClusterAndDistance.data(),
								       conversion_op);
      
  
    // count # of samples in each cluster
    kmeans::detail::countLabels(handle,
				itr,
				sampleCountInCluster.data(),
				n_samples, n_clusters,
				workspace,
				stream);

}

template<typename DataT,
	 typename IndexT>
void
kmeansPlusPlus(const cumlHandle_impl& handle,
	       int verbose,
	       int n_clusters,
	       int seed,
	       Tensor<DataT, 2, IndexT> &C,
	       Tensor<int, 1, IndexT> &weights,
	       MLCommon::Distance::DistanceType metric,
	       MLCommon::device_buffer<char> &workspace,
	       MLCommon::device_buffer<DataT> &centroidsRawData,
	       cudaStream_t stream){
    ML::detail::streamSyncer _(handle);

  
    auto n_pot_centroids  = C.getSize(0); // # of potential centroids
    auto n_features = C.getSize(1);
  
    // temporary buffer for probabilities 
    Tensor<DataT, 1, IndexT> prob({n_pot_centroids},
				  handle.getDeviceAllocator(),
				  stream);

    thrust::transform(thrust::cuda::par.on(stream),
		      weights.begin(),
		      weights.end(),
		      prob.begin(),
		      [] __device__(int weight){
			  return static_cast<DataT>(weight);
		      });
  
    host_buffer<DataT> h_prob(handle.getHostAllocator(), stream);
    h_prob.resize(n_pot_centroids, stream);

    std::mt19937 gen(seed);

    // reset buffer to store the chosen centroid
    centroidsRawData.resize(n_clusters * n_features, stream);

    Tensor<DataT, 1, IndexT> minClusterDistance({n_pot_centroids},
						handle.getDeviceAllocator(),
						stream);

  
    int dataBatchSize = kmeans::detail::getDataBatchSize(n_pot_centroids, n_features);

    device_buffer<DataT> pairwiseDistanceRaw(handle.getDeviceAllocator(), stream);
    pairwiseDistanceRaw.reserve(dataBatchSize * n_clusters,
				stream);

    int n_pts_sampled = 0;
    for (int iter = 0; iter < n_clusters; iter++) {    
	LOG(verbose,
	    "KMeans++ - Iteraton %d/%d\n", iter, n_clusters);

	MLCommon::copy(h_prob.data(),
		       prob.data(),
		       prob.numElements(),
		       stream);
	CUDA_CHECK(cudaStreamSynchronize(stream) );      

	std::discrete_distribution<> d(h_prob.begin(), h_prob.end());
	// d(gen) returns random # between [0...n_pot_centroids], mod is  unncessary but just placing it to avoid untested behaviors
	int cIdx = d(gen) % n_pot_centroids; 

	LOG(verbose,
	    "Chosing centroid-%d randomly from %d potential centroids\n",
	    cIdx, n_pot_centroids);

	auto curCentroid = C.template view<2>({1, n_features},
					      {cIdx, 0});

	MLCommon::copy(centroidsRawData.data() + n_pts_sampled * n_features,
		       curCentroid.data(),
		       curCentroid.numElements(),
		       stream);
	n_pts_sampled++;    

	auto centroids = std::move(Tensor<DataT, 2, IndexT>(centroidsRawData.data(), {n_pts_sampled, n_features}));

	Tensor<DataT, 2, IndexT> pairwiseDistance((DataT *)pairwiseDistanceRaw.data(),
						  {dataBatchSize, centroids.getSize(0)});

	kmeans::detail::minClusterDistance(handle,
					   C,
					   centroids,
					   pairwiseDistance,
					   minClusterDistance,
					   workspace,
					   metric,
					   stream);

    
	DataT clusteringCost
	    = kmeans::detail::computeClusterCost(handle,
						 minClusterDistance,
						 workspace,
						 [] __device__(const DataT &a,
							       const DataT &b){
						     return a + b;
						 },
						 stream);

    
	cub::ArgIndexInputIterator<int*> itr_w(weights.data());
    
	thrust::transform(thrust::cuda::par.on(stream),
			  minClusterDistance.begin(),
			  minClusterDistance.end(),
			  itr_w,
			  prob.begin(),
			  [=] __device__ (const DataT &minDist,
					  const cub::KeyValuePair<ptrdiff_t, int> &weight) {
			      if(weight.key == cIdx){
				  // sample was chosen in the previous iteration, so reset the weights to avoid future selection...
				  return static_cast<DataT>(0); 
			      }else{
				  return weight.value * minDist/clusteringCost;
			      }
			  });


	CUDA_CHECK(cudaStreamSynchronize(stream));
    }  
}

} // end namespace detail
} // end namespace kmeans

template <typename DataT,
	  typename IndexT>
KMeans<DataT, IndexT>::~KMeans(){
    cudaStream_t stream = _handle.getStream();
    _workspace.release(stream);
    _centroidsRawData.release(stream);
    _labelsRawData.release(stream);
}

// constructor
template <typename DataT,
	  typename IndexT>
KMeans<DataT, IndexT>::KMeans(const ML::cumlHandle_impl &handle,
			      int k,
			      int metric,
			      kmeans::InitMethod init,
			      int max_iterations,
			      double tolerance,
			      int seed,
			      int verbose)
    : _handle(handle),
      n_clusters(k),
      tol(tolerance),
      max_iter(max_iterations),
      _init (init),
      _metric (static_cast<MLCommon::Distance::DistanceType>(metric)),
      _verbose(verbose),
      _workspace(handle.getDeviceAllocator(), handle.getStream()),
      _centroidsRawData(handle.getDeviceAllocator(), handle.getStream()),
      _labelsRawData(handle.getDeviceAllocator(), handle.getStream()){
    ML::detail::streamSyncer _(_handle);
    cudaStream_t stream = _handle.getStream();
    
    options.oversampling_factor = 2.0 * n_clusters; 
    if (seed < 0) {
	std::random_device rd;
	options.seed = rd();
    }else{
	options.seed = seed;
    }
    
    inertia = std::numeric_limits<DataT>::infinity();;
    n_iter = 0;
    _n_features = 0;    
}

template <typename DataT,
	  typename IndexT>
void
KMeans<DataT, IndexT>::setCentroids(const DataT *X,
				    int n_samples,
				    int n_features){
    ML::detail::streamSyncer _(_handle);
    cudaStream_t stream     = _handle.getStream();

    n_clusters = n_samples;
    options.oversampling_factor = 2.0 * n_clusters; 
    _n_features = n_features;

    _centroidsRawData.resize(0, stream);
    _centroidsRawData.reserve(n_clusters * n_features, stream);

    MLCommon::copy(_centroidsRawData.begin(), X,
		   n_samples * n_features, stream);

    auto centroids = std::move(Tensor<DataT, 2, IndexT>(_centroidsRawData.data(), {n_clusters, n_features}));

    
}

  
template <typename DataT,
	  typename IndexT>
DataT*
KMeans<DataT, IndexT>::centroids(){
    return _centroidsRawData.data();
}

  
// Selects 'n_clusters' samples randomly from X
template <typename DataT,
	  typename IndexT>
void
KMeans<DataT, IndexT>::initRandom(Tensor<DataT, 2, IndexT> &X){
    ML::detail::streamSyncer _(_handle);
    cudaStream_t stream     = _handle.getStream();
    auto         n_features = X.getSize(1);

    // allocate centroids buffer
    _centroidsRawData.resize(0, stream);
    _centroidsRawData.reserve(n_clusters * n_features, stream);
    CUDA_CHECK( cudaStreamSynchronize(stream) );
    
    auto centroids = std::move(Tensor<DataT, 2, IndexT>(_centroidsRawData.data(), {n_clusters, n_features}));

    kmeans::detail::shuffleAndGather(_handle,
				     X,
				     centroids,
				     n_clusters,
				     options.seed,
				     stream);
      
}


/*
 * @brief Selects 'n_clusters' samples from X using scalable kmeans++ algorithm
 * Scalable kmeans++ pseudocode
 * 1: C = sample a point uniformly at random from X
 * 2: psi = phi_X (C)
 * 3: for O( log(psi) ) times do
 * 4:   C' = sample each point x in X independently with probability p_x = l * ( d^2(x, C) / phi_X (C) )
 * 5:   C = C U C'
 * 6: end for
 * 7: For x in C, set w_x to be the number of points in X closer to x than any other point in C
 * 8: Recluster the weighted points in C into k clusters
 */
  
template <typename DataT,
	  typename IndexT>
void
KMeans<DataT, IndexT>::initKMeansPlusPlus(Tensor<DataT, 2, IndexT> &X){
    ML::detail::streamSyncer _(_handle);
    cudaStream_t stream = _handle.getStream();
    auto n_samples  = X.getSize(0);
    auto n_features = X.getSize(1);

    Random::Rng rng(options.seed,
		    options.gtype);

    // <<<< Step-1 >>> : C <- sample a point uniformly at random from X
    std::mt19937 gen(options.seed);
    std::uniform_int_distribution<> dis(0,
					n_samples - 1);
      
    int cIdx = dis(gen);
    auto initialCentroid = X.template view<2>({1, n_features},
					      {cIdx, 0});

    // flag the sample that is chosen as initial centroid 
    host_buffer<int> h_isSampleCentroid(_handle.getHostAllocator(), stream, n_samples);
    std::fill(h_isSampleCentroid.begin(), h_isSampleCentroid.end(), 0);
    h_isSampleCentroid[cIdx] = 1;
    
    // flag the sample that is chosen as initial centroid 
    Tensor<int, 1> isSampleCentroid({n_samples},
				    _handle.getDeviceAllocator(),
				    stream);
    
    MLCommon::copy(isSampleCentroid.data(), h_isSampleCentroid.data(),
		   isSampleCentroid.numElements(), stream);
    
    device_buffer<DataT> centroidsRawData(_handle.getDeviceAllocator(), stream);
  
    // reset buffer to store the chosen centroid
    centroidsRawData.reserve(n_clusters * n_features, stream);
    centroidsRawData.resize(initialCentroid.numElements(), stream);
    MLCommon::copy(centroidsRawData.begin(), initialCentroid.data(),
		   initialCentroid.numElements(), stream);

    CUDA_CHECK( cudaStreamSynchronize(stream) );
    
    auto potentialCentroids = std::move(Tensor<DataT, 2, IndexT>(centroidsRawData.data(), {initialCentroid.getSize(0), initialCentroid.getSize(1)}));
    // <<< End of Step-1 >>>

    
    int dataBatchSize = kmeans::detail::getDataBatchSize(n_samples, n_features);

    device_buffer<DataT> pairwiseDistanceRaw(_handle.getDeviceAllocator(), stream);
    pairwiseDistanceRaw.reserve(dataBatchSize * n_clusters,
				stream);

    
    Tensor<DataT, 1, IndexT> minClusterDistance({n_samples},
						_handle.getDeviceAllocator(), stream);   
    Tensor<DataT, 1, IndexT> uniformRands({n_samples},
					  _handle.getDeviceAllocator(), stream);
    Tensor<DataT, 1, IndexT> clusterCostD({1},
					  _handle.getDeviceAllocator(), stream);

    // <<< Step-2 >>>: psi <- phi_X (C)  
    Tensor<DataT, 2, IndexT> pairwiseDistance((DataT *)pairwiseDistanceRaw.data(),
					      {dataBatchSize, potentialCentroids.getSize(0)});

    kmeans::detail::minClusterDistance(_handle,
				       X,
				       potentialCentroids,
				       pairwiseDistance,
				       minClusterDistance,
				       _workspace,
				       _metric,
				       stream);


    DataT phi = kmeans::detail::computeClusterCost(_handle,
						   minClusterDistance,
						   _workspace,
						   [] __device__(const DataT &a, const DataT &b){
						       return a+b;
						   },
						   stream);

 
    // Scalable kmeans++ paper claims 8 rounds is sufficient 
    int niter = std::min(8, (int)ceil(log(phi)));
  
    // <<< End of Step-2 >>>

    // <<<< Step-3 >>> : for O( log(psi) ) times do  
    for(int iter = 0;
	iter < niter;
	++iter){
	LOG(_verbose,
	    "KMeans|| - Iteration %d: # potential centroids sampled - %d\n", iter, potentialCentroids.getSize(0));
    
	pairwiseDistanceRaw.resize(dataBatchSize * potentialCentroids.getSize(0),
				   stream);
	Tensor<DataT, 2, IndexT> pairwiseDistance((DataT *)pairwiseDistanceRaw.data(),
						  {dataBatchSize, potentialCentroids.getSize(0)});

	kmeans::detail::minClusterDistance(_handle,
					   X,
					   potentialCentroids,
					   pairwiseDistance,
					   minClusterDistance,
					   _workspace,
					   _metric,
					   stream);

    
	DataT clusterCost = kmeans::detail::computeClusterCost(_handle,
							       minClusterDistance,
							       _workspace,
							       [] __device__(const DataT &a, const DataT &b){
								   return a+b;
							       },
							       stream);

	// <<<< Step-4 >>> : Sample each point x in X independently and identify new potentialCentroids
	rng.uniform(uniformRands.data(), uniformRands.getSize(0), (DataT)0, (DataT)1, stream);
	kmeans::detail::SamplingOp<DataT> select_op(clusterCost,
						    options.oversampling_factor,
						    uniformRands.data(),
						    isSampleCentroid.data());
      
	auto Cp = kmeans::detail::sampleCentroids(_handle,
						  X,
						  minClusterDistance,
						  isSampleCentroid,
						  select_op,
						  _workspace,
						  stream);
	CUDA_CHECK(cudaStreamSynchronize(stream) );      
	/// <<<< End of Step-4 >>>>


	/// <<<< Step-5 >>> : C = C U C'
	// append the data in Cp to the buffer holding the potentialCentroids
	centroidsRawData.reserve(centroidsRawData.size() + Cp.numElements(), stream);
	MLCommon::copy(centroidsRawData.end(),
		       Cp.data(),
		       Cp.numElements(),
		       stream);
	centroidsRawData.resize(centroidsRawData.size() + Cp.numElements(), stream);
      
	CUDA_CHECK( cudaStreamSynchronize(stream) );
      
	int tot_centroids = potentialCentroids.getSize(0) + Cp.getSize(0);
	potentialCentroids = std::move(Tensor<DataT, 2, IndexT>(centroidsRawData.data(), {tot_centroids, n_features}));
	/// <<<< End of Step-5 >>>
    } /// <<<< Step-6 >>>

    if(potentialCentroids.getSize(0) > n_clusters){  
	// <<< Step-7 >>>: For x in C, set w_x to be the number of pts closest to X
	// temporary buffer to store the sample count per cluster, destructor releases the resource
	Tensor<int, 1, IndexT> weights({potentialCentroids.getSize(0)},
				       _handle.getDeviceAllocator(),
				       stream);

 
	kmeans::detail::countSamplesInCluster(_handle, X, potentialCentroids, _workspace, _metric, weights, stream);

	// <<< end of Step-7 >>>

	//Step-8: Recluster the weighted points in C into k clusters
	_centroidsRawData.reserve(n_clusters * n_features, stream);
	kmeans::detail::kmeansPlusPlus(_handle, _verbose, n_clusters, options.seed, potentialCentroids, weights, _metric, _workspace, _centroidsRawData, stream);
    }else{
	ASSERT(potentialCentroids.getSize(0) < n_clusters,
	       "failed to converge, restart with different seed");
    
	_centroidsRawData.reserve(n_clusters * n_features, stream);
	MLCommon::copy(_centroidsRawData.data(),
		       potentialCentroids.data(),
		       potentialCentroids.numElements(),
		       stream);
    }
}


template <typename DataT,
	  typename IndexT>
__host__
void
KMeans<DataT, IndexT>::fit(Tensor<DataT, 2, IndexT>& X){
    ML::detail::streamSyncer _(_handle);
    cudaStream_t stream  = _handle.getStream();
    auto n_samples = X.getSize(0);
    auto n_features = X.getSize(1);
    
    auto dataBatchSize = kmeans::detail::getDataBatchSize(n_samples, n_features);

    // stores (key, value) pair corresponding to each sample where
    //   - key is the index of nearest cluster 
    //   - value is the distance to the nearest cluster
    Tensor<cub::KeyValuePair<IndexT, DataT>, 1, IndexT> minClusterAndDistance({n_samples},
									      _handle.getDeviceAllocator(),
									      stream);

    // temporary buffer to store distance matrix, destructor releases the resource
    Tensor<DataT, 2, IndexT> pairwiseDistance({dataBatchSize, n_clusters},
					      _handle.getDeviceAllocator(),
					      stream);

    // temporary buffer to store intermediate centroids, destructor releases the resource
    Tensor<DataT, 2, IndexT> newCentroids({n_clusters, n_features},
					  _handle.getDeviceAllocator(),
					  stream);

    // temporary buffer to store the sample count per cluster, destructor releases the resource
    Tensor<int, 1, IndexT> sampleCountInCluster({n_clusters},
						_handle.getDeviceAllocator(),
						stream);

    
    DataT priorClusteringCost = 0;    
    for(n_iter = 0; n_iter < max_iter; ++n_iter){
	CUDA_CHECK(cudaStreamSynchronize(stream) );

	auto centroids = std::move(Tensor<DataT, 2, IndexT>(_centroidsRawData.data(), {n_clusters, n_features}));

	// computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i] is a <key, value> pair where
	//   'key' is index to an sample in 'centroids' (index of the nearest centroid) and
	//   'value' is the distance between the sample 'X[i]' and the 'centroid[key]'
	kmeans::detail::minClusterAndDistance(_handle,
					      X,
					      centroids,
					      pairwiseDistance,
					      minClusterAndDistance,
					      _workspace,
					      _metric,
					      stream);
      
	CUDA_CHECK( cudaStreamSynchronize(stream) );
      
	// Using TransformInputIteratorT to dereference an array of cub::KeyValuePair and converting them to just return the Key to be used in reduce_rows_by_key prims
	kmeans::detail::KeyValueIndexOp<IndexT, DataT> conversion_op;
	cub::TransformInputIterator<IndexT, kmeans::detail::KeyValueIndexOp<IndexT, DataT>,
				    cub::KeyValuePair<IndexT, DataT>*> itr(minClusterAndDistance.data(),
									   conversion_op);
      
	_workspace.reserve(n_samples, stream);

	// Calculates sum of all the samples assigned to cluster-i and store the result in newCentroids[i]
	LinAlg::reduce_rows_by_key(X.data(),
				   X.getSize(1),
				   itr,
				   _workspace.data(),
				   X.getSize(0),
				   X.getSize(1),
				   n_clusters,
				   newCentroids.data(),
				   stream);

      
	// count # of samples in each cluster
	kmeans::detail::countLabels(_handle,
				    itr,
				    sampleCountInCluster.data(),
				    n_samples, n_clusters,
				    _workspace,
				    stream);

      
	// wait for stream to finish before sending data to other ranks...
	CUDA_CHECK( cudaStreamSynchronize(stream) );


	// Computes newCentroids[i] = newCentroids[i]/sampleCountInCluster[i] where 
	//   newCentroids[n_samples x n_features] - 2D array, newCentroids[i] has sum of all the samples assigned to cluster-i
	//   sampleCountInCluster[n_clusters] - 1D array, sampleCountInCluster[i] contains # of samples in cluster-i. 
	// Note - when sampleCountInCluster[i] is 0, newCentroid[i] is reset to 0
      
	// transforms int values in sampleCountInCluster to its inverse and more importantly to DataT because matrixVectorOp supports only when matrix and vector are of same type
	_workspace.reserve(sampleCountInCluster.numElements() * sizeof(DataT), stream);
	auto sampleCountInClusterInverse = std::move(Tensor<DataT, 1, IndexT>((DataT *)_workspace.data(), {n_clusters}));

	thrust::transform(thrust::cuda::par.on(stream),
			  sampleCountInCluster.begin(),
			  sampleCountInCluster.end(),
			  sampleCountInClusterInverse.begin(),
			  [=] __device__ (int count){
			      if(count == 0)
				  return static_cast<DataT>(0);
			      else
				  return static_cast<DataT>(1.0)/static_cast<DataT>(count);
			  });

	LinAlg::matrixVectorOp(newCentroids.data(),
			       newCentroids.data(),
			       sampleCountInClusterInverse.data(),
			       newCentroids.getSize(1),
			       newCentroids.getSize(0),
			       true,
			       false,
			       [=] __device__ (DataT mat, DataT vec){
				   return mat * vec;
			       },
			       stream);

	// copy the centroids[i] to newCentroids[i] when sampleCountInCluster[i] is 0
	cub::ArgIndexInputIterator<int*> itr_sc(sampleCountInCluster.data());
	Matrix::gather_if(centroids.data(),
			  centroids.getSize(1),
			  centroids.getSize(0),
			  itr_sc,
			  itr_sc,
			  sampleCountInCluster.numElements(),
			  newCentroids.data(),
			  [=] __device__ (cub::KeyValuePair<ptrdiff_t, int> map){ // predicate 
			      // copy when the # of samples in the cluster is 0
			      if(map.value == 0)
				  return true;
			      else
				  return false;
			  },
			  [=] __device__ (cub::KeyValuePair<ptrdiff_t, int> map) { // map 
			      return map.key;
			  },
			  stream);


	// compute the squared norm between the newCentroids and the original centroids, destructor releases the resource
	Tensor<DataT, 1> sqrdNorm({1}, _handle.getDeviceAllocator(), stream);    
	LinAlg::mapThenSumReduce(sqrdNorm.data(),
				 newCentroids.numElements(),
				 [=] __device__(const DataT a, const DataT b){
				     DataT diff = a - b;
				     return diff * diff;
				 },
				 stream,
				 centroids.data(),
				 newCentroids.data());

      
	DataT sqrdNormError = 0;
	MLCommon::copy(&sqrdNormError,
		       sqrdNorm.data(),
		       sqrdNorm.numElements(),
		       stream);
	CUDA_CHECK( cudaStreamSynchronize(stream) );

	MLCommon::copy(_centroidsRawData.data(),
		       newCentroids.data(),
		       newCentroids.numElements(),
		       stream);
						      
	bool done = false;      
	if(options.inertia_check){
	    // calculate cluster cost phi_x(C)
	    const cub::KeyValuePair<IndexT, DataT> clusteringCost
		= kmeans::detail::computeClusterCost(_handle,
						     minClusterAndDistance,
						     _workspace,
						     [] __device__(const cub::KeyValuePair<IndexT, DataT> &a,
								   const cub::KeyValuePair<IndexT, DataT> &b){
							 cub::KeyValuePair<IndexT, DataT> res;
							 res.key = 0;
							 res.value = a.value + b.value;
							 return res;
						     },
						     stream);
     
	    DataT curClusteringCost = clusteringCost.value;
	
	    ASSERT(curClusteringCost != (DataT)0.0,
		   "Too few points and centriods being found is getting 0 cost from centers\n");
      

	    if(n_iter > 0){
		DataT delta = curClusteringCost/priorClusteringCost;
		if(delta > 1 - tol) done = true;
	    }      
	    priorClusteringCost = curClusteringCost;
	}

	if (sqrdNormError < tol) done = true;

	if(done){
	    LOG(_verbose, "Threshold triggered after %d iterations. Terminating early.\n", n_iter);
	    break;
	}     
    }
}
  
   
  
template <typename DataT,
	  typename IndexT>
__host__
void
KMeans<DataT, IndexT>::fit(const DataT *X,
			   int n_samples,
			   int n_features){
    if(n_clusters >= n_samples){
	n_clusters = n_samples;
	options.oversampling_factor = 2.0 * n_clusters; 
    }
    _n_features = n_features;
    
    ASSERT(options.oversampling_factor > 0,
	   "oversampling factor must be > 0 (requested %d)",
	   (int) options.oversampling_factor); 


    ASSERT(memory_type(X) == cudaMemoryTypeDevice ||
	   memory_type(X) == cudaMemoryTypeManaged,
	   "input data must be device accessible");
    
    Tensor<DataT, 2, IndexT> data((DataT *)X,
				  {n_samples, n_features});

   
    if(_init == kmeans::InitMethod::Random){
	//initializing with random samples from input dataset
	LOG(_verbose, "KMeans.fit: initialize cluster centers by randomly choosing from the input data.\n");
	initRandom(data);
    }else if(_init == kmeans::InitMethod::KMeansPlusPlus){
	// default method to initialize is kmeans++ 
	LOG(_verbose, "KMeans.fit: initialize cluster centers using k-means++ algorithm.\n");
	initKMeansPlusPlus(data);
    }else if(_init == kmeans::InitMethod::Array){
	LOG(_verbose, "KMeans.fit: initialize cluster centers from the ndarray array input passed to init arguement.\n");
	ASSERT(_centroidsRawData.data(),
	       "call setCentroids() method to set the initial centroids or use the other initialized methods");
    }else{
	THROW("unknown initialization method to select initial centers");
    }
    
    
    fit(data);
}


template <typename DataT,
	  typename IndexT>
__host__
void
KMeans<DataT, IndexT>::predict(Tensor<DataT, 2, IndexT> &X){
    ML::detail::streamSyncer _(_handle);
    cudaStream_t stream  = _handle.getStream();
    auto n_samples       = X.getSize(0);
    auto n_features      = X.getSize(1);
    auto dataBatchSize   = kmeans::detail::getDataBatchSize(n_samples, n_features);

    // reset inertia 
    inertia = std::numeric_limits<DataT>::infinity();

    auto centroids = std::move(Tensor<DataT, 2, IndexT>(_centroidsRawData.data(), {n_clusters, n_features}));

    Tensor<cub::KeyValuePair<IndexT, DataT>, 1> minClusterAndDistance({n_samples},
								      _handle.getDeviceAllocator(),
								      stream);
    Tensor<DataT, 2> pairwiseDistance({dataBatchSize, n_clusters},
				      _handle.getDeviceAllocator(),
				      stream);

    // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i] is a <key, value> pair where
    //   'key' is index to an sample in 'centroids' (index of the nearest centroid) and
    //   'value' is the distance between the sample 'X[i]' and the 'centroid[key]'
    kmeans::detail::minClusterAndDistance(_handle,
					  X,
					  centroids,
					  pairwiseDistance,
					  minClusterAndDistance,
					  _workspace,
					  _metric,
					  stream);


    // calculate cluster cost phi_x(C)
    const cub::KeyValuePair<IndexT, DataT> clusteringCost
	= kmeans::detail::computeClusterCost(_handle,
					     minClusterAndDistance,
					     _workspace,
					     [] __device__(const cub::KeyValuePair<IndexT, DataT> &a,
							   const cub::KeyValuePair<IndexT, DataT> &b){
						 cub::KeyValuePair<IndexT, DataT> res;
						 res.key = 0;
						 res.value = a.value + b.value;
						 return res;
					     },
					     stream);
     
    // Cluster cost phi_x(C) from all ranks
    inertia = clusteringCost.value;
  
    _labelsRawData.reserve(n_samples, stream);
    CUDA_CHECK( cudaStreamSynchronize(stream) );

    auto labels = std::move(Tensor<IndexT, 1>(_labelsRawData.data(), {n_samples}));
    thrust::transform(thrust::cuda::par.on(stream),
		      minClusterAndDistance.begin(),
		      minClusterAndDistance.end(),
		      labels.begin(),
		      [=] __device__ (cub::KeyValuePair<IndexT, DataT> pair){
			  return pair.key;
		      });
}
  
  
template <typename DataT,
	  typename IndexT>
__host__
void
KMeans<DataT, IndexT>::predict(const DataT *X,
			       int n_samples,
			       int n_features,
			       IndexT *labelsRawPtr){
    ML::detail::streamSyncer _(_handle);
    cudaStream_t stream  = _handle.getStream();

    ASSERT(_n_features == n_features,
	   "model is trained for %d-dimensional data (provided data is %d-dimensional)", _n_features, n_features);
    
    ASSERT(n_clusters > 0,
	   "no clusters exist");
    
    Tensor<DataT, 2> data((DataT *)X, {n_samples, n_features});

    predict(data);

    auto labels = std::move(Tensor<IndexT, 1>(_labelsRawData.data(), {n_samples}));
  
    MLCommon::copy(labelsRawPtr,
		   labels.data(),
		   labels.numElements(),
		   stream);
}

template <typename DataT,
	  typename IndexT>
__host__
void
KMeans<DataT, IndexT>::transform(const DataT *X,
				 int n_samples,
				 int n_features,
				 DataT *X_new){
    ML::detail::streamSyncer _(_handle);
    cudaStream_t stream  = _handle.getStream();

    ASSERT(_n_features == n_features,
	   "model is trained for %d-dimensional data (provided data is %d-dimensional)", _n_features, n_features);
    
    ASSERT(n_clusters > 0,
	   "no clusters exist");

    ASSERT(memory_type(X) == cudaMemoryTypeDevice ||
	   memory_type(X) == cudaMemoryTypeManaged,
	   "input data must be device accessible");

    ASSERT(memory_type(X) == cudaMemoryTypeDevice ||
	   memory_type(X) == cudaMemoryTypeManaged,
	   "output data storage must be device accessible");
  
  
    auto centroids = std::move(Tensor<DataT, 2, IndexT>(_centroidsRawData.data(), {n_clusters, n_features}));

    Tensor<DataT, 2> dataset((DataT *)X, {n_samples, n_features});
    Tensor<DataT, 2> pairwiseDistance((DataT *)X_new, {n_samples, n_clusters});

    auto dataBatchSize   = kmeans::detail::getDataBatchSize(n_samples, n_features);

  
    // tile over the input data and calculate distance matrix [n_samples x n_clusters]
    for (int dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
	// # of samples for the current batch
	int  ns   = std::min(dataBatchSize, n_samples - dIdx);

	// datasetView [ns x n_features] - view representing the current batch of input dataset 
	auto datasetView   = dataset.template view<2>({ns, n_features},
						      {dIdx, 0});

	// pairwiseDistanceView [ns x n_clusters]
	auto pairwiseDistanceView = pairwiseDistance.template view<2>({ns, n_clusters},
								      {dIdx, 0});

    
	// calculate pairwise distance between cluster centroids and current batch of input dataset
	kmeans::detail::pairwiseDistance(_handle,
					 datasetView,
					 centroids,
					 pairwiseDistanceView,
					 _workspace,
					 _metric,
					 stream);
    }
}
}; // end namespace ML
