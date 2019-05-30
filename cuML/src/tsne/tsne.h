
#include "utils.h"

using namespace ML;
using namespace MLCommon::Sparse;

#include <assert.h>
#include <stdio.h>

#include "distances.h"
#include "perplexity_search.h"
#include "intialization.h"
#include "bounding_box.h"
#include "build_tree.h"
#include "summary.h"
#include "repulsion.h"
#include "attraction.h"

#pragma once


template <typename Type>
float *runTsne( const Type * __restrict__ X,
                const int n,
                const int p,
                const int n_components = 2,
                const float perplexity = 50f,
                const float perplexity_epsilon = 1e-3,
                const float early_exaggeration = 2.0f,
                int n_neighbors = 100,

                // Learning rates and momentum
                const float learning_rate = 200,
                const float pre_momentum = 0.5f,
                const float post_momentum = 0.8f,

                // Barnes Hut parameters
                const float theta = 0.5f,
                const float epsilon_squared = 0.0025f,

                // Iterations, termination crtierion
                const int exaggeration_iter = 250,
                const int max_iter = 1000,
                const float min_grad_norm = 1e-7f,

                // Seed for random data
                const long long seed = -1)
{   
    // Currently only allows n_components = 2
    assert(n_components == 2);
    if (n_neighbors > n) n_neighbors = n;


    // Intialize cache levels and errors
    int *errd;       cuda_malloc(errd, 1);
    Intialization_::Initialize(errd);
    //


    // Get GPU information
    cudaDeviceProp GPU_info;
    cudaGetDeviceProperties(&GPU_info, 0);

    if (GPU_info.warpSize != WARPSIZE) {
        fprintf(stderr, "Warp size must be %d\n", GPU_info.warpSize);
        exit(-1);
    }
    //


    // Nodes needed for BH
    int nnodes = 2*n;
    if (nnodes < 1024 * BLOCKS) nnodes = 1024 * BLOCKS;
    while ((nnodes & (WARPSIZE - 1)) != 0) nnodes++;
    nnodes--;

    const int N_NODES = nnodes;
    const int SIZE = n*n_neighbors;


    // From glm.cu
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));


    // Allocate distances
    float *distances;   cuda_malloc(distances, SIZE);
    long *indices;      cuda_malloc(indices, SIZE);

    // Use FAISS for nearest neighbors [returns squared norm]
    // Divide by max(abs(D)) to not cause exp(D) to explode
    Distances_::getDistances(X, n, p, indices, distances, n_neighbors, SIZE, stream);


    // Allocate Pij
    float *Pij;         cuda_malloc(Pij, SIZE);
    // Search Perplexity
    Perplexity_Search_::searchPerplexity(Pij, distances, perplexity, perplexity_epsilon,
        n, n_neighbors, SIZE, stream);
    cuda_free(distances);


    // Change P to COO matrix
    COO<float> P;
    MLCommon::Sparse::from_knn(indices, Pij, n, n_neighbors, &P);
    cuda_free(Pij);


    // Perform P + P.T
    COO<float> P_PT;    // P and P + P.T
    Perplexity_Search_::P_add_PT(indices, n, n_neighbors, &P, &P_PT, stream);
    P.destroy();
    cuda_free(indices);
    const int NNZ = P_PT.nnz;   // Get total NNZ

    // Convert COO to CSR matrix
    float *VAL = P_PT.vals;
    int *COL = P_PT.cols;
    int *ROW;       cuda_malloc(ROW, n+1);
    MLCommon::Sparse::sorted_coo_to_csr(&P_PT, ROW, stream);


    // Create indices array
    int *mapping_i_to_j;    cuda_malloc(mapping_i_to_j, NNZ);
    Attraction_::map_i_to_j\
        <<<Utils_::ceildiv(NNZ, 1024), 1024>>>(n, NNZ, ROW, COL, mapping_i_to_j);
    //

    // Allocate space
    const int SIZE_EMBEDDING = (N_NODES + 1)*2;

    float *PQ;              cuda_malloc(PQ, NNZ);
    float *repulsion;       cuda_calloc(repulsion, SIZE_EMBEDDING, 0.0f, stream);
    float *attraction;      cuda_calloc(attraction, n*2, 0.0f, stream);

    float *normalization;   cuda_malloc(normalization, N_NODES+1);
    thrust::device_ptr<float> normalization_begin = thrust::device_pointer_cast(normalization);
    thrust::device_ptr<float> normalization_end = normalization_begin + N_NODES+1;
    float norm;


    float *gains;           cuda_calloc(gains, n*2, 1.0f, stream);
    float *old_forces;      cuda_calloc(prev_forces, n*2, 0.0f, stream);

    int *cell_starts;       cuda_malloc(cell_starts, N_NODES+1);
    int *children;          cuda_malloc(children, (N_NODES+1)*4);
    float *cell_mass;       cuda_calloc(cell_mass, N_NODES+1, 1.0f, stream);
    int *cell_counts;       cuda_malloc(cell_counts, N_NODES+1);
    int *cell_sorted;       cuda_malloc(cell_sorted, N_NODES+1);
    
    float *x_max;           cuda_malloc(x_max, BLOCKS*FACTOR1);
    float *y_max;           cuda_malloc(y_max, BLOCKS*FACTOR1);
    float *x_min;           cuda_malloc(x_min, BLOCKS*FACTOR1);
    float *y_min;           cuda_malloc(y_min, BLOCKS*FACTOR1);
    float *ones;            cuda_calloc(ones, n*2, 1.0f, stream);


    // Intialize embedding
    float *embedding = Intialization_::randomVector(-100, 100, SIZE_EMBEDDING, seed, stream);
    thrust::device_ptr<float> embedding_begin = thrust::device_pointer_cast(embedding);

    // Make a random vector to add noise to the embeddings
    float *noise = Intialization_::randomVector(-0.05, 0.05, SIZE_EMBEDDING, seed, stream);
    thrust::device_ptr<float> noise_begin = thrust::device_pointer_cast(noise);


    // Gradient updates
    float exaggeration = early_exaggeration;
    float momentum = pre_momentum;

    // Component locations
    float * __restrict__ embedding_x = embedding;
    float * __restrict__ embedding_y = embedding + N_NODES + 1;
    float * __restrict__ repulsion_x = repulsion;
    float * __restrict__ repulsion_y = repulsion + N_NODES + 1;


    // Calculate attraction force grid / block size
    int ATTRACT_BLOCKSIZE;
    int ATTRACT_MIN_GRID;
    cudaOccupancyMaxPotentialBlockSize(
        &ATTRACT_MIN_GRID, &ATTRACT_BLOCKSIZE, Attraction_::PQ_Kernel, 0, 0);
    const int ATTRACT_GRIDSIZE = (NNZ + ATTRACT_BLOCKSIZE - 1) / ATTRACT_BLOCKSIZE;
    //



    // Convert to cuSPARSE layout
    CSR_t CSR_Matrix;
    Dense_t Ones_Matrix, Embedding_Matrix, Output_Matrix;
    Sparse_handle_t Sparse_Handle; createHandle(&Sparse_Handle);

    Utils_::createCSR(&CSR_Matrix, n, n, NNZ, PQ, COL, ROW);
    Utils_::createDense(&Ones_Matrix, n, 2, ones);
    Utils_::createDense(&Embedding_Matrix, N_NODES+1, 2, embedding);
    Utils_::createDense(&Output_Matrix, n, 2, attraction);

    void *buffer = Utils_::createBuffer(Sparse_Handle, CSR_Matrix, Ones_Matrix, Output_Matrix);
    //


    // Gradient Loop
    for (size_t i = 0; i < max_iter; i++) {
        if (i == exaggeration_iter) {
            exaggeration = 1.0f;
            momentum = post_momentum;
        }

        // Zero out repulsion and attraction
        cuda_memset(attraction, n*2);
        cuda_memset(repulsion, SIZE_EMBEDDING);
        //


        // Find bounding boxes for points
        BoundingBox_::boundingBoxKernel<<<BLOCKS*FACTOR1, THREADS1>>>(
            N_NODES, n,
            cell_starts,
            children,
            cell_mass,
            embedding_x,
            embedding_y,
            x_max,
            y_max,
            x_min,
            y_min);
        cuda_synchronize();
        //


        // Create KD Tree
        BuildTree_::clearKernel1<<<BLOCKS, 1024>>>(N_NODES, n, children);

        BuildTree_::treeBuildingKernel<<<BLOCKS*FACTOR2, THREADS2>>>(
            N_NODES, n, errd,
            children,
            embedding_x,
            embedding_y);
        
        BuildTree_::clearKernel2<<<BLOCKS, 1024>>>(N_NODES, cell_starts, cell_mass);
        cuda_synchronize();
        //


        // Summarize KD Tree and sort the cells
        Summary_::summarizationKernel<<<BLOCKS*FACTOR3, THREADS3>>>(
            N_NODES, n,
            cell_counts,
            children,
            cell_mass,
            embedding_x,
            embedding_y);
        cuda_synchronize();

        Summary_::sortKernel<<<BLOCKS*FACTOR4, THREADS4>>>(
            N_NODES, n,
            cell_sorted,
            cell_counts,
            cell_starts,
            children);
        cuda_synchronize();
        //


        // Repulsive forces
        Repulsion_::repulsionKernel<<<BLOCKS*FACTOR5, THREADS5>>>(
            N_NODES, n, errd,
            theta, epsilon_squared,
            cell_sorted,
            children,
            cell_mass,
            embedding_x,
            embedding_y,
            repulsion_x,
            repulsion_y,
            normalization);
        cuda_synchronize();

        // Calculate normalization
        norm = Utils_::sum_array(normalization_begin, normalization_end);


        // Attractive forces
        Attraction_::attractionForces(
            n, NNZ, N_NODES,
            ATTRACT_GRIDSIZE, ATTRACT_BLOCKSIZE,
            stream, Sparse_Handle,
            VAL,
            mapping_i_to_j,
            PQ,
            embedding,
            attraction,
            CSR_Matrix,
            Ones_Matrix,
            Embedding_Matrix,
            Output_Matrix,
            buffer);
        cuda_synchronize();


        // Apply all forces
        ApplyForces_::applyForcesKernell<<<BLOCKS*FACTOR6, THREADS6>>>(
            n, N_NODES,
            learning_rate,
            norm,
            momentum,
            exaggeration,
            embedding,
            attraction,
            repulsion,
            gains,
            old_forces);
        cuda_synchronize();


        // Add random noise to embedding so points don't overlap
        // embedding[:n] = embedding + noise
        thrust::transform(embedding_begin, embedding_begin + n, 
            noise_begin, embedding_begin, thrust::plus<float>());

        //
        thrust::transform(embedding_begin + N_NODES+1, embedding_begin + N_NODES+1+n,
            noise_begin + n, embedding_begin + N_NODES+1, thrust::plus<float>());
    }
    //

    // Free everything
    P_PT.destroy();
    cuda_free(ROW);

    destroyCSR(CSR_Matrix);
    destroyDense(Ones_Matrix);
    destroyDense(Embedding_Matrix);
    destroyDense(Output_Matrix);

    cuda_free(noise);
    cuda_free(embedding);

    cuda_free(ones);
    cuda_free(y_min);
    cuda_free(x_min);
    cuda_free(y_max);
    cuda_free(x_max);

    cuda_free(cell_sorted);
    cuda_free(cell_mass);
    cuda_free(children);
    cuda_free(cell_starts);

    cuda_free(old_forces);
    cuda_free(gains);

    cuda_free(normalization);
    cuda_free(attraction);
    cuda_free(repulsion);
    cuda_free(PQ);

    cuda_free(mapping_i_to_j);


    cuda_free(errd);

    // Destory CUDA stream
    CUDA_CHECK(cudaStreamDestroy(stream));


    // Return data
    float *return_data;         cuda_malloc(return_data, 2*n);
    thrust::device_ptr<float> return_data_begin = thrust::device_pointer_cast(return_data);

    thrust::copy(embedding_begin, embedding_begin + n, 
        return_data_begin + 2*n);

    thrust::copy(embedding_begin + N_NODES+1, embedding_begin + N_NODES+1+n, 
        return_data_begin + n);

    return return_data;
}
