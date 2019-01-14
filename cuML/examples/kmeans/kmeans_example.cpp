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
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

#include <cuda_runtime.h>

#include <kmeans/kmeans_c.h>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                       \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if ( cudaSuccess != cudaStatus )                                                               \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
}
#endif //CUDA_RT_CALL

template<typename T>
T get_argval(char ** begin, char ** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char ** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char ** begin, char ** end, const std::string& arg) {
    char ** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

int main(int argc, char * argv[])
{
    const int dev_id        = get_argval<int>(argv, argv+argc,"-dev_id", 0);
    const size_t num_rows   = get_argval<size_t>(argv, argv+argc,"-num_rows", 0);
    const size_t num_cols   = get_argval<size_t>(argv, argv+argc,"-num_cols", 0);
    const std::string input = get_argval<std::string>(argv, argv+argc,"-input", std::string(""));
    //Default values for k and max_iterations are taken from
    //https://github.com/h2oai/h2o4gpu/blob/master/examples/py/demos/H2O4GPU_KMeans_Homesite.ipynb
    int k = get_argval<int>(argv, argv+argc,"-k", 10);
    int max_iterations = get_argval<int>(argv, argv+argc,"-max_iterations", 300);
    {
        cudaError_t cudaStatus = cudaSuccess;
        cudaStatus = cudaSetDevice( dev_id );
        if ( cudaSuccess != cudaStatus )
        {
            std::cerr<<"ERROR: Could not select CUDA device with the id: "<<dev_id<<"("<<cudaGetErrorString(cudaStatus)<<")"<<std::endl;
            return 1;
        }
        cudaStatus = cudaFree(0);
        if ( cudaSuccess != cudaStatus )
        {
            std::cerr<<"ERROR: Could not initialize CUDA on device: "<<dev_id<<"("<<cudaGetErrorString(cudaStatus)<<")"<<std::endl;
            return 1;
        }
    }
    
    std::vector<double> h_srcdata;
    if ( "" != input )
    {
        std::ifstream input_stream(input, std::ios::in);
        if (!input_stream.is_open())
        {
            std::cerr<<"ERROR: Could not open input file "<<input<<std::endl;
            return 1;
        }
        std::cout<<"Reading input with "<<num_rows<<" rows and "<<num_cols<<" columns from "<<input<<"."<<std::endl;
        h_srcdata.reserve(num_rows*num_cols);
        double val = 0.0;
        while ( input_stream >> val )
        {
            h_srcdata.push_back(val);
        }
    }
    bool results_correct = true;
    if ( 0 == h_srcdata.size() || (num_rows*num_cols) == h_srcdata.size() )
    {
        int k_max = k;

        double threshold = 1.0E-4; //Scikit-Learn default

        //Input parameters copied from kmeans_test.cu
        if (0 == h_srcdata.size()) {
            k = 2;
            k_max = k;
            max_iterations = 300;
            threshold = 0.05;
        }
        int dopredict = 0;
        int verbose = 0;
        int seed = 1;
        int n_gpu = 1;
        int init_from_data = 0;
        
        //Inputs copied from kmeans_test.cu
        size_t mTrain = 4; 
        size_t n = 2;
        char ord = 'c'; // here c means col order, NOT C (vs F) order
        if (0 == h_srcdata.size())
        {
            h_srcdata = {1.0,1.0,3.0,4.0, 1.0,2.0,2.0,3.0};
        }
        else
        {
            ord = 'r'; // r means row order
            mTrain = num_rows;
            n = num_cols;
        }
        std::cout<<"Run KMeans with k="<<k<<", max_iterations="<<max_iterations<<std::endl;

        //srcdata size n * mTrain
        double *srcdata = nullptr;
        CUDA_RT_CALL( cudaMalloc(&srcdata, n*mTrain*sizeof(double) ) );
        CUDA_RT_CALL( cudaMemcpy( srcdata, h_srcdata.data(), n*mTrain*sizeof(double), cudaMemcpyHostToDevice ) );

        // centroids can be nullpr
        double *centroids = nullptr;

        //output pred_centroids size k * n
        double *pred_centroids = nullptr;
        CUDA_RT_CALL( cudaMalloc(&pred_centroids, k*n*sizeof(double) ) );
        //output pred_labels size mTrain
        int *pred_labels = nullptr;
        CUDA_RT_CALL( cudaMalloc(&pred_labels, mTrain*sizeof(int) ) );

        ML::make_ptr_kmeans(dopredict, verbose, seed, dev_id, n_gpu, mTrain, n, ord, k, k_max, max_iterations, init_from_data, threshold, srcdata, centroids, pred_centroids, pred_labels);

        std::vector<int> h_pred_labels(mTrain);
        CUDA_RT_CALL( cudaMemcpy( h_pred_labels.data(), pred_labels, mTrain*sizeof(int), cudaMemcpyDeviceToHost ) );
        std::vector<double> h_pred_centroids(k * n);
        CUDA_RT_CALL( cudaMemcpy( h_pred_centroids.data(), pred_centroids, k*n*sizeof(double), cudaMemcpyDeviceToHost ) );

        if (8 == h_srcdata.size())
        {
            int h_labels_ref_fit[mTrain] = {1, 1, 0, 0};
            for ( int i = 0; i < mTrain; ++i ) {
                if ( h_labels_ref_fit[i] != h_pred_labels[i] ) {
                std::cerr<<"ERROR: h_labels_ref_fit["<<i<<"] = "<<h_labels_ref_fit[i]<<" != "<<h_pred_labels[i]<<" = h_pred_labels["<<i<<"]!"<<std::endl;
                    results_correct = false;
                }
            }

            double h_centroids_ref[k * n] = {3.5,2.5, 1.0,1.5};
            for ( int i = 0; i < k * n; ++i ) {
                if ( std::abs(h_centroids_ref[i] - h_pred_centroids[i])/std::abs(h_centroids_ref[i]) > std::numeric_limits<double>::epsilon() ) {
                std::cerr<<"ERROR: h_centroids_ref["<<i<<"] = "<<h_centroids_ref[i]<<" !~= "<<h_pred_centroids[i]<<" = h_pred_centroids["<<i<<"]!"<<std::endl;
                    results_correct = false;
                }
            }
        }
        else
        {
            std::vector<std::pair<size_t,double> > cluster_stats(k,std::make_pair(static_cast<size_t>(0),0.0));
            double global_inertia = 0.0;
            size_t max_points = 0;
            for ( size_t i = 0; i < mTrain; ++i ) {
                int label = h_pred_labels[i];
                cluster_stats[label].first += 1;
                max_points = std::max(cluster_stats[label].first,max_points);
                
                double sd = 0.0;
                for ( int j = 0; j < n; ++j ) {
                    const double cluster_centroid_comp = h_pred_centroids[label*n+j];
                    const double point_comp = h_srcdata[i*n+j];
                    sd += (cluster_centroid_comp-point_comp)*(cluster_centroid_comp-point_comp);
                }
                cluster_stats[label].second += sd;
                global_inertia += sd;
            }
            int lable_widht = 0;
            int max_label = (k-1);
            do
            {
                lable_widht += 1;
                max_label   /= 10;
            }
            while ( max_label > 0 );
            int num_pts_width = 0;
            do
            {
                num_pts_width += 1;
                max_points    /= 10;
            }
            while ( max_points > 0 );
            num_pts_width = std::max(num_pts_width,7);
            
            for ( int c = 0; c < lable_widht; ++c) std::cout<<" ";
            std::cout<<"  num_pts       inertia"<<std::endl;
            for ( int l = 0; l < k; ++l ) {
                std::cout<<std::setw(lable_widht)<<l<<"  "<<std::setw(num_pts_width)<<cluster_stats[l].first<<"  "<<std::scientific<<std::setprecision(6)<<cluster_stats[l].second<<std::endl;
            }
            std::cout<<"Global inertia = "<<global_inertia<<std::endl;
        }

        CUDA_RT_CALL( cudaFree( pred_labels ) );
        pred_labels  = nullptr;
        CUDA_RT_CALL( cudaFree( pred_centroids ) );
        pred_centroids  = nullptr;
        CUDA_RT_CALL( cudaFree( srcdata ) );
        srcdata  = nullptr;
    }
    else
    {
        std::cerr<<"ERROR: Number of input values = "<<h_srcdata.size()<<" != "<<num_rows*num_cols<<" = "<<num_rows<<"*"<<num_cols<<" !"<<std::endl;
        return 1;
    }
    CUDA_RT_CALL( cudaDeviceReset() );
    return results_correct ? 0 : 1;
}
