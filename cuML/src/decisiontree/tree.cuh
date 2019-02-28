/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include <utils.h>
#include "histogram/histogram.cuh"
#include "kernels/gini.cuh"
#include <vector>
#include <algorithm>
#include <numeric>

namespace ML {
  namespace DecisionTree {

    struct Question
    {
      int column;
      float value;
    };

    struct LeafNode
    {
      int class_predict;
    };
    
    struct TreeNode
    {
      TreeNode *left = NULL;
      TreeNode *right = NULL;
      LeafNode *leaf = NULL;  
      Question question;
    };
    
    class DecisionTreeClassifier
    {
    public:
      TreeNode *root = NULL;
      void *tempstoragedata;
      void *tempstoragelabels;
      const int nbins = 8;
      
      void fit(float *data,const int ncols,const int nrows,const float colper,int *labels)
      {
	return plant(data,ncols,nrows,colper,labels);
      }
      
      void plant(float *data,const int ncols,const int nrows,const float colper,int *labels)
      {
	CUDA_CHECK(cudaMalloc(&tempstoragedata,nrows*ncols*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&tempstoragelabels,nrows*sizeof(int)));
	
	root = grow_tree(data,ncols,nrows,colper,labels,tempstoragedata,tempstoragelabels);
	
	CUDA_CHECK(cudaFree(tempstoragedata));
	CUDA_CHECK(cudaFree(tempstoragelabels));
	return;
      }
      
      TreeNode* grow_tree(float *data,const int ncols,const int nrows,const float colper,int *labels,void *tempstorage,void *templabels)
      {
	TreeNode *node = new TreeNode();
	Question ques;
	float gain = 0.0;
	find_best_fruit(data,labels,ncols,nrows,colper,ques,gain);  //ques and gain are output here
	if(gain == 0.0)
	  {
	    
	  }
	else
	  {
	    int nrowsleft,nrowsright;
	    split_branch(data,labels,ques,nrows,ncols,tempstorage,templabels,nrowsleft,nrowsright);
	  }
	return node;
      }

      void find_best_fruit(float *data,int *labels,const int ncols,const int nrows,const float colper,Question& ques,float& gain)
      {
	float maxinfo = 0.0;
	int splitcol;
	float splitval;
	float ginibefore = gini(labels,nrows);

	// Bootstrap columns
	std::vector<int> colselector(ncols);
	std::iota(colselector.begin(),colselector.end(),0);
	std::random_shuffle(colselector.begin(),colselector.end());
	colselector.resize((int)(colper*ncols));

	int* d_histogram;
	CUDA_CHECK(cudaMalloc((void**)&d_histogram,sizeof(int)*nbins));
	for(int i=0;i<colselector.size();i++)
	  {
	    histogram(&data[nrows*colselector[i]],d_histogram,nbins,nrows);
	  }
	
      }
      
      void split_branch(float *data,int *labels,const Question ques,const int nrows,const int ncols,void *tempdata,void *templabels,int& nrowsleft,int& nrowsright)
      {

      }
    };
    
  } //End namespace DecisionTree

} //End namespace ML 
