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
#include "kernels/minmax.cuh"
#include "kernels/split_labels.cuh"
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
			const int nbins = 8;
			
			void fit(float *data,const int ncols,const int nrows,const float colper,int *labels,unsigned int *rowids)
			{
				return plant(data,ncols,nrows,colper,labels,rowids);
			}
			
			void plant(float *data,const int ncols,const int nrows,const float colper,int *labels,unsigned int *rowids)
			{
				root = grow_tree(data,ncols,nrows,colper,labels,0,rowids);
				return;
			}
			
			TreeNode* grow_tree(float *data,const int ncols,const int nrows,const float colper,int *labels,int depth,unsigned int* rowids)
			{
				TreeNode *node = new TreeNode();
				Question ques;
				float gain = 0.0;
				
				find_best_fruit(data,labels,ncols,nrows,colper,ques,gain,rowids);  //ques and gain are output here
				if(gain == 0.0)
					{
						
					}
				else
					{
						int nrowsleft,nrowsright;
						split_branch(data,labels,ques,nrows,ncols,nrowsleft,nrowsright,rowids);
					}
				return node;
			}
			
			void find_best_fruit(float *data,int *labels,const int ncols,const int nrows,const float colper,Question& ques,float& gain,unsigned int* rowids)
			{
				float maxinfo = 0.0;
				int splitcol;
				float splitval;
				float ginibefore = gini(labels,nrows);
				float *sampledcolumn;
				
				// Bootstrap columns
				std::vector<int> colselector(ncols);
				std::iota(colselector.begin(),colselector.end(),0);
				std::random_shuffle(colselector.begin(),colselector.end());
				colselector.resize((int)(colper*ncols));
				
				int *leftlabels, *rightlabels;
				CUDA_CHECK(cudaMalloc((void**)&leftlabels,nrows*sizeof(int)));
				CUDA_CHECK(cudaMalloc((void**)&rightlabels,nrows*sizeof(int)));
				CUDA_CHECK(cudaMalloc((void**)&sampledcolumn,nrows*sizeof(float)));
				
				for(int i=0;i<colselector.size();i++)
					{
						float *colptr = &data[nrows*colselector[i]];
						
						float min = minimum(colptr,nrows);
						float max = maximum(colptr,nrows);
						float delta = (max - min)/ nbins ;
						
						for(int j=1;j<nbins;j++)
							{
								float quesval = min + delta*j;
								float info_gain = evaluate_split(colptr,labels,leftlabels,rightlabels,ginibefore,quesval,nrows);
								if(info_gain == -1)
									continue;
								
								if(info_gain > maxinfo)
									{
										maxinfo = info_gain;
										splitval = quesval;
										splitcol = colselector[i];
									}
							}
						
					}
				
				ques.column = splitcol;
				ques.value = splitval;
				gain = maxinfo;

				CUDA_CHECK(cudaFree(sampledcolumn));
				CUDA_CHECK(cudaFree(leftlabels));
				CUDA_CHECK(cudaFree(rightlabels));
				
			}
			
			float evaluate_split(float *column,int* labels,int* leftlabels,int* rightlabels,float ginibefore,float quesval,const int nrows)
			{
				int lnrows,rnrows;
				
				split_labels(column,labels,leftlabels,rightlabels,nrows,lnrows,rnrows,quesval);
				
				if(lnrows == 0 || rnrows == 0)
					return -1;
				
				float ginileft = gini(leftlabels,lnrows);
				float giniright = gini(rightlabels,rnrows);
				
				float impurity = (lnrows/nrows) * ginileft + (rnrows/nrows) * giniright;
				return (ginibefore - impurity);
			}
			
			void split_branch(float *data,int *labels,const Question ques,const int nrows,const int ncols,int& nrowsleft,int& nrowsright,unsigned int* rowids)
			{
				
				
			}
			
		};
		
	} //End namespace DecisionTree
	
} //End namespace ML 
