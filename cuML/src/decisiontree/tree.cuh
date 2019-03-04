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
#include "kernels/col_condenser.cuh"
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
		
		struct TreeNode
		{
			TreeNode *left = NULL;
			TreeNode *right = NULL;
			int class_predict;  
			Question question;
		};

		struct DataInfo
		{
			unsigned int NLocalrows;
			unsigned int NGlobalrows;
			unsigned int Ncols;
		};
		
		class DecisionTreeClassifier
		{
		public:
			TreeNode *root = NULL;
			const int nbins = 8;
			DataInfo dinfo;
			void fit(float *data,const int ncols,const int nrows,const float colper,int *labels,unsigned int *rowids,const int n_sampled_rows)
			{
				return plant(data,ncols,nrows,colper,labels,rowids,n_sampled_rows);
			}
			
			void plant(float *data,const int ncols,const int nrows,const float colper,int *labels,unsigned int *rowids,const int n_sampled_rows)
			{
				dinfo.NLocalrows = nrows;
				dinfo.NGlobalrows = nrows;
				dinfo.Ncols = ncols;
				
				root = grow_tree(data,colper,labels,0,rowids,n_sampled_rows);
				return;
			}
			
			TreeNode* grow_tree(float *data,const float colper,int *labels,int depth,unsigned int* rowids,const int n_sampled_rows)
			{
				TreeNode *node = new TreeNode();
				Question ques;
				float gain = 0.0;
				
				find_best_fruit(data,labels,colper,ques,gain,rowids,n_sampled_rows);  //ques and gain are output here
				if(gain == 0.0)
					{
						node->class_predict = get_class(labels);
					}
				else
					{
						int nrowsleft,nrowsright;
						split_branch(data,ques,n_sampled_rows,nrowsleft,nrowsright,rowids);
						node->left = grow_tree(data,colper,&labels[0],depth+1,&rowids[0],nrowsleft);
						node->right = grow_tree(data,colper,&labels[nrowsleft],depth+1,&rowids[nrowsleft],nrowsright);
					}
				return node;
			}
			
			void find_best_fruit(float *data,int *labels,const float colper,Question& ques,float& gain,unsigned int* rowids,const int n_sampled_rows)
			{
				gain = 0.0;
				float *sampledcolumn;
				int *sampledlabels;
				
				// Bootstrap columns
				std::vector<int> colselector(dinfo.Ncols);
				std::iota(colselector.begin(),colselector.end(),0);
				std::random_shuffle(colselector.begin(),colselector.end());
				colselector.resize((int)(colper * dinfo.Ncols ));
				
				int *leftlabels, *rightlabels;
				CUDA_CHECK(cudaMalloc((void**)&leftlabels,n_sampled_rows*sizeof(int)));
				CUDA_CHECK(cudaMalloc((void**)&rightlabels,n_sampled_rows*sizeof(int)));
				CUDA_CHECK(cudaMalloc((void**)&sampledcolumn,n_sampled_rows*sizeof(float)));
				CUDA_CHECK(cudaMalloc((void**)&sampledlabels,n_sampled_rows*sizeof(int)));
				
				get_sampled_labels(labels,sampledlabels,rowids,n_sampled_rows);
				int *labelptr = sampledlabels;
				float ginibefore = gini(labelptr,n_sampled_rows);
				
				
				for(int i=0;i<colselector.size();i++)
					{
						
						get_sampled_column(&data[dinfo.NLocalrows*colselector[i]],sampledcolumn,rowids,n_sampled_rows);
						float *colptr = sampledcolumn;
						float min = minimum(colptr,n_sampled_rows);
						float max = maximum(colptr,n_sampled_rows);
						float delta = (max - min)/ nbins ;
						
						for(int j=1;j<nbins;j++)
							{
								float quesval = min + delta*j;
								float info_gain = evaluate_split(colptr,labelptr,leftlabels,rightlabels,ginibefore,quesval,n_sampled_rows);
								if(info_gain == -1.0)
									continue;
								
								if(info_gain > gain)
									{
										gain = info_gain;
										ques.value = quesval;
										ques.column = colselector[i];
									}
							}
						
					}
				

				CUDA_CHECK(cudaFree(sampledcolumn));
				CUDA_CHECK(cudaFree(sampledlabels));
				CUDA_CHECK(cudaFree(leftlabels));
				CUDA_CHECK(cudaFree(rightlabels));
				
			}
			
			float evaluate_split(float *column,int* labels,int* leftlabels,int* rightlabels,float ginibefore,float quesval,const int nrows)
			{
				int lnrows,rnrows;
				
				split_labels(column,labels,leftlabels,rightlabels,nrows,lnrows,rnrows,quesval);
				
				if(lnrows == 0 || rnrows == 0)
					return -1.0;
				
				float ginileft = gini(leftlabels,lnrows);       
				float giniright = gini(rightlabels,rnrows);
				
				
				float impurity = (lnrows/nrows) * ginileft + (rnrows/nrows) * giniright;
				
				return (ginibefore - impurity);
			}
			
			void split_branch(float *data,const Question ques,const int n_sampled_rows,int& nrowsleft,int& nrowsright,unsigned int* rowids)
			{
				float *colptr = &data[dinfo.NLocalrows * ques.column];
				float *sampledcolumn;
				
				CUDA_CHECK(cudaMalloc((void**)&sampledcolumn,n_sampled_rows*sizeof(float)));
				get_sampled_column(colptr,sampledcolumn,rowids,n_sampled_rows);
				
				make_split(sampledcolumn,ques.value,n_sampled_rows,nrowsleft,nrowsright,rowids);
				CUDA_CHECK(cudaFree(sampledcolumn));
				return;
			}

			/* Predict a label for single row for a given tree. */
			int predict(const float * row) {
				return classify(row, root);	
			}


			int classify(const float * row, TreeNode * node) {
				if (node->left || node->right) {
					if (row[node->question.column] <= node->question.value) { //FIXME confirm question is <= format
						return classify(row, node->left);
					} else  {
						return classify(row, node->right);
					}
				} else {
					//return node->class_predict; //FIXME however we decide to implement this 
					return 0;
				}
			}
			
		};
		
	} //End namespace DecisionTree
	
} //End namespace ML 
