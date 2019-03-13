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
#include "kernels/evaluate.cuh"
#include "memory.cuh"
#include "Timer.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <omp.h>

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
			void print(std::ostream& os)
			{
				if (left == NULL && right == NULL)
					os << "(leaf, " << class_predict << ")" ;
				else
					os << "(" << question.column << ", " << question.value << ")" ;
				return;
			}
		};
		std::ostream& operator<<(std::ostream& os, TreeNode* node)
		{
			node->print(os);
			return os;
		}
		struct DataInfo
		{
			unsigned int NLocalrows;
			unsigned int NGlobalrows;
			unsigned int Ncols;
		};
		
		class DecisionTreeClassifier
		{
		private:
			TreeNode *root = NULL;
			int nbins;
			DataInfo dinfo;
			int treedepth;
			int depth_counter = 0;
			int maxleaves;
			int leaf_counter = 0;
			std::vector<TemporaryMemory*> tempmem;
			size_t total_temp_mem;
			const int MAXSTREAMS = 1;
			int n_unique_labels = -1; // number of unique labels in dataset
			double construct_time;
		public:
			// Expects column major float dataset, integer labels
			// data, labels are both device ptr.
			// Assumption: labels are all mapped to contiguous numbers starting from 0 during preprocessing. Needed for gini hist impl.
			void fit(float *data, const int ncols, const int nrows, int *labels, unsigned int *rowids, const int n_sampled_rows, int unique_labels, int maxdepth = -1, int max_leaf_nodes = -1, const float colper = 1.0, int n_bins = 8)
			{
				return plant(data, ncols, nrows, labels, rowids, n_sampled_rows, unique_labels, maxdepth, max_leaf_nodes, colper, n_bins);
			}
			
			// Same as above fit, but planting is better for a tree then fitting.
			void plant(float *data, const int ncols, const int nrows, int *labels, unsigned int *rowids, const int n_sampled_rows, int unique_labels, int maxdepth = -1, int max_leaf_nodes = -1, const float colper = 1.0, int n_bins = 8)
			{
				dinfo.NLocalrows = nrows;
				dinfo.NGlobalrows = nrows;
				dinfo.Ncols = ncols;
				nbins = n_bins;
				treedepth = maxdepth;
				maxleaves = max_leaf_nodes;
				tempmem.resize(MAXSTREAMS);
				n_unique_labels = unique_labels;
				
				for(int i = 0;i<MAXSTREAMS;i++)
					{
						tempmem[i] = new TemporaryMemory(n_sampled_rows,MAXSTREAMS,unique_labels);
						
					}
				total_temp_mem = tempmem[0]->totalmem;
				total_temp_mem *= MAXSTREAMS;
				GiniInfo split_info;
				Timer timer;
				root = grow_tree(data, colper, labels, 0, rowids, n_sampled_rows, split_info);
				construct_time = timer.getElapsedSeconds();
				
				for(int i = 0;i<MAXSTREAMS;i++)
					{
						delete tempmem[i];
					}
				
				return;
			}
			
			/* Predict a label for single row for a given tree. */
			int predict(const float * row, bool verbose=false) {
				ASSERT(root, "Cannot predict w/ empty tree!");
				return classify(row, root, verbose);	
			}

			// Printing utility for high level tree info.
			void print_tree_summary()
			{
				std::cout << " Decision Tree depth --> " << depth_counter << " and n_leaves --> " << leaf_counter << std::endl;
				std::cout << " Total temporary memory usage--> "<< ((double)total_temp_mem / (1024*1024)) << "  MB" << std::endl;
				std::cout << " Tree growing time --> " << construct_time << " seconds" << std::endl;
			}

			// Printing utility for debug and looking at nodes and leaves.
			void print()
			{
				print_tree_summary();
				print_node("", root, false);
			}

		private:
			TreeNode* grow_tree(float *data, const float colper, int *labels, int depth, unsigned int* rowids, const int n_sampled_rows, GiniInfo prev_split_info)
			{
				TreeNode *node = new TreeNode();
				Question ques;
				float gain = 0.0;
				GiniInfo split_info[3]; // basis, left, right. Populate this
				split_info[0] = prev_split_info;

				find_best_fruit(data, labels, colper, ques, gain, rowids, n_sampled_rows, &split_info[0], depth);  //ques and gain are output here
				bool condition = (gain == 0.0);
				
				if (treedepth != -1)
					condition = (condition || (depth == treedepth));

				if (maxleaves != -1)
					condition = (condition || (leaf_counter >= maxleaves)); // FIXME not fully respecting maxleaves, but >= constraints it more than ==
				
				if (condition)
					{
						int *sampledlabels = tempmem[0]->sampledlabels;
						get_sampled_labels(labels, sampledlabels, rowids, n_sampled_rows);
						//node->class_predict = get_class(sampledlabels, n_sampled_rows,tempmem[0]);
						node->class_predict = get_class_hist(split_info[0].hist);
						leaf_counter++;
						if (depth > depth_counter)
							depth_counter = depth;
					}
				else
					{
						int nrowsleft, nrowsright;
						split_branch(data, ques, n_sampled_rows, nrowsleft, nrowsright, rowids);
						node->question = ques;
						node->left = grow_tree(data, colper, labels, depth+1, &rowids[0], nrowsleft, split_info[1]);
						node->right = grow_tree(data, colper, labels, depth+1, &rowids[nrowsleft], nrowsright, split_info[2]);
					}
				return node;
			}
			
			/* depth is used to distinguish between root and other tree nodes for computations */
			void find_best_fruit(float *data, int *labels, const float colper, Question& ques, float& gain, unsigned int* rowids, const int n_sampled_rows, GiniInfo split_info[3], int depth)
			{
				gain = 0.0;
				
				// Bootstrap columns
				std::vector<int> colselector(dinfo.Ncols);
				std::iota(colselector.begin(), colselector.end(), 0);
				std::random_shuffle(colselector.begin(), colselector.end());
				colselector.resize((int)(colper * dinfo.Ncols ));
				int *labelptr = tempmem[0]->sampledlabels;
				get_sampled_labels(labels, labelptr, rowids, n_sampled_rows);
				// Optimize ginibefore; no need to compute except for root.
				if (depth == 0) {
					gini(labelptr, n_sampled_rows, tempmem[0], split_info[0], n_unique_labels);
				}
				float ginibefore = split_info[0].best_gini;
				int current_nbins = (n_sampled_rows < nbins) ? n_sampled_rows+1 : nbins;

				
				
#pragma omp parallel for num_threads(1)
				for (int i=0; i<colselector.size(); i++)
					{
						GiniInfo local_split_info[3];
						local_split_info[0] = split_info[0];
						//int streamid = i % MAXSTREAMS;
						int streamid = omp_get_thread_num();
						float *sampledcolumn = tempmem[streamid]->sampledcolumns;
						int *sampledlabels = tempmem[streamid]->sampledlabels;
						int *leftlabels = tempmem[streamid]->leftlabels;
						int *rightlabels = tempmem[streamid]->rightlabels;

						get_sampled_column(&data[dinfo.NLocalrows*colselector[i]], sampledcolumn, rowids, n_sampled_rows, tempmem[streamid]->stream);
						
						float min,max;						
						min_and_max(sampledcolumn, n_sampled_rows, min, max, tempmem[streamid]);
						float delta = (max - min)/ nbins ;
						
						for (int j=1; j<current_nbins; j++)
							{
								float quesval = min + delta*j;
								float info_gain = evaluate_split(sampledcolumn, labelptr, leftlabels, rightlabels, ginibefore, quesval, &local_split_info[0], n_sampled_rows, streamid);
								CUDA_CHECK(cudaStreamSynchronize(tempmem[streamid]->stream));
								
#pragma omp critical
								{
									if (info_gain > gain)
										{
											gain = info_gain;
											ques.value = quesval;
											ques.column = colselector[i];
											for (int tmp = 0; tmp < 3; tmp++) split_info[tmp] = local_split_info[tmp];
										}
								}
								
							}
						
					}
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			
			float evaluate_split(float *column, int* labels, int* leftlabels, int* rightlabels, float ginibefore, float quesval, GiniInfo split_info[3], const int nrows, const int streamid)
			{
				int lnrows, rnrows;
				
				evaluate_and_leftgini(column,labels,quesval,nrows,n_unique_labels,ginibefore,split_info[1],lnrows,rnrows,tempmem[streamid]);
				
				// Compute giniright from the histograms of parent and parent's left node. Currently CPU only.
				gini_right_node(rnrows, split_info[0], split_info[1], split_info[2], n_unique_labels, tempmem[streamid]->stream);

				//ginileft is split_info[1].best_gini and giniright is split_info[2].best_gini	
				float impurity = (lnrows/nrows) * split_info[1].best_gini + (rnrows/nrows) * split_info[2].best_gini;
				
				return (ginibefore - impurity);
			}
			
			void split_branch(float *data, const Question ques, const int n_sampled_rows, int& nrowsleft, int& nrowsright, unsigned int* rowids)
			{
				float *colptr = &data[dinfo.NLocalrows * ques.column];
				float *sampledcolumn = tempmem[0]->sampledcolumns;
				
				get_sampled_column(colptr, sampledcolumn, rowids, n_sampled_rows);
				make_split(sampledcolumn, ques.value, n_sampled_rows, nrowsleft, nrowsright, rowids, tempmem[0]);
				
				return;
			}
			
			
			int classify(const float * row, TreeNode * node, bool verbose=false) {
				Question q = node->question;
				if (node->left && (row[q.column] <= q.value)) {
					if (verbose)
						std::cout << "Classifying Left @ node w/ column " << q.column << " and value " << q.value << std::endl;
					return classify(row, node->left, verbose);
				} else if (node->right && (row[q.column] > q.value)) {
					if (verbose)
						std::cout << "Classifying Right @ node w/ column " << q.column << " and value " << q.value << std::endl;
					return classify(row, node->right, verbose);
				} else {
					if (verbose)
						std::cout << "Leaf node. Predicting " << node->class_predict << std::endl;
					return node->class_predict;
				}
			}
			
			void print_node(const std::string& prefix, TreeNode* node, bool isLeft)
			{
				if ( node != NULL )
					{
						std::cout << prefix;
						
						std::cout << (isLeft ? "├" : "└" );
						
						// print the value of the node
						std::cout << node << std::endl;
						
						// enter the next tree level - left and right branch
						print_node( prefix + (isLeft ? "│   " : "    "), node->left, true);
						print_node( prefix + (isLeft ? "│   " : "    "), node->right, false);
					}
			}
						
		};
		
	} //End namespace DecisionTree
	
} //End namespace ML
