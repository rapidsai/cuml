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
#include "algo_helper.h"
#include "kernels/gini.cuh"
#include "kernels/split_labels.cuh"
#include "kernels/col_condenser.cuh"
#include "kernels/evaluate.cuh"
#include "kernels/quantile.cuh"
#include "memory.cuh"
#include "Timer.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <climits>

namespace ML {
	namespace DecisionTree {
		
		template<class T>
		struct Question {
			int column;
			T value;

			void update(GiniQuestion<T> ques)
			{
				column = ques.original_column;
				value = ques.value;
			};
		};
		
		template<class T>
		struct TreeNode
		{
			TreeNode *left = NULL;
			TreeNode *right = NULL;
			int class_predict;
			Question<T> question;
			T gini_val;
			
			void print(std::ostream& os)
			{
				if (left == NULL && right == NULL)
					os << "(leaf, " << class_predict << ", " << gini_val << ")" ;
				else
					os << "(" << question.column << ", " << question.value << ", " << gini_val << ")" ;

				return;
			}
		};
		template<typename T>
		std::ostream& operator<<(std::ostream& os, TreeNode<T> * node)
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

		template<class T>
		class DecisionTreeClassifier
		{
		private:
			int split_algo;
			TreeNode<T> *root = NULL;
			int nbins;
			DataInfo dinfo;
			int treedepth;
			int depth_counter = 0;
			int maxleaves;
			int leaf_counter = 0;
			std::vector<TemporaryMemory<T>*> tempmem;
			size_t total_temp_mem;
			const int MAXSTREAMS = 1;
			size_t max_shared_mem;
			size_t shmem_used = 0;
			int n_unique_labels = -1; // number of unique labels in dataset
			double construct_time;
			
		public:
			// Expects column major T dataset, integer labels
			// data, labels are both device ptr.
			// Assumption: labels are all mapped to contiguous numbers starting from 0 during preprocessing. Needed for gini hist impl.
			void fit(T *data, const int ncols, const int nrows, int *labels, unsigned int *rowids, const int n_sampled_rows, int unique_labels, int maxdepth = -1, int max_leaf_nodes = -1, const float colper = 1.0, int n_bins = 8, int split_algo=SPLIT_ALGO::HIST)
			{
				return plant(data, ncols, nrows, labels, rowids, n_sampled_rows, unique_labels, maxdepth, max_leaf_nodes, colper, n_bins, split_algo);
			}

			// Same as above fit, but planting is better for a tree then fitting.
			void plant(T *data, const int ncols, const int nrows, int *labels, unsigned int *rowids, const int n_sampled_rows, int unique_labels, int maxdepth = -1, int max_leaf_nodes = -1, const float colper = 1.0, int n_bins = 8, int split_algo_flag = SPLIT_ALGO::HIST)

			{
				split_algo = split_algo_flag;
				dinfo.NLocalrows = nrows;
				dinfo.NGlobalrows = nrows;
				dinfo.Ncols = ncols;
				nbins = n_bins;
				treedepth = maxdepth;
				maxleaves = max_leaf_nodes;
				tempmem.resize(MAXSTREAMS);
				n_unique_labels = unique_labels;

				cudaDeviceProp prop;
				CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
				max_shared_mem = prop.sharedMemPerBlock;
				
				if (split_algo == SPLIT_ALGO::HIST) {
					shmem_used += 2 * sizeof(T) * ncols;
					shmem_used += nbins * n_unique_labels * sizeof(int) * ncols;
				} else {
					shmem_used += nbins * n_unique_labels * sizeof(int) * ncols;
				}
				ASSERT(shmem_used <= max_shared_mem, "Shared memory per block limit %zd , requested %zd \n", max_shared_mem, shmem_used);
				
				for (int i = 0; i<MAXSTREAMS; i++) {
					tempmem[i] = new TemporaryMemory<T>(n_sampled_rows, ncols, MAXSTREAMS, unique_labels, n_bins, split_algo);
					if (split_algo == SPLIT_ALGO::GLOBAL_QUANTILE) {
						preprocess_quantile(data, rowids, n_sampled_rows, ncols, dinfo.NLocalrows, n_bins, tempmem[i]);
					}
				}
				total_temp_mem = tempmem[0]->totalmem;
				total_temp_mem *= MAXSTREAMS;
				GiniInfo split_info;
				Timer timer;
				root = grow_tree(data, colper, labels, 0, rowids, n_sampled_rows, split_info);
				construct_time = timer.getElapsedSeconds();

				for (int i = 0;i<MAXSTREAMS;i++) {
					delete tempmem[i];
				}
				
				return;
			}
			
			/* Predict a label for single row for a given tree. */
			int predict(const T * row, bool verbose=false) {
				ASSERT(root, "Cannot predict w/ empty tree!");
				return classify(row, root, verbose);	
			}

			// Printing utility for high level tree info.
			void print_tree_summary() {
				std::cout << " Decision Tree depth --> " << depth_counter << " and n_leaves --> " << leaf_counter << std::endl;
				std::cout << " Total temporary memory usage--> "<< ((double)total_temp_mem / (1024*1024)) << "  MB" << std::endl;
				std::cout << " Tree growing time --> " << construct_time << " seconds" << std::endl;
				std::cout << " Shared memory used --> " << shmem_used << "  bytes " << std::endl;
			}

			// Printing utility for debug and looking at nodes and leaves.
			void print()
			{
				print_tree_summary();
				print_node("", root, false);
			}

		private:
			TreeNode<T> * grow_tree(T *data, const float colper, int *labels, int depth, unsigned int* rowids, const int n_sampled_rows, GiniInfo prev_split_info)
			{
				TreeNode<T> *node = new TreeNode<T>();
				GiniQuestion<T> ques;
				Question<T> node_ques;
				float gain = 0.0;
				GiniInfo split_info[3]; // basis, left, right. Populate this
				split_info[0] = prev_split_info;
				
				bool condition = ((depth != 0) && (prev_split_info.best_gini == 0.0f));  // This node is a leaf, no need to search for best split
				if (!condition)  {
					find_best_fruit_all(data,  labels, colper, ques, gain, rowids, n_sampled_rows, &split_info[0], depth);  //ques and gain are output here
					condition = condition || (gain == 0.0f);
				}
				
				if (treedepth != -1)
					condition = (condition || (depth == treedepth));

				if (maxleaves != -1)
					condition = (condition || (leaf_counter >= maxleaves)); // FIXME not fully respecting maxleaves, but >= constraints it more than ==
				
				if (condition) {
					node->class_predict = get_class_hist(split_info[0].hist);
					node->gini_val = split_info[0].best_gini;

					leaf_counter++;
					if (depth > depth_counter)
						depth_counter = depth;
				} else {
					int nrowsleft, nrowsright;
					split_branch(data, ques, n_sampled_rows, nrowsleft, nrowsright, rowids); // populates ques.value
					node_ques.update(ques);
					//std::cout << "split branch: " << n_sampled_rows << ", " << nrowsleft << ", " << nrowsright <<  ", ques (value, column) " << ques.value << ", " << ques.original_column << std::endl;
					node->question = node_ques;
					node->left = grow_tree(data, colper, labels, depth+1, &rowids[0], nrowsleft, split_info[1]);
					node->right = grow_tree(data, colper, labels, depth+1, &rowids[nrowsleft], nrowsright, split_info[2]);
					node->gini_val = split_info[0].best_gini;
				}
				return node;
			}
			
			/* depth is used to distinguish between root and other tree nodes for computations */
			void find_best_fruit_all(T *data, int *labels, const float colper, GiniQuestion<T> & ques, float& gain, unsigned int* rowids, const int n_sampled_rows, GiniInfo split_info[3], int depth)
			{

				// Bootstrap columns
				std::vector<int> colselector(dinfo.Ncols);
				std::iota(colselector.begin(), colselector.end(), 0);
				std::random_shuffle(colselector.begin(), colselector.end());
				colselector.resize((int)(colper * dinfo.Ncols ));

				CUDA_CHECK(cudaHostRegister(colselector.data(), sizeof(int) * colselector.size(), cudaHostRegisterDefault));
				// Copy sampled column IDs to device memory
				CUDA_CHECK(cudaMemcpy(tempmem[0]->d_colids, colselector.data(), sizeof(int) * colselector.size(), cudaMemcpyHostToDevice));

				// Optimize ginibefore; no need to compute except for root.
				if (depth == 0) {
					int *labelptr = tempmem[0]->sampledlabels;
					get_sampled_labels(labels, labelptr, rowids, n_sampled_rows);
					gini(labelptr, n_sampled_rows, tempmem[0], split_info[0], n_unique_labels);
				}

				int current_nbins = (n_sampled_rows < nbins) ? n_sampled_rows : nbins;
				best_split_all_cols(data, rowids, labels, current_nbins, n_sampled_rows, n_unique_labels, dinfo.NLocalrows, colselector, tempmem[0], &split_info[0], ques, gain, split_algo);

				//Unregister
				CUDA_CHECK(cudaHostUnregister(colselector.data()));
			}

			void split_branch(T *data, GiniQuestion<T> & ques, const int n_sampled_rows, int& nrowsleft, int& nrowsright, unsigned int* rowids)
			{
				T *temp_data = tempmem[0]->temp_data;
				T *sampledcolumn = &temp_data[n_sampled_rows * ques.bootstrapped_column];
				make_split(sampledcolumn, ques, n_sampled_rows, nrowsleft, nrowsright, rowids, split_algo, tempmem[0]);
			}


			int classify(const T * row, TreeNode<T> * node, bool verbose=false) {
				Question<T> q = node->question;
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

			void print_node(const std::string& prefix, TreeNode<T>* node, bool isLeft)
			{
				if (node != NULL) {
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
