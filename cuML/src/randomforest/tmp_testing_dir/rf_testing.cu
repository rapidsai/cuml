/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "../randomforest.cu"
#include "../../../../thirdparty/cuml/ml-prims/src/utils.h"
#include "ml_utils.h"
#include "cuda_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

//Modified version of TIMEIT_LOOP from test_utils.h
#define TIMEIT_LOOP(ms, count, func)			\
	do {										\
		cudaEvent_t start, stop;				\
		CUDA_CHECK(cudaEventCreate(&start));	\
		CUDA_CHECK(cudaEventCreate(&stop));		\
		CUDA_CHECK(cudaEventRecord(start));		\
		for (int i = 0; i < count; ++i) {		\
			func;								\
		}										\
		CUDA_CHECK(cudaEventRecord(stop));		\
		CUDA_CHECK(cudaEventSynchronize(stop));	\
		ms = 0.f;								\
		CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));	\
		ms /= count;									\
} while (0)

using namespace MLCommon;
using namespace std;

template<typename L>
void parse_csv(string dataset_name, int n_cols, std::vector<float> & data, std::vector<L> & labels, int train_cnt,
			   std::vector<float> & test_data, std::vector<L> & test_labels, int test_cnt, bool test_is_train) {

	string data_file;
	int col_offset = 0;
	int label_id =  0; // column that is the label (i.e., target feature)
 	if (dataset_name == "higgs") {
 		data_file = "/gpfs/fs1/myrtop/rapids_repos/HIGGS.csv";
 		col_offset = 1; //because the first column in higgs is the label
	} else if (dataset_name == "year") {
		data_file = "/gpfs/fs1/myrtop/rapids_repos/year.csv";
		col_offset = 1; //because the first column in year is the label
		label_id = 0;
	} else if ((dataset_name == "airline_regression") || (dataset_name == "airline")) {
		data_file = "/gpfs/fs1/myrtop/rapids_repos/airline_14col.data_modified";
		label_id = n_cols;
	}

	cout << "train_cnt " << train_cnt << " test_cnt " << test_cnt << endl;
	ifstream myfile;
	myfile.open(data_file);
	string line;

	int counter = 0;
	data.resize(train_cnt * n_cols);
	labels.resize(train_cnt);

	test_data.resize(test_cnt * n_cols);
	test_labels.resize(test_cnt);

	int break_cnt = (test_is_train) ? train_cnt : train_cnt + test_cnt;

	while (getline(myfile,line) && (counter < break_cnt)) {
			stringstream str(line);
			vector<float> row;
			float i;
			while ( str >> i) {
				row.push_back(i);
				if(str.peek() == ',')
					str.ignore();
			}
			for (int col = 0; col < n_cols; col++) {
				if (counter < train_cnt)  {
					data[counter + col * train_cnt] = row[col + col_offset]; //train data should be col major
					if (test_is_train)
						test_data[counter*n_cols + col] = row[col + col_offset]; // test data should be row major
				} else if (!test_is_train)
					test_data[(counter - train_cnt)*n_cols + col] = row[col + col_offset]; // test data should be row major
			}

			if (counter < train_cnt)  {
				labels[counter] = (dataset_name == "airline") ? (int) (row[label_id] > 0) : row[label_id];
				if (test_is_train) test_labels[counter] = labels[counter];
			} else if (!test_is_train) {
				test_labels[counter - train_cnt] = (dataset_name == "airline") ? (int) (row[label_id] > 0) : row[label_id];
			}
			counter++;
		}
	cout << "Lines processed " << counter << endl;
	myfile.close();

}


struct RF_inputs {
	int n_rows, n_cols, n_inference_rows;
	int n_trees, max_depth, max_leaves, n_bins, split_algo;
	float max_features, rows_sample, train_ratio;
	bool bootstrap, test_is_train, bootstrap_features;
	string dataset;
	int min_rows_per_node;
	ML::CRITERION split_criterion;

	RF_inputs(int cfg_n_rows, int cfg_n_cols, int cfg_n_trees, float cfg_max_features,
				float cfg_rows_sample, float cfg_train_ratio, int cfg_max_depth,
				int cfg_max_leaves, bool cfg_bootstrap, bool cfg_test_is_train, int cfg_n_bins,
				string cfg_dataset, int cfg_split_algo, int cfg_min_rows_per_node, bool cfg_bootstrap_features, ML::CRITERION cfg_split_criterion) {

		train_ratio = cfg_train_ratio;
		test_is_train = cfg_test_is_train;
		dataset = cfg_dataset;

		n_cols = cfg_n_cols; // Will be overwritten based on dataset
		n_trees = cfg_n_trees;
		max_features = cfg_max_features;
		rows_sample = cfg_rows_sample;
		max_depth = cfg_max_depth;
		max_leaves = cfg_max_leaves;
		bootstrap = cfg_bootstrap;
		if (dataset == "year") {
			// Year has hard requirements on train/test examples
			// Dataset has 515K lines (515345 to be exact).  Hard reqs on train/test examples: Train: first 463,715 examples; test: last 51,630 examples
			n_rows = 463715; //hard coded - train_ratio plays no role
			n_inference_rows = test_is_train ? n_rows : 51630; //hard-coded
		} else {
			n_rows = test_is_train ? cfg_n_rows : train_ratio * cfg_n_rows;
			n_inference_rows = test_is_train ? cfg_n_rows : (1.0f -train_ratio) * cfg_n_rows;
		}
		n_bins = cfg_n_bins;
		split_algo = cfg_split_algo;
		min_rows_per_node = cfg_min_rows_per_node;
		bootstrap_features = cfg_bootstrap_features;
		split_criterion = cfg_split_criterion;

		if (dataset == "higgs") {
			n_cols = 28;
		} else  if ((dataset == "airline") || (dataset == "airline_regression")) {
			n_cols = 13;
		} else  if (dataset == "year") {
			n_cols = 90;
		} else {
			cerr << "Invalid dataset " << dataset << endl;
			exit(1);
		}

		ASSERT((split_algo >= 0) && (split_algo < 3), "Unsupported split_algo %d option. Not in [0, 2].", split_algo);

		cout << "Dataset " << dataset << ", train ratio " << train_ratio << " test_is_train " << test_is_train << ", n_rows " << n_rows << ", n_cols " << n_cols << " n_trees " << n_trees << " col_per " << max_features << " row_per " << rows_sample << " max_depth " << max_depth << " max_leaves " << max_leaves  << " bootstrap " << bootstrap << " n_inference_rows " << n_inference_rows << " n_bins " << n_bins << " split_algo " << split_algo << endl;
	}

};

template<typename T>
void solve_classification_problem(RF_inputs & params);

template<typename T>
void solve_regression_problem(RF_inputs & params);

int main(int argc, char **argv) {

	/* Command line args:
		- # rows
	   	- # cols (fixed per dataset)
		- # trees
		- col_per
		- row_per
		- train_ratio (e.g., 0.8 will use 80% of rows for training and 20% for testing)
		- max_depth
		- max_leaves
		- bootstrap
		- test_is_train (otherwise 80% of rows is used for training and 20% for testing)
		- n_bins
		- dataset name
		- split_algo
		- bootstrap_features
		- split_criterion
	*/

	const int expected_args_cnt = 15;
	if (argc != expected_args_cnt) {
		cout << "Error! " << expected_args_cnt - 1 << " args are needed\n";
		return 0;
	}
	RF_inputs params(stoi(argv[1]), stoi(argv[2]), stoi(argv[3]), stof(argv[4]), stof(argv[5]), stof(argv[6]), stoi(argv[7]), stoi(argv[8]), (strcmp(argv[9], "true") == 0), (strcmp(argv[10], "true") == 0), stoi(argv[11]), argv[12], stoi(argv[13]), 2, false, (ML::CRITERION) stoi(argv[14]));

	bool is_regression = (params.dataset == "year") || (params.dataset == "airline_regression");
	if (is_regression) {
		std::cout << "Regression problem\n";
		solve_regression_problem<float>(params);
	} else {
		std::cout << "Classification problem\n";
		solve_classification_problem<float>(params);
	}
	return 0;
}

template<typename T>
void solve_classification_problem(RF_inputs & params) {
	T * input_data;
	int * input_labels;

 	std::map<int, int> labels_map; //unique map of labels to int vals starting from 0

    int input_data_len = params.n_rows * params.n_cols;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream) );
    allocate(input_data, input_data_len);
    allocate(input_labels, params.n_rows);

	std::vector<T> h_input_data, inference_data;
	std::vector<int> h_input_labels, inference_labels;

	// Populate labels and data
	parse_csv<int>(params.dataset, params.n_cols, h_input_data, h_input_labels, params.n_rows, inference_data, inference_labels, params.n_inference_rows, params.test_is_train); //last arg makes test same as training

 	//Preprocess labels
 	ML::preprocess_labels(params.n_rows, h_input_labels, labels_map);
 	int n_unique_labels = labels_map.size();
 	std::cout << "Dataset has " << n_unique_labels << " labels." << std::endl;

	updateDevice(input_data, h_input_data.data(), input_data_len, stream);
	updateDevice(input_labels, h_input_labels.data(), params.n_rows, stream);
	cout << "Finished populating device labels and data\n";

	// Fit input_dataset
	ML::rfClassifier<T> * my_rf;
	ML::cumlHandle handle;
	handle.setStream(stream);


	ML::DecisionTree::DecisionTreeParams tree_params(params.max_depth, params.max_leaves, params.max_features, params.n_bins, params.split_algo, params.min_rows_per_node, params.bootstrap_features, params.split_criterion);
	ML::RF_params rf_params(params.bootstrap, params.bootstrap_features, params.n_trees, params.rows_sample, tree_params);

	my_rf = new typename ML::rfClassifier<T>::rfClassifier(rf_params);
	cout << "Called RF constructor\n";

	float ms;
	TIMEIT_LOOP(ms, 1, fit(handle, my_rf, input_data, params.n_rows, params.n_cols, input_labels, n_unique_labels));

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaStreamDestroy(stream));


	my_rf->print_rf_detailed();
	cout << "Planted the random forest in " << ms << " ms, " << ms /1000.0 << " s." << endl;

	std::vector<int> predicted_labels;
	predicted_labels.resize(params.n_inference_rows);

 	ML::postprocess_labels(params.n_rows, h_input_labels, labels_map);
 	ML::preprocess_labels(params.n_inference_rows, inference_labels, labels_map); //use same map as labels

	cout << "Will start testing\n";
	ML::RF_metrics metrics = cross_validate(handle, my_rf, inference_data.data(), inference_labels.data(), params.n_inference_rows, params.n_cols, predicted_labels.data(), false);
	metrics.print();
	ML::postprocess_labels(params.n_inference_rows, inference_labels, labels_map);

	cout << "Free memory\n";
	CUDA_CHECK(cudaFree(input_data));
	CUDA_CHECK(cudaFree(input_labels));
	delete my_rf;
}

template<typename T>
void solve_regression_problem(RF_inputs & params) {
	T * input_data;
	T * input_labels;

    int input_data_len = params.n_rows * params.n_cols;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream) );
    allocate(input_data, input_data_len);
    allocate(input_labels, params.n_rows);

	std::vector<T> h_input_data, inference_data;
	std::vector<T> h_input_labels, inference_labels;

	// Populate labels and data
	parse_csv<T>(params.dataset, params.n_cols, h_input_data, h_input_labels, params.n_rows, inference_data, inference_labels, params.n_inference_rows, params.test_is_train); //last arg makes test same as training

	updateDevice(input_data, h_input_data.data(), input_data_len, stream);
	updateDevice(input_labels, h_input_labels.data(), params.n_rows, stream);
	cout << "Finished populating device labels and data\n";

	// Fit input_dataset
	ML::rfRegressor<T> * my_rf;
	ML::cumlHandle handle;
	handle.setStream(stream);


	ML::DecisionTree::DecisionTreeParams tree_params(params.max_depth, params.max_leaves, params.max_features, params.n_bins, params.split_algo, params.min_rows_per_node, params.bootstrap_features, params.split_criterion);
	ML::RF_params rf_params(params.bootstrap, params.bootstrap_features, params.n_trees, params.rows_sample, tree_params);

	my_rf = new typename ML::rfRegressor<T>::rfRegressor(rf_params);
	cout << "Called RF constructor\n";

	float ms;
	TIMEIT_LOOP(ms, 1, fit(handle, my_rf, input_data, params.n_rows, params.n_cols, input_labels));

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaStreamDestroy(stream));


	my_rf->print_rf_detailed();
	cout << "Planted the random forest in " << ms << " ms, " << ms /1000.0 << " s." << endl;

	std::vector<T> predicted_labels;
	predicted_labels.resize(params.n_inference_rows);

	cout << "Will start testing\n";
	ML::RF_metrics metrics = cross_validate(handle, my_rf, inference_data.data(), inference_labels.data(), params.n_inference_rows, params.n_cols, predicted_labels.data(), false);
	metrics.print();

	cout << "Free memory\n";
	CUDA_CHECK(cudaFree(input_data));
	CUDA_CHECK(cudaFree(input_labels));
	delete my_rf;
}

