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


#include "randomforest.h"
#include "../../../thirdparty/cuml/ml-prims/src/utils.h"
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

void parse_csv(string dataset_name, int n_cols, std::vector<float> & data, std::vector<int> & labels, int train_cnt,
			   std::vector<float> & test_data, std::vector<int> & test_labels, int test_cnt, bool test_is_train) {

	string data_file;
	int col_offset = 0;
	int label_id =  0; // column that is the label (i.e., target feature)
	if (dataset_name == "higgs") {
		data_file = "/gpfs/fs1/myrtop/rapids_repos/HIGGS.csv";
		col_offset = 1; //because the first column in higgs is the label
		label_id = 0;
	} else if (dataset_name == "airline") {
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

			// In airline, the row[label_id] > 0 convert the delay to categorical data.
			// No effect for higgs that has 0, 1 labels.
			if (counter < train_cnt)  {
				labels[counter] = (int) (row[label_id] > 0);
				if (test_is_train) test_labels[counter] = labels[counter];
			} else if (!test_is_train)
				test_labels[counter - train_cnt] = (int) (row[label_id] > 0);
			counter++;
		}
	cout << "Lines processed " << counter << endl;  
	myfile.close();

}
	

struct RF_inputs {
	int n_rows, n_cols, n_inference_rows;
	int n_trees, max_depth, max_leaves, n_bins, split_algo;
	float max_features, rows_sample, train_ratio;
	bool bootstrap, test_is_train;
	string dataset;

	RF_inputs(int cfg_n_rows, int cfg_n_cols, int cfg_n_trees, float cfg_max_features, 
			  	float cfg_rows_sample, float cfg_train_ratio, int cfg_max_depth, 
				int cfg_max_leaves, bool cfg_bootstrap, bool cfg_test_is_train, int cfg_n_bins,
				string cfg_dataset, int cfg_split_algo) {

		train_ratio = cfg_train_ratio;
		test_is_train = cfg_test_is_train;

		n_rows = test_is_train ? cfg_n_rows : train_ratio * cfg_n_rows;
	 	n_cols = cfg_n_cols; // Will be overwriten based on dataset 
		n_trees = cfg_n_trees;
		max_features = cfg_max_features;
		rows_sample = cfg_rows_sample;
		max_depth = cfg_max_depth;
		max_leaves = cfg_max_leaves;
		bootstrap = cfg_bootstrap;
		n_inference_rows = test_is_train ? cfg_n_rows : (1.0f -train_ratio) * cfg_n_rows;
		n_bins = cfg_n_bins;
		dataset = cfg_dataset;
		split_algo = cfg_split_algo;

		if (dataset == "higgs") {
			n_cols = 28;
		} else  if (dataset == "airline") {
			n_cols = 13;
		} else {
			cerr << "Invalid dataset " << dataset << endl;
			exit(1);
		}

		ASSERT((split_algo >= 0) && (split_algo < 3), "Unsupported split_algo %d option. Not in [0, 2].", split_algo);

		cout << "Dataset " << dataset << ", train ratio " << train_ratio << " test_is_train " << test_is_train << ", n_rows " << n_rows << ", n_cols " << n_cols << " n_trees " << n_trees << " col_per " << max_features << " row_per " << rows_sample << " max_depth " << max_depth << " max_leaves " << max_leaves  << " bootstrap " << bootstrap << " n_inference_rows " << n_inference_rows << " n_bins " << n_bins << " split_algo " << split_algo << endl;
	}

};

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
	*/

	std::map<int, int> labels_map; //unique map of labels to int vals starting from 0

	if (argc != 14) {
		cout << "Error! 13 args are needed\n";
		return 0;
	}
	RF_inputs params(stoi(argv[1]), stoi(argv[2]), stoi(argv[3]), stof(argv[4]), stof(argv[5]), stof(argv[6]), stoi(argv[7]), stoi(argv[8]), (strcmp(argv[9], "true") == 0), (strcmp(argv[10], "true") == 0), stoi(argv[11]), argv[12], stoi(argv[13]));

	float * input_data;
	int * input_labels;

    int input_data_len = params.n_rows * params.n_cols;
    allocate(input_data, input_data_len);
    allocate(input_labels, params.n_rows);

	std::vector<float> h_input_data, inference_data;
	std::vector<int> h_input_labels, inference_labels;

	// Populate labels and data
	parse_csv(params.dataset, params.n_cols, h_input_data, h_input_labels, params.n_rows, inference_data, inference_labels, params.n_inference_rows, params.test_is_train); //last arg makes test same as training

	//Preprocess labels 
	ML::preprocess_labels(params.n_rows, h_input_labels, labels_map);
	int n_unique_labels = labels_map.size();
	std::cout << "Dataset has " << n_unique_labels << " labels." << std::endl;

	updateDevice(input_data, h_input_data.data(), input_data_len);
	updateDevice(input_labels, h_input_labels.data(), params.n_rows);
	cout << "Finished populating device labels and data\n";

	// Classify input_dataset
	ML::rfClassifier<float> * rf_classifier;
 	rf_classifier = new ML::rfClassifier<float>::rfClassifier(params.n_trees, params.bootstrap, params.max_depth, params.max_leaves, 0, params.n_bins, params.rows_sample, params.max_features, params.split_algo);
	cout << "Called RF constructor\n";
	//rf_classifier->fit(input_data, params.n_rows, params.n_cols, input_labels);

	float ms;
	TIMEIT_LOOP(ms, 1, rf_classifier->fit(input_data, params.n_rows, params.n_cols, input_labels, n_unique_labels));

	rf_classifier->print_rf_detailed();
	cout << "Planted the random forest in " << ms << " ms, " << ms /1000.0 << " s." << endl;

	//Predict w/ test dataset
	/*predictions = rf_classifier->predict(inference_data.data(), params.n_inference_rows, params.n_cols, false);
	for (int i = 0; i < params.n_inference_rows; i++) {
		std::cout << "Random forest predicted " << predictions[i] << std::endl;
	}*/

	ML::postprocess_labels(params.n_rows, h_input_labels, labels_map);
	ML::preprocess_labels(params.n_inference_rows, inference_labels, labels_map); //use same map as labels

	cout << "Will start testing\n";
	ML::RF_metrics metrics = rf_classifier->cross_validate(inference_data.data(), inference_labels.data(), params.n_inference_rows, params.n_cols, false);
	metrics.print();

	ML::postprocess_labels(params.n_inference_rows, inference_labels, labels_map); 

	cout << "Free memory\n";
	CUDA_CHECK(cudaFree(input_data));
	CUDA_CHECK(cudaFree(input_labels));
	delete rf_classifier;

	return 0;
}

