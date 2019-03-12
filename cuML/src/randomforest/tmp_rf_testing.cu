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

#define HIGGS_DATA "/gpfs/fs1/myrtop/rapids_repos/HIGGS.csv"
//#define HIGGS_COLS 28 // + 1 for label (it's the first per csv line)
//#define TRAIN_RATIO 0.8

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

void parse_csv(int n_cols, std::vector<float> & data, std::vector<int> & labels, int train_cnt,
			   std::vector<float> & test_data, std::vector<int> & test_labels, int test_cnt, bool test_is_train) {

	cout << "train_cnt " << train_cnt << " test_cnt " << test_cnt << endl;
	ifstream myfile;
	myfile.open(HIGGS_DATA);
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
					data[counter + col * train_cnt] = row[col + 1]; // 1st column is label; train data should be col major
					if (test_is_train) 
						test_data[counter*n_cols + col] = row[col + 1]; // test data should be row major
				} else if (!test_is_train)
					test_data[(counter - train_cnt)*n_cols + col] = row[col + 1]; // test data should be row major
			}
			if (counter < train_cnt)  {
				labels[counter] = (int) row[0];
				if (test_is_train) test_labels[counter] = labels[counter];
			} else if (!test_is_train)
				test_labels[counter - train_cnt] = (int) row[0];
			counter++;
		}
	cout << "Lines processed " << counter << endl;  
	myfile.close();

}
	

struct RF_inputs {
	int n_rows, n_cols, n_inference_rows;
	int n_trees, max_depth, max_leaves, n_bins;
	float max_features, rows_sample, train_ratio;
	bool bootstrap, test_is_train;

	RF_inputs(int cfg_n_rows, int cfg_n_cols, int cfg_n_trees, float cfg_max_features, 
			  	float cfg_rows_sample, float cfg_train_ratio, int cfg_max_depth, 
				int cfg_max_leaves, bool cfg_bootstrap, bool cfg_test_is_train, int cfg_n_bins) {

		train_ratio = cfg_train_ratio;
		test_is_train = cfg_test_is_train;

		n_rows = test_is_train ? cfg_n_rows : train_ratio * cfg_n_rows;
	 	n_cols = cfg_n_cols;
		n_trees = cfg_n_trees;
		max_features = cfg_max_features;
		rows_sample = cfg_rows_sample;
		max_depth = cfg_max_depth;
		max_leaves = cfg_max_leaves;
		bootstrap = cfg_bootstrap;
		n_inference_rows = test_is_train ? cfg_n_rows : (1.0f -train_ratio) * cfg_n_rows;
		n_bins = cfg_n_bins;
		cout << "Train ratio " << train_ratio << " test_is_train " << test_is_train << ", n_rows " << n_rows << ", n_cols " << n_cols << " n_trees " << n_trees << " col_per " << max_features << " row_per " << rows_sample << " max_depth " << max_depth << " max_leaves " << max_leaves  << " bootstrap " << bootstrap << " n_inference_rows " << n_inference_rows << " n_bins " << n_bins << endl;
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
	*/

	std::map<int, int> labels_map; //unique map of labels to int vals starting from 0

	if (argc != 12) {
		cout << "Error! 11 args are needed\n";
		return 0;
	}
	RF_inputs params(stoi(argv[1]), stoi(argv[2]), stoi(argv[3]), stof(argv[4]), stof(argv[5]), stof(argv[6]), stoi(argv[7]), stoi(argv[8]), (strcmp(argv[9], "true") == 0), (strcmp(argv[10], "true") == 0), stoi(argv[11]));

	float * higgs_data;
	int * higgs_labels;

    int higgs_data_len = params.n_rows * params.n_cols;
    allocate(higgs_data, higgs_data_len);
    allocate(higgs_labels, params.n_rows);

	std::vector<float> h_higgs_data, inference_data;
	std::vector<int> h_higgs_labels, inference_labels;

	// Populate labels and data
	parse_csv(params.n_cols, h_higgs_data, h_higgs_labels, params.n_rows, inference_data, inference_labels, params.n_inference_rows, params.test_is_train); //last arg makes test same as training

	//Preprocess labels 
	ML::preprocess_labels(params.n_rows, h_higgs_labels, labels_map);

	updateDevice(higgs_data, h_higgs_data.data(), higgs_data_len);
	updateDevice(higgs_labels, h_higgs_labels.data(), params.n_rows);
	cout << "Finished populating device labels and data\n";

	// Classify higgs_dataset
	ML::rfClassifier * rf_classifier;
 	rf_classifier = new ML::rfClassifier::rfClassifier(params.n_trees, params.bootstrap, params.max_depth, params.max_leaves, 0, params.n_bins, params.rows_sample, params.max_features);
	cout << "Called RF constructor\n";
	//rf_classifier->fit(higgs_data, params.n_rows, params.n_cols, higgs_labels);

	float ms;
	TIMEIT_LOOP(ms, 1, rf_classifier->fit(higgs_data, params.n_rows, params.n_cols, higgs_labels));

	rf_classifier->print_rf_detailed();
	cout << "Planted the random forest in " << ms << " ms, " << ms /1000.0 << " s." << endl;

	//Predict w/ test dataset
	/*predictions = rf_classifier->predict(inference_data.data(), params.n_inference_rows, params.n_cols, false);
	for (int i = 0; i < params.n_inference_rows; i++) {
		std::cout << "Random forest predicted " << predictions[i] << std::endl;
	}*/

	ML::postprocess_labels(params.n_rows, h_higgs_labels, labels_map);
	ML::preprocess_labels(params.n_inference_rows, inference_labels, labels_map); //use same map as labels

	cout << "Will start testing\n";
	ML::RF_metrics metrics = rf_classifier->cross_validate(inference_data.data(), inference_labels.data(), params.n_inference_rows, params.n_cols, false);
	metrics.print();

	ML::postprocess_labels(params.n_inference_rows, inference_labels, labels_map); 

	cout << "Free memory\n";
	CUDA_CHECK(cudaFree(higgs_data));
	CUDA_CHECK(cudaFree(higgs_labels));
	delete rf_classifier;


#if 0
	RF_inputs params(4, 2, 3, 1.0f, 1.0f, 4, 8, -1, true);
	float * data;
    int * labels, * predictions;

	ML::rfClassifier * rf_classifier;

    int data_len = params.n_rows * params.n_cols;
    allocate(data, data_len);
    allocate(labels, params.n_rows);

	// Populate data (assume Col major)
	std::vector<float> data_h = {30.0f, 1.0f, 2.0f, 0.0f, 10.0f, 20.0f, 10.0f, 40.0f};
	data_h.resize(data_len);
    updateDevice(data, data_h.data(), data_len);

	// Populate labels
	std::vector<int> labels_h = {0, 1, 0, 4};
	labels_h.resize(params.n_rows);
	updateDevice(labels, labels_h.data(), params.n_rows);

 	rf_classifier = new ML::rfClassifier::rfClassifier(params.n_trees, params.bootstrap, 0, 0);
	rf_classifier->fit(data, params.n_rows, params.n_cols, labels, params.n_trees, params.max_features, params.rows_sample);

	int inference_data_len = params.n_inference_rows * params.n_cols;
	std::vector<float> inference_data_h = {30.0f, 10.0f, 1.0f, 20.0f, 2.0f, 10.0f, 0.0f, 40.0f};
	inference_data_h.resize(inference_data_len);

	predictions = rf_classifier->predict(inference_data_h.data(), params.n_inference_rows, params.n_cols, false);
	for (int i = 0; i < params.n_inference_rows; i++) {
		std::cout << "Random forest predicted " << predictions[i] << std::endl;
	}

	rf_classifier->cross_validate(inference_data_h.data(), labels_h.data(), params.n_inference_rows, params.n_cols, false);

	CUDA_CHECK(cudaFree(labels));
	CUDA_CHECK(cudaFree(data));
	delete predictions;

#endif
	return 0;
}

