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
#define HIGGS_COLS 28 // + 1 for label (it's the first per csv line)
#define HIGGS_ROWS 1000
#define TRAIN_RATIO 0.8

using namespace MLCommon;
using namespace std;

void parse_csv(std::vector<float> & data, std::vector<int> & labels, int train_cnt,
			   std::vector<float> & test_data, std::vector<int> & test_labels, int test_cnt, bool test_is_train) {

	cout << "train_cnt " << train_cnt << " test_cnt " << test_cnt << endl;
	ifstream myfile;
	myfile.open(HIGGS_DATA);
	string line;

	int counter = 0;
	data.resize(train_cnt * HIGGS_COLS);
	labels.resize(train_cnt);

	test_data.resize(test_cnt * HIGGS_COLS);
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
			for (int col = 0; col < HIGGS_COLS; col++) {
				if (counter < train_cnt)  {
					data[counter + col * train_cnt] = row[col + 1]; // 1st column is label; train data should be col major
					if (test_is_train) 
						test_data[counter*HIGGS_COLS + col] = row[col + 1]; // test data should be row major
				} else if (!test_is_train)
					test_data[(counter - train_cnt)*HIGGS_COLS + col] = row[col + 1]; // test data should be row major
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
	int n_rows;
	int n_cols;
	int n_trees;
	float max_features;
	float rows_sample;
	int n_inference_rows;
	int max_depth;
	int max_leaves;
	bool bootstrap;

	RF_inputs(int cfg_n_rows, int cfg_n_cols, int cfg_n_trees, float cfg_max_features, float cfg_rows_sample, int cfg_n_inference_rows, int cfg_max_depth, int cfg_max_leaves, bool cfg_bootstrap) {
		 n_rows = cfg_n_rows;
		 n_cols = cfg_n_cols;
		 n_trees = cfg_n_trees;
		 max_features = cfg_max_features;
		 rows_sample = cfg_rows_sample;
		 n_inference_rows = cfg_n_inference_rows;
		 max_depth = cfg_max_depth;
		 max_leaves = cfg_max_leaves;
		 bootstrap = cfg_bootstrap;
	}
};

int main() {

	RF_inputs higgs_params(HIGGS_ROWS, HIGGS_COLS, 10, 1.0f, 1.0f, HIGGS_ROWS, 8, -1, true);
	float * higgs_data;
	int * higgs_labels;

	ML::rfClassifier * rf_classifier;
	bool test_is_train = true;

	int train_rows = TRAIN_RATIO * higgs_params.n_rows;
	int inference_rows = test_is_train ? train_rows : (1.0f - TRAIN_RATIO) * higgs_params.n_rows;
    int higgs_data_len = train_rows * higgs_params.n_cols;
    allocate(higgs_data, higgs_data_len);
    allocate(higgs_labels, train_rows);

	std::vector<float> h_higgs_data, inference_data;
	std::vector<int> h_higgs_labels, inference_labels;

	// Populate labels and data
	parse_csv(h_higgs_data, h_higgs_labels, train_rows, inference_data, inference_labels, inference_rows, test_is_train); //last arg makes test same as training
	updateDevice(higgs_data, h_higgs_data.data(), higgs_data_len);
	updateDevice(higgs_labels, h_higgs_labels.data(), train_rows);
	cout << "Finished populating device labels and data\n";

	// Classify higgs_dataset
 	rf_classifier = new ML::rfClassifier::rfClassifier(higgs_params.n_trees, higgs_params.bootstrap, 0, 0);
	cout << "Called RF constructor\n";
	rf_classifier->fit(higgs_data, train_rows, higgs_params.n_cols, higgs_labels, higgs_params.n_trees, higgs_params.max_features, higgs_params.rows_sample);
	cout << "Planted the random forest\n";

	//Predict w/ test dataset
	/*predictions = rf_classifier->predict(inference_data.data(), inference_rows, higgs_params.n_cols, false);
	for (int i = 0; i < inference_rows; i++) {
		std::cout << "Random forest predicted " << predictions[i] << std::endl;
	}*/

	cout << "Will start testing\n";
	rf_classifier->cross_validate(inference_data.data(), inference_labels.data(), inference_rows, higgs_params.n_cols, false);

	cout << "Free memory\n";
	CUDA_CHECK(cudaFree(higgs_data));
	CUDA_CHECK(cudaFree(higgs_labels));
	

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

