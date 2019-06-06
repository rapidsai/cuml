#pragma once
#include "decisiontree/decisiontree_rf_params.h"

namespace ML {

struct RF_metrics {
	float accuracy;

	RF_metrics(float cfg_accuracy);
	void print();
};

enum RF_type {
	CLASSIFICATION, REGRESSION,
};

struct RF_params {
	/**
	 * Control bootstrapping. If set, each tree in the forest is built on a bootstrapped sample with replacement.
	 * If false, sampling without replacement is done.
	 */
	bool bootstrap = true;
	/**
	 * Control bootstrapping for features. If features are drawn with or without replacement
	 */
	bool bootstrap_features = false;
	/**
	 * Number of decision trees in the random forest.
	 */
	int n_trees;
	/**
	 * Ratio of dataset rows used while fitting each tree.
	 */
	float rows_sample = 1.0f;
	/**
	 * Decision tree traingin hyper parameter struct.
	 */
	DecisionTree::DecisionTreeParams tree_params;
	RF_params();
	RF_params(int cfg_n_trees);
	RF_params(bool cfg_bootstrap, bool cfg_bootstrap_features, int cfg_n_trees, float cfg_rows_sample);
	RF_params(bool cfg_bootstrap, bool cfg_bootstrap_features, int cfg_n_trees, float cfg_rows_sample, DecisionTree::DecisionTreeParams cfg_tree_params);
	void validity_check() const;
	void print() const;
};


template<class T>
class rf {
	protected:
		RF_params rf_params;
		int rf_type;
		DecisionTree::DecisionTreeClassifier<T> * trees;

	public:
		rf(RF_params cfg_rf_params, int cfg_rf_type=RF_type::CLASSIFICATION);
		~rf();

		int get_ntrees();
		void print_rf_summary();
		void print_rf_detailed();
};

};
