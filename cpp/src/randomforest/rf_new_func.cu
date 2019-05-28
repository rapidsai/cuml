#pragma once
#include "randomforest.cu"

namespace ML {

void fit_f32(const cumlHandle& user_handle, const rfClassifier<float> * rf_classifier, const float * input, int n_rows, int n_cols, int * labels, int n_unique_labels){
    fit(user_handle, input, n_rows, n_cols, labels, n_unique_labels);
}


void fit_f64(const cumlHandle& user_handle, const rfClassifier<double> * rf_classifier, const double  * input, int n_rows, int n_cols, int * labels, int n_unique_labels){
    fit(user_handle, input, n_rows, n_cols, labels, n_unique_labels);
}


void predict_f32(const cumlHandle& user_handle, const rfClassifier<float> * rf_classifier, const float * input, int n_rows, int n_cols, int * predictions, bool verbose=false){
    predict(user_handle, input, n_rows, n_cols, predictions, verbose);
}

void predict_f64(cuser_handle, const rfClassifier<double> * rf_classifier, const double * input, int n_rows, int n_cols, int * predictions, bool verbose=false){
    predict(user_handle, input, n_rows, n_cols, predictions, verbose);
}

};
