/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <random>

namespace MLCommon {

typedef enum { MAX, SUM } AggregationType;

/*
 * Fills in data and labels with an easy-to-predict classification dataset.
 * Both vectors will be resized to hold datasets of size nrows * ncols.
 * Data will be laid out in row-major format.
 * The labels are computed as max(...) or sum(all_informative_cols), scaled to
 * map to n_classes integer outputs.
 */
template <typename T>
void makeClassificationDataHost(std::vector<T>& data, std::vector<int>& labels,
                                unsigned n_samples, unsigned n_features,
                                unsigned n_classes, unsigned n_informative = 2,
                                unsigned n_redundant = 2, int rand_seed = 42,
                                AggregationType agg_type = MAX) {
  ASSERT(n_informative + n_redundant <= n_features,
         "informative + redundant cannot exceed cols");
  data.resize(n_samples * n_features);
  labels.resize(n_samples);

  srand(rand_seed);
  T agg_relevant_vals = 10.0 + n_classes;  // Data pattern respects this max
  if (agg_type == SUM) {
    agg_relevant_vals *= n_informative;
  }

  // first NI columns are informative, next NR are identical, rest are junk
  for (int i = 0; i < n_samples; i++) {
    T max_relevant_row = agg_type == MAX ? -1e6 : 0.0;

    for (int j = 0; j < n_features; j++) {
      if (j < n_informative) {
        // Totally arbitrary data pattern that spreads out a bit
        // This differs from sklearn's random pattern
        T val = 10.0 * ((i + 1) / (float)n_samples) + ((i + j) % n_classes);
        if (agg_type == MAX) {
          max_relevant_row = max(max_relevant_row, val);
        } else if (agg_type == SUM) {
          max_relevant_row += val;
        }
        data[(j * n_samples) + i] = val;
      } else if (j < n_informative + n_redundant) {
        // Trivial combination of two adjacent informatives
        unsigned col1 = j % n_informative, col2 = (j + 1) % n_informative;
        data[(j * n_samples) + i] =
          data[(col1 * n_samples) + i] + data[(col2 * n_samples) + i];
      } else {
        // Totally junk data (irrelevant distractors)
        data[(j * n_samples) + i] = 10.0 * ((rand() / (float)RAND_MAX) - 0.5);
      }
    }

    labels[i] = (int)n_classes * (max_relevant_row / agg_relevant_vals);
  }
}

template <typename T>
void tranposeHostMatrix(std::vector<T> in, std::vector<T> out, unsigned n_rows,
                        unsigned n_cols) {
  for (unsigned i = 0; i < n_rows; i++) {
    for (unsigned j = 0; j < n_cols; j++) {
      out[i * n_cols + j] = in[j * n_rows + i];
    }
  }
}

/**
 * Returns an X, y pair constructed from an underlying linear model:
 *   y = X*coef + bias + epsilon
 *
 * Where epsilon is a normal random variate with standard deviation
 * of 'noise_sd'.
 *
 * The 'coef' vector is randomly generated from a gaussian, but only
 * the first n_informative elements have nonzero values, so the remainder
 * of the columns are uninformative for y.
 *
 * Inspired by sklearn.datasets.make_regression, but the approach is
 * NOT identical.
 */
template <typename T>
void makeRegressionDataHost(std::vector<T>& X, std::vector<T>& y,
                            std::vector<T>& coeff, unsigned n_samples,
                            unsigned n_features, unsigned n_informative,
                            T noise_sd = 0.0, T bias = 0.0,
                            int rand_seed = 42) {
  std::default_random_engine rng(rand_seed);
  std::normal_distribution<float> data_distribution(0.0, 3.0);
  std::normal_distribution<float> eps_distribution(0.0, noise_sd);

  ASSERT(n_informative <= n_features,
         "informative may not exceed total features");
  X.resize(n_features * n_samples);
  y.resize(n_samples);
  coeff.resize(n_features);

  // Generate random coefficients
  for (int i = 0; i < n_features; i++) {
    if (i < n_informative) {
      coeff[i] = data_distribution(rng);
    } else {
      coeff[i] = 0.0;
    }
  }
  for (int i = 0; i < n_samples; i++) {
    for (int j = 0; j < n_features; j++) {
      X[j + i * n_features] = data_distribution(rng);
    }
    // Compute X[i]*beta + bias
    T tmp = bias;
    for (int j = 0; j < n_features; j++) {
      tmp += X[j + i * n_features] * coeff[j] + bias;
    }
    tmp += eps_distribution(rng);  // epsilon
    y[i] = tmp;
  }
}

// TODO: Add GPU-based make_blobs, make_regression

}  // namespace MLCommon
