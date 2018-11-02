/*!
 * Copyright 2017-2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once
#include <vector>
// #include "cblas.h"
#include "logger.h"

template<typename T>
void self_dot(std::vector<T> array_in, int n, int dim, std::vector<T>& dots) {
	for (int pt = 0; pt < n; pt++) {
		T sum = 0.0;
		for (int i = 0; i < dim; i++) {
			sum += array_in[pt * dim + i] * array_in[pt * dim + i];
		}
		dots[pt] = sum;
	}
}

// void compute_distances(std::vector<double> data_in,
//                        std::vector<double> centroids_in,
//                        std::vector<double> &pairwise_distances,
//                        int n, int dim, int k) {
//   std::vector<double> data_dots(n);
//   std::vector<double> centroid_dots(k);
//   self_dot(data_in, n, dim, data_dots);
//   self_dot(centroids_in, k, dim, centroid_dots);
//   for (int nn=0; nn<n; nn++)
//     for (int c=0; c<k; c++) {
//       pairwise_distances[nn*k+c] = data_dots[nn] +
//           centroid_dots[c];
//     }
//   double alpha = -2.0;
//   double beta = 1.0;
//   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, k,
//               dim, alpha, &data_in[0], dim, &centroids_in[0], dim,
//               beta, &pairwise_distances[0], k);
// }

// void compute_distances(std::vector<float> data_in,
//                        std::vector<float> centroids_in,
//                        std::vector<float> &pairwise_distances,
//                        int n, int dim, int k) {
//   std::vector<float> data_dots(n);
//   std::vector<float> centroid_dots(k);
//   self_dot(data_in, n, dim, data_dots);
//   self_dot(centroids_in, k, dim, centroid_dots);
//   for (int nn=0; nn<n; nn++)
//     for (int c=0; c<k; c++) {
//       pairwise_distances[nn*k+c] = data_dots[nn] +
//           centroid_dots[c];
//     }
//   float alpha = -2.0;
//   float beta = 1.0;
//   cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, k,
//               dim, alpha, &data_in[0], dim, &centroids_in[0], dim,
//               beta, &pairwise_distances[0], k);
// }
