#ifndef ARIMA_BATCHED_KALMAN_H
#define ARIMA_BATCHED_KALMAN_H

#include <vector>
#include <string>

// reference implementation
void batched_kalman_filter_cpu(const std::vector<double*>& h_ys_b, // { vector size batches, each item size nobs }
                               int nobs,
                               const std::vector<double*>& h_Zb, // { vector size batches, each item size Zb }
                               const std::vector<double*>& h_Rb, // { vector size batches, each item size Rb }
                               const std::vector<double*>& h_Tb, // { vector size batches, each item size Tb }
                               int r,
                               std::vector<double>& h_loglike_b,
                               std::vector<std::vector<double>>& h_vs_b,
                               bool initP_with_kalman_iterations=true);
                               

void batched_kalman_filter(double* d_ys_b,
                           int nobs,
                           const std::vector<double*>& h_Zb, // { vector size batches, each item size Zb }
                           const std::vector<double*>& h_Rb, // { vector size batches, each item size Rb }
                           const std::vector<double*>& h_Tb, // { vector size batches, each item size Tb }
                           const std::vector<double*>& h_P0b, // { vector size batches, each item size Tb }
                           int r,
                           int num_batches,
                           std::vector<double>& loglike_b,
                           std::vector<std::vector<double>>& h_vs_b,
                           bool initP_with_kalman_iterations=true);


void nvtx_range_push(std::string msg);

void nvtx_range_pop();

#endif
