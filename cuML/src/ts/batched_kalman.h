#ifndef ARIMA_BATCHED_KALMAN_H
#define ARIMA_BATCHED_KALMAN_H

#include <vector>

void batched_kalman_filter(const std::vector<double*>& ptr_ys_b,
                           const std::vector<double*>& ptr_Zb,
                           const std::vector<double*>& ptr_Rb,
                           const std::vector<double*>& ptr_Tb,
                           int r,
                           std::vector<double*>& ptr_vs_b,
                           std::vector<double*>& ptr_Fs_b,
                           std::vector<double>& ptr_loglike_b,
                           std::vector<double>& ptr_sigma2_b);

#endif
