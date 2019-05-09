#ifndef ARIMA_KALMAN_H
#define ARIMA_KALMAN_H
void kalman_filter(double* ptr_ys, int ys_len, double* ptr_Z, double* ptr_R, double* ptr_T, int r,
                   double* ptr_vs, double* ptr_loglike, bool P_init_by_single_iteration=true);
#endif
