#ifndef ARIMA_KALMAN_H
#define ARIMA_KALMAN_H
void kalman_filter(double* ptr_ys, int ys_len, double* ptr_Z, double* ptr_R, double* ptr_T, int r,
                   double* ptr_vs, double* ptr_Fs, double* ptr_loglike, double* ptr_sigma2);
#endif
