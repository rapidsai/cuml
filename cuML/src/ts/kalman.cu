#include "kalman.h"

#include <iostream>

#include "Eigen/Dense"
#include <stdexcept>

using namespace Eigen;

using MatrixT = Matrix<double, Dynamic, Dynamic, RowMajor>;
using VectorT = VectorXd;
using MapMatrixT = Map<MatrixT>;
using MapVectorT = Map<VectorXd>;

void kalman_filter(double* ptr_ys, int ys_len, double* ptr_Z, double* ptr_R, double* ptr_T, int r,
                   double* ptr_vs, double* ptr_Fs, double* ptr_loglike, double* ptr_sigma2) {
  
  int nobs = ys_len;

  MapVectorT ys(ptr_ys, ys_len);
  MapMatrixT Z(ptr_Z, 1, r);
  MapMatrixT R(ptr_R, r, 1);
  MapMatrixT T(ptr_T, r, r);

  // return results
  MapVectorT vs(ptr_vs, nobs);
  MapVectorT Fs(ptr_Fs, nobs);

  // use a single kalman iteration as covariance (P) initialization
  MatrixT P = T * T.transpose() - T * Z.transpose() * Z * T.transpose() + R * R.transpose();
  MatrixT alpha = MatrixT::Zero(r, 1);

  double loglikelihood = 0.0;

  for(int it=0; it<nobs; it++) {
    
    vs[it] = ys[it] - alpha(0,0);
    Fs[it] = P(0,0);

    if(Fs[it] < 0) {
      std::cout << "P=" << P << "\n";
      throw std::runtime_error("ERROR: F < 0");
    }

    MatrixT K = 1.0/Fs[it] * (T * P * Z.transpose());
    alpha = T*alpha + K*vs[it];
    MatrixT L = T - K*Z;
    P = T * P * L.transpose() + R * R.transpose();

    loglikelihood += std::log(Fs[it]);

  }

  double sigma2 = ((vs.array().pow(2.0)).array() / Fs.array()).mean();
  if(sigma2 < 0) {
    throw std::runtime_error("ERROR: Sigma2 < 0");
  }
  double loglike = -.5 * (loglikelihood + nobs * std::log(sigma2));
  loglike -= nobs / 2. * (std::log(2 * M_PI) + 1);

  // return results
  *ptr_loglike = loglike;
  *ptr_sigma2 = sigma2;
}
