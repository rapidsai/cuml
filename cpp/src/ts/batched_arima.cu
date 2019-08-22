
#include <cstdio>
#include <tuple>
#include <vector>

#include "batched_arima.h"
#include "batched_kalman.h"

#include <common/nvtx.hpp>

// y - series to fit: (nobs, num_bathces) (nobs-major, i.e., column major)
// num_batches - number of time series
// order - ARIMA order (p: number of ar-parameters, d: difference parameter, q: number of ma-parameters)
// params - parameters to evaluate: [mu, ar.., ma.., ...]
// trans - run `jones_transform` on params;
// returns: vector of log likelihood, one for each series (size: num_batches)
void batched_loglike(double* y, int num_batches, int nobs, int p, int d, int q,
                     double* params, std::vector<double>& loglike_b,
                     bool trans) {
  using std::get;
  using std::vector;

  ML::PUSH_RANGE(__FUNCTION__);

  // unpack parameters
  vector<double> mu(num_batches);
  vector<double> ar(num_batches * p);
  vector<double> ma(num_batches * q);

  int N = (p + d + q);

  for (int ib = 0; ib < num_batches; ib++) {
    if (d > 0) {
      mu.at(ib) = params[ib * N];
    }
    for (int ip = 0; ip < p; ip++) {
      ar.at(p * ib + ip) = params[ib * N + d + ip];
    }
    for (int iq = 0; iq < q; iq++) {
      ma.at(q * ib + iq) = params[ib * N + d + p + iq];
    }
  }

  // Transformed parameters
  vector<double> Tar;
  vector<double> Tma;

  if (trans) {
    batched_jones_transform(p, q, num_batches, false, ar, ma, Tar, Tma);
  } else {
    // non-transformed case: just use original parameters
    Tar = ar;
    Tma = ma;
  }

  std::vector<std::vector<double>> vs_b;
  if (d == 0) {
    // no diff
    batched_kalman_filter(y, nobs, Tar, Tma, p, q, num_batches, loglike_b,
                          vs_b);
  } else if (d == 1) {
    // diff and center (with `mu`):
    std::vector<double> y_diff(num_batches * (nobs - 1));
    for (int ib = 0; ib < num_batches; ib++) {
      auto mu_ib = mu[ib];
      for (int i = 0; i < nobs - 1; i++) {
        y_diff[ib * (nobs - 1) + i] =
          (y[ib * nobs + i + 1] - y[ib * nobs + i]) - mu_ib;
      }
    }

    batched_kalman_filter(y_diff.data(), nobs - d, Tar, Tma, p, q, num_batches,
                          loglike_b, vs_b);
  } else {
    throw std::runtime_error("Not supported difference parameter: d=0, 1");
  }
  ML::POP_RANGE();
}
