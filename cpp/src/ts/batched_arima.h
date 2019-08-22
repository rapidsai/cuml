#include <vector>

void batched_loglike(double* y, int num_batches, int nobs, int p, int d, int q,
                     double* params, std::vector<double>& loglike,
                     bool trans = true);
