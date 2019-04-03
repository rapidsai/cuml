#include "kalman.h"
#include "batched_kalman.h"

using std::vector;

void batched_kalman_filter(const vector<double*>& ptr_ys_b, const vector<int>& ys_len,
                           const vector<double*>& ptr_Zb,
                           const vector<double*>& ptr_Rb,
                           const vector<double*>& ptr_Tb,
                           int r,
                           vector<double*>& ptr_vs_b,
                           vector<double*>& ptr_Fs_b,
                           vector<double>& ptr_loglike_b,
                           vector<double>& ptr_sigma2_b) {

  // just use single kalman for now
  const size_t num_batches = ptr_ys_b.size();
  for(int i=0; i<num_batches; i++) {
    kalman_filter(ptr_ys_b[i], ys_len[i], ptr_Zb[i], ptr_Rb[i], ptr_Tb[i], r,
                  ptr_vs_b[0], ptr_Fs_b[0], &ptr_loglike_b[i], &ptr_sigma2_b[i]);
  }

}
