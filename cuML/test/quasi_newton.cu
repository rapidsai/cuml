#include "test_utils.h"
#include <glm/glm_c.h>
#include <glm/qn/glm_linear.h>
#include <glm/qn/glm_logistic.h>
#include <glm/qn/glm_softmax.h>
#include <glm/qn/qn.h>
#include <gtest/gtest.h>
#include <linalg/transpose.h>
#include <vector>

namespace ML {
namespace GLM {

using namespace MLCommon;

struct QuasiNewtonTest : ::testing::Test {
  static constexpr int N = 10;
  static constexpr int D = 2;

  const static double *nobptr;
  const static double tol;
  const static double X[N][D];
  cumlHandle cuml_user;
  cudaStream_t stream;
  void SetUp() {
    stream = 0;
    cuml_user.setStream(stream);
  }
  void TearDown() {}
};

const double *QuasiNewtonTest::nobptr = 0;
const double QuasiNewtonTest::tol = 5e-6;
const double QuasiNewtonTest::X[QuasiNewtonTest::N][QuasiNewtonTest::D] = {
    {-0.2047076594847130, 0.4789433380575482},
    {-0.5194387150567381, -0.5557303043474900},
    {1.9657805725027142, 1.3934058329729904},
    {0.0929078767437177, 0.2817461528302025},
    {0.7690225676118387, 1.2464347363862822},
    {1.0071893575830049, -1.2962211091122635},
    {0.2749916334321240, 0.2289128789353159},
    {1.3529168351654497, 0.8864293405915888},
    {-2.0016373096603974, -0.3718425371402544},
    {1.6690253095248706, -0.4385697358355719}};

template <typename T, class Comp>
::testing::AssertionResult
checkParamsEqual(const T *host_weights, const T *host_bias, const T *w,
                 const int C, const int D, const bool fit_intercept,
                 Comp &comp) {
  std::vector<T> w_ref_cm(C * D);
  int idx = 0;
  for (int d = 0; d < D; d++)
    for (int c = 0; c < C; c++) {
      w_ref_cm[idx++] = host_weights[c * D + d];
    }

  SimpleVecOwning<T> w_ref(C * (D + fit_intercept));
  updateDevice(w_ref.data, &w_ref_cm[0], C * D);
  if (fit_intercept) {
    updateDevice(&w_ref.data[C * D], host_bias, C);
  }
  return devArrMatch(w_ref.data, w, w_ref.len, comp);
}

struct InputSpec {
  int n_row;
  int n_col;
  bool fit_intercept;
};

template <class T> struct DevUpload {
  SimpleMatOwning<T> devX;
  SimpleVecOwning<T> devY;
  DevUpload(const InputSpec &inSpec, const T *x, const T *y,
            const cublasHandle_t &cublas)
      : devX(inSpec.n_row, inSpec.n_col), devY(inSpec.n_row) {

    SimpleMatOwning<T> devXtmp(inSpec.n_row, inSpec.n_col);

    updateDevice(devX.data, x, inSpec.n_row * inSpec.n_col);
    updateDevice(devY.data, y, inSpec.n_row);
  }
};

template <typename T, class LossFunction>
T run(LossFunction &loss, DevUpload<T> &devUpload, InputSpec &in, T l1, T l2,
      T *w, SimpleMat<T> &z, const cumlHandle_impl &cuml, int verbosity = 0) {

  int max_iter = 100;
  T grad_tol = 1e-8;
  int linesearch_max_iter = 50;
  int lbfgs_memory = 5;
  int num_iters = 0;

  T fx;
  SimpleVec<T> w0(w, loss.n_param);

  qn_fit<T, LossFunction>(loss, devUpload.devX.data, devUpload.devY.data,
                          z.data, in.n_row, loss.fit_intercept, l1, l2,
                          max_iter, grad_tol, linesearch_max_iter, lbfgs_memory,
                          verbosity, w0.data, &fx, &num_iters, ROW_MAJOR, cuml);

  return fx;
}

template <typename T>
T run_api(int loss_type, int C, bool fit_intercept, DevUpload<T> &devUpload,
          InputSpec &in, T l1, T l2, T *w, SimpleMat<T> &z, int verbosity = 0,
          cudaStream_t stream = 0) {

  int max_iter = 100;
  T grad_tol = 1e-8;
  int linesearch_max_iter = 50;
  int lbfgs_memory = 5;
  int num_iters = 0;

  SimpleVec<T> w0(w, in.n_col + fit_intercept);
  w0.fill(T(0));
  T fx;

  qnFit(devUpload.devX.data, devUpload.devY.data, in.n_row, in.n_col, C,
        fit_intercept, l1, l2, max_iter, grad_tol, linesearch_max_iter,
        lbfgs_memory, verbosity, w, &fx, &num_iters, false, loss_type);

  return fx;
}

TEST_F(QuasiNewtonTest, binary_logistic_vs_sklearn) {
  CompareApprox<double> compApprox(tol);
  // Test case generated in python and solved with sklearn
  double y[10] = {1, 1, 1, 0, 1, 0, 1, 0, 1, 0};

  InputSpec in;
  in.n_row = 10;
  in.n_col = 2;
  double alpha = 0.01;

  LogisticLoss<double> loss_b(in.n_col, true, cuml_user.getImpl());
  LogisticLoss<double> loss_no_b(in.n_col, false, cuml_user.getImpl());

  SimpleVecOwning<double> w0(in.n_col + 1);
  SimpleVecOwning<double> z(in.n_row);

  DevUpload<double> devUpload(in, &X[0][0], &y[0],
                              cuml_user.getImpl().getCublasHandle());
  double l1, l2, fx;

  in.fit_intercept = true;
  double w_l1_b[2] = {-1.6899370396155091, 1.9021577534928300};
  double b_l1_b = 0.8057670813749118;
  double obj_l1_b = 0.44295941481024703;

  l1 = alpha;
  l2 = 0.0;
  fx = run(loss_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);
  ASSERT_TRUE(compApprox(obj_l1_b, fx));
  ASSERT_TRUE(checkParamsEqual(&w_l1_b[0], &b_l1_b, w0.data, 1, in.n_col,
                               in.fit_intercept, compApprox));

  fx = run_api(0, 1, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               cuml_user.getStream());
  ASSERT_TRUE(compApprox(obj_l1_b, fx));

  in.fit_intercept = true;
  double w_l2_b[2] = {-1.5339880402781370, 1.6788639581350926};
  double b_l2_b = 0.806087868102401;
  double obj_l2_b = 0.4378085369889721;

  l1 = 0;
  l2 = alpha;
  fx = run(loss_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);

  ASSERT_TRUE(compApprox(obj_l2_b, fx));
  ASSERT_TRUE(checkParamsEqual(&w_l2_b[0], &b_l2_b, w0.data, 1, in.n_col,
                               in.fit_intercept, compApprox));

  fx = run_api(0, 1, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               cuml_user.getStream());
  ASSERT_TRUE(compApprox(obj_l2_b, fx));

  in.fit_intercept = false;
  double w_l1_no_b[2] = {-1.6215035298864591, 2.3650868394981086};
  double obj_l1_no_b = 0.4769896009200278;

  l1 = alpha;
  l2 = 0.0;
  fx =
      run(loss_no_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));
  ASSERT_TRUE(checkParamsEqual(&w_l1_no_b[0], nobptr, w0.data, 1, in.n_col,
                               in.fit_intercept, compApprox));

  fx = run_api(0, 1, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               stream);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));

  in.fit_intercept = false;
  double w_l2_no_b[2] = {-1.3931049893764620, 2.0140103094119621};
  double obj_l2_no_b = 0.47502098062114273;

  l1 = 0;
  l2 = alpha;
  fx =
      run(loss_no_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));
  ASSERT_TRUE(checkParamsEqual(&w_l2_no_b[0], nobptr, w0.data, 1, in.n_col,
                               in.fit_intercept, compApprox));

  fx = run_api(0, 1, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               stream);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));
}

TEST_F(QuasiNewtonTest, multiclass_logistic_vs_sklearn) {
  // The data seems to small for the objective to be strongly convex
  // leaving out exact param checks

  CompareApprox<double> compApprox(tol);
  double y[10] = {2, 2, 0, 3, 3, 0, 0, 0, 1, 0};

  double fx, l1, l2;
  int C = 4;

  double alpha = 0.016;
  InputSpec in;
  in.n_row = 10;
  in.n_col = 2;

  DevUpload<double> devUpload(in, &X[0][0], &y[0],
                              cuml_user.getImpl().getCublasHandle());
  SimpleMatOwning<double> z(C, in.n_row);
  SimpleVecOwning<double> w0(C * (in.n_col + 1));

  Softmax<double> loss_b(in.n_col, C, true, cuml_user.getImpl());
  Softmax<double> loss_no_b(in.n_col, C, false, cuml_user.getImpl());

  l1 = alpha;
  l2 = 0.0;
  in.fit_intercept = true;
  double obj_l1_b = 0.5407911382311313;

  fx = run(loss_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);
  ASSERT_TRUE(compApprox(obj_l1_b, fx));

  fx = run_api(2, C, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               stream);
  ASSERT_TRUE(compApprox(obj_l1_b, fx));

  l1 = 0.0;
  l2 = alpha;
  in.fit_intercept = true;
  double obj_l2_b = 0.5721784062720949;

  fx = run(loss_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);
  ASSERT_TRUE(compApprox(obj_l2_b, fx));

  fx = run_api(2, C, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               stream);
  ASSERT_TRUE(compApprox(obj_l2_b, fx));

  l1 = alpha;
  l2 = 0.0;
  in.fit_intercept = false;
  double obj_l1_no_b = 0.6606929813245878;

  fx =
      run(loss_no_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));

  fx = run_api(2, C, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               stream);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));

  l1 = 0.0;
  l2 = alpha;
  in.fit_intercept = false;

  double obj_l2_no_b = 0.6597171282106854;

  fx =
      run(loss_no_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));

  fx = run_api(2, C, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               stream);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));
}

TEST_F(QuasiNewtonTest, linear_regression_vs_sklearn) {

  CompareApprox<double> compApprox(tol);
  double y[10] = {0.2675836026202781,  -0.0678277759663704, -0.6334027174275105,
                  -0.1018336189077367, 0.0933815935886932,  -1.1058853496996381,
                  -0.1658298189619160, -0.2954290675648911, 0.7966520536712608,
                  -1.0767450516284769};
  InputSpec in;
  in.n_row = 10;
  in.n_col = 2;
  double fx, l1, l2;
  double alpha = 0.01;

  DevUpload<double> devUpload(in, &X[0][0], &y[0],
                              cuml_user.getImpl().getCublasHandle());
  SimpleVecOwning<double> w0(in.n_col + 1);
  SimpleVecOwning<double> z(in.n_row);
  SquaredLoss<double> loss_b(in.n_col, true, cuml_user.getImpl());
  SquaredLoss<double> loss_no_b(in.n_col, false, cuml_user.getImpl());

  in.fit_intercept = true;
  l1 = alpha;
  l2 = 0.0;
  double w_l1_b[2] = {-0.4952397281519840, 0.3813315300180231};
  double b_l1_b = -0.08140861819001188;
  double obj_l1_b = 0.011136986298775138;
  fx = run(loss_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);
  ASSERT_TRUE(compApprox(obj_l1_b, fx));
  ASSERT_TRUE(checkParamsEqual(&w_l1_b[0], &b_l1_b, w0.data, 1, in.n_col,
                               in.fit_intercept, compApprox));

  fx = run_api(1, 1, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               stream);
  ASSERT_TRUE(compApprox(obj_l1_b, fx));

  in.fit_intercept = true;
  l1 = 0.0;
  l2 = alpha;
  double w_l2_b[2] = {-0.5022384743587150, 0.3937352417485087};
  double b_l2_b = -0.08062397391797513;
  double obj_l2_b = 0.004268621967866347;

  fx = run(loss_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);
  ASSERT_TRUE(compApprox(obj_l2_b, fx));
  ASSERT_TRUE(checkParamsEqual(&w_l2_b[0], &b_l2_b, w0.data, 1, in.n_col,
                               in.fit_intercept, compApprox));

  fx = run_api(1, 1, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               stream);
  ASSERT_TRUE(compApprox(obj_l2_b, fx));

  in.fit_intercept = false;
  l1 = alpha;
  l2 = 0.0;
  double w_l1_no_b[2] = {-0.5175178128147135, 0.3720844589831813};
  double obj_l1_no_b = 0.013981355746112447;

  fx =
      run(loss_no_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));
  ASSERT_TRUE(checkParamsEqual(&w_l1_no_b[0], nobptr, w0.data, 1, in.n_col,
                               in.fit_intercept, compApprox));

  fx = run_api(1, 1, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               stream);
  ASSERT_TRUE(compApprox(obj_l1_no_b, fx));

  in.fit_intercept = false;
  l1 = 0.0;
  l2 = alpha;
  double w_l2_no_b[2] = {-0.5241651041233270, 0.3846317886627560};
  double obj_l2_no_b = 0.007061261366969662;

  fx =
      run(loss_no_b, devUpload, in, l1, l2, w0.data, z, cuml_user.getImpl(), 0);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));
  ASSERT_TRUE(checkParamsEqual(&w_l2_no_b[0], nobptr, w0.data, 1, in.n_col,
                               in.fit_intercept, compApprox));

  fx = run_api(1, 1, in.fit_intercept, devUpload, in, l1, l2, w0.data, z, 0,
               stream);
  ASSERT_TRUE(compApprox(obj_l2_no_b, fx));
}

TEST_F(QuasiNewtonTest, dense_vs_sparse) {
  const cumlHandle_impl &cuml = cuml_user.getImpl();

  CompareApprox<double> compApprox(tol);
  // Test case generated in python and solved with sklearn
  double yhost[10] = {1, 1, 1, 0, 1, 0, 1, 0, 1, 0};

  std::vector<double> Xsparsified(N * D, 0);
  int nnz = 0;
  const double *Xptr = &X[0][0];
  for (int it = 0; it < N * D; it++) {
    if (std::abs(Xptr[it]) < 0.5) {
      Xsparsified[it] = Xptr[it];
      nnz++;
    }
  }
  SimpleMatOwning<double> X(N, D, COL_MAJOR);
  updateDevice(X.data, &Xsparsified[0], X.len);
  SimpleVecOwning<double> y(N);
  updateDevice(y.data, &yhost[0], y.len);
  SimpleVecOwning<double> csrVal(nnz);
  SimpleVecOwning<int> csrRowPtr(N + 1);
  SimpleVecOwning<int> csrColInd(nnz);

  SimpleVecOwning<int> nnzPerRow(N);
  SimpleVecOwning<double> tmp(N);
  int nnzTotal;

  cusparseMatDescr_t descr;
  CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  cusparseDnnz(cuml.getcusparseHandle(), CUSPARSE_DIRECTION_ROW, N, D, descr,
               X.data, N, nnzPerRow.data, &nnzTotal);

  cusparseDdense2csr(cuml.getcusparseHandle(), N, D, descr, X.data, N,
                     nnzPerRow.data, csrVal.data, csrRowPtr.data,
                     csrColInd.data);

  LogisticLoss<double> logLoss(D, false, cuml);
  GLMWithData<double, decltype(logLoss)> lossDense(&logLoss, X.data, y.data,
                                                   tmp.data, N, COL_MAJOR);
  LBFGSParam<double> opt_param;
  opt_param.epsilon = 1e-5;
  opt_param.max_iterations = 100;
  opt_param.m = 2;
  opt_param.max_linesearch = 50;
  SimpleVecOwning<double> w(logLoss.n_param);

  double fxd, fxs;
  int num_iters;
  double l1 = 0.001;
  qn_minimize(w, &fxd, &num_iters, lossDense, l1, opt_param, cuml, 0);

  CsrMat<double> csr(csrVal.data, csrRowPtr.data, csrColInd.data, N, D, nnz);
  GLMWithCsrData<double, decltype(logLoss)> lossSparse(&logLoss, csr, y.data,
                                                       tmp.data, N);

  w.fill(0);
  qn_minimize(w, &fxs, &num_iters, lossSparse, l1, opt_param, cuml, 0);

  ASSERT_TRUE(compApprox(fxd, fxs));
}

} // namespace GLM
} // end namespace ML
