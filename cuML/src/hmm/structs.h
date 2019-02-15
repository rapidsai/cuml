#include <hmm/random.h>

using namespace MLCommon::HMM;

namespace ML {
namespace HMM {

// template <typename T>
// struct paramsRandom {
//         T start, end;
//         unsigned long long seed;
//         paramsRandom(T _start, T _end, unsigned long long _seed) : start(_start),
//                 end(_end), seed(_seed){
//         };
// };

struct paramsEM {
        int n_iter;
        paramsEM(int _n_iter) {
                n_iter = _n_iter;
        }
};

template <typename T>
struct GMM {
        T *z, *x;
        T *mus, *sigmas, *ps;
        T *rhos;

        int nCl;
        int nDim;
        int nObs;

        // EM parameters
        paramsEM *paramsEm;

        // // Random parameters
        paramsRandom<T> *paramsRd;

        // all the workspace related pointers
        int *info;
        bool initialized=false;

        // cusolverDnHandle_t *cusolverHandle;
        cublasHandle_t *cublasHandle;
};

}
}
