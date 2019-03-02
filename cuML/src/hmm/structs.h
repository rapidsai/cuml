#include <hmm/random.h>
#include "linalg/cusolver_wrappers.h"


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



}
}
