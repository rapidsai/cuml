#include "distance/distance.h"
#include "harness.h"
#include "random/rng.h"
#include "cuda_utils.h"

namespace MLCommon {
namespace Distance {

struct DistanceParams {
    int m, n, k;
		DistanceType type;
    unsigned long long int seed;

    std::string str() const {
        std::ostringstream oss;
        oss << "m: " << m << " n: " << n << " k: " << k;
        return oss.str();
    }
};

template <typename T>
struct OutParams {
  T* dist;
};

template <typename T>
struct InParams {};

template<typename T>
struct DistanceBenchmark : Benchmark<DistanceParams> {
  typedef cutlass::Shape<8, 128, 128> OutputTile_t;
  void setup() {
    Random::Rng<T> r(params.seed);
    int m = params.m;
    int n = params.n;
    int k = params.k;
    allocate(x, m*k);
    allocate(y, n*k);
    allocate(out, m*n);
    r.uniform(x, m*k, T(-1.0), T(1.0));
    r.uniform(y, n*k, T(-1.0), T(1.0));

    OutParams<T> out_params = { out };
    InParams<T> in_params;

    workspace = nullptr;
    worksize = 0;

    distance<T,T,InParams<T>,OutParams<T>,OutputTile_t>(x, y, m, n, k,
                                                        in_params, out_params, params.type,
                                                        nullptr, worksize);
    if (worksize != 0) {
        allocate(workspace, worksize);
    }
  }

  void teardown() {
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(workspace));
  }

  void run() {
    int m = params.m;
    int n = params.n;
    int k = params.k;

    OutParams<T> out_params = { out };
    InParams<T> in_params;

    distance<T,T,InParams<T>,OutParams<T>,OutputTile_t>(x, y, m, n, k,
                                                        in_params, out_params, params.type,
                                                        (void*)workspace, worksize);
  }

  IdealTime getIdealTime() const {
    auto gd = Harness::GetDetails();
    float com = getIdealComputeTime();
    float mem = getIdealMemoryTime();
    return IdealTime(com, mem);
  }

	float getIdealComputeTime() const {
    auto gd = Harness::GetDetails();
    float ops = 0.f;
		switch(params.type) {
			case EucExpandedL2:
        ops += 2.f * params.m * params.n * params.k; // X*Y+C
        ops += 2.f * (params.m * params.k + params.k * params.n); // X^2, Y^2
        ops += 3.f * params.m * params.n; // X^2 + Y^2 - 2 X*Y
				break;
			case EucUnexpandedL2:
        ops += 3.f * params.m * params.n * params.k; // diff * diff + C
				break;
			default:
				abort();
		}
    float com = ops / (gd.nSMs * 64 * gd.smClk * 2 * 1e6f);
		com *= 1e3f;
		return com;
	}
	float getIdealMemoryTime() const {
    auto gd = Harness::GetDetails();
    float len = 0.f;
		switch(params.type) {
			case EucExpandedL2:
				len += params.m * params.n + params.m * params.k + params.n * params.k;
				len += params.m * params.k + params.m;
				len += params.k * params.n + params.n;
				len += params.m * params.n + params.m + params.n;
				break;
			case EucUnexpandedL2:
				len += params.m * params.n + params.m * params.k + params.n * params.k;
				break;
			default:
				abort();
		}
    float mem = len * sizeof(float) / gd.getMemBW();
    mem /= 1e6f; // time in ms
		return mem;
	}

  T* x;
  T* y;
  T* out;
  char* workspace;
  size_t worksize;
};

static std::vector<DistanceParams > inputs = {
  {32,    16384, 16384,  EucExpandedL2, 1234ULL},
  {64,    16384, 16384,  EucExpandedL2, 1234ULL},
  {128,   16384, 16384,  EucExpandedL2, 1234ULL},
  {256,   16384, 16384,  EucExpandedL2, 1234ULL},
  {512,   16384, 16384,  EucExpandedL2, 1234ULL},
  {1024,  16384, 16384,  EucExpandedL2, 1234ULL},
  {16384, 32,    16384,  EucExpandedL2, 1234ULL},
  {16384, 64,    16384,  EucExpandedL2, 1234ULL},
  {16384, 128,    16384, EucExpandedL2, 1234ULL},
  {16384, 256,    16384, EucExpandedL2, 1234ULL},
  {16384, 512,    16384, EucExpandedL2, 1234ULL},
  {16384, 1024,   16384, EucExpandedL2, 1234ULL},
  {16384, 16384,  32,    EucExpandedL2, 1234ULL},
  {16384, 16384,  64,    EucExpandedL2, 1234ULL},
  {16384, 16384,  128,   EucExpandedL2, 1234ULL},
  {16384, 16384,  256,   EucExpandedL2, 1234ULL},
  {16384, 16384,  512,   EucExpandedL2, 1234ULL},
  {16384, 16384,  1024,  EucExpandedL2, 1234ULL},
  {16384, 16384,  16384, EucExpandedL2, 1234ULL},
  {32,    16384, 16384,  EucUnexpandedL2, 1234ULL},
  {64,    16384, 16384,  EucUnexpandedL2, 1234ULL},
  {128,   16384, 16384,  EucUnexpandedL2, 1234ULL},
  {256,   16384, 16384,  EucUnexpandedL2, 1234ULL},
  {512,   16384, 16384,  EucUnexpandedL2, 1234ULL},
  {1024,  16384, 16384,  EucUnexpandedL2, 1234ULL},
  {16384, 32,    16384,  EucUnexpandedL2, 1234ULL},
  {16384, 64,    16384,  EucUnexpandedL2, 1234ULL},
  {16384, 128,    16384, EucUnexpandedL2, 1234ULL},
  {16384, 256,    16384, EucUnexpandedL2, 1234ULL},
  {16384, 512,    16384, EucUnexpandedL2, 1234ULL},
  {16384, 1024,   16384, EucUnexpandedL2, 1234ULL},
  {16384, 16384,  32,    EucUnexpandedL2, 1234ULL},
  {16384, 16384,  64,    EucUnexpandedL2, 1234ULL},
  {16384, 16384,  128,   EucUnexpandedL2, 1234ULL},
  {16384, 16384,  256,   EucUnexpandedL2, 1234ULL},
  {16384, 16384,  512,   EucUnexpandedL2, 1234ULL},
  {16384, 16384,  1024,  EucUnexpandedL2, 1234ULL},
  {16384, 16384,  16384, EucUnexpandedL2, 1234ULL},
};

REGISTER_BENCH(DistanceBenchmark<float>, DistanceParams, "Distance", inputs);

}
}
