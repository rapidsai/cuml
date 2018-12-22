#include <gtest/gtest.h>
#include "distance/distance.h"
#include "test_utils.h"
#include "random/rng.h"
#include "cuda_utils.h"
#include "nvToolsExt.h"

#include <omp.h>
#include <iostream>
#include <vector>
#include <list>
#include <algorithm>


namespace MLCommon {
namespace Distance {

// TODO(minseok): double check the result difference between CPU and GPU for the small matrx size
// TODO(minseok): it would be useful if we have a macro for the current device recovery
// TODO(minseok): consider non-power of two cases for m, n, and n_gpu

template <typename Type>
__global__ void naiveDistanceKernel(Type* out, const Type* x, const Type* y,
                                    int m, int n, int k, DistanceType type) {
    int midx = threadIdx.x + blockIdx.x * blockDim.x;
    int nidx = threadIdx.y + blockIdx.y * blockDim.y;
    if(midx >= m || nidx >= n)
        return;
    Type acc = Type(0);
    for(int i=0; i<k; ++i) {
        auto diff = x[i + midx * k] - y[i + nidx * k];
        acc += diff * diff;
    }
    if(type == EucExpandedL2Sqrt || type == EucUnexpandedL2Sqrt)
        acc = mySqrt(acc);
    out[midx * n + nidx] = acc;
}

template <typename Type>
__global__ void naiveL1DistanceKernel(
    Type* out, const Type* x, const Type* y,
    int m, int n, int k)
{
    int midx = threadIdx.x + blockIdx.x * blockDim.x;
    int nidx = threadIdx.y + blockIdx.y * blockDim.y;
    if(midx >= m || nidx >= n) {
        return;
    }

    Type acc = Type(0);
    for(int i = 0; i < k; ++i) {
        auto a = x[i + midx * k];
        auto b = y[i + nidx * k];
        auto diff = (a > b) ? (a - b) : (b - a);
        acc += diff;
    }

    out[midx * n + nidx] = acc;
}

template <typename Type>
__global__ void naiveCosineDistanceKernel(
    Type* out, const Type* x, const Type* y,
    int m, int n, int k)
{
    int midx = threadIdx.x + blockIdx.x * blockDim.x;
    int nidx = threadIdx.y + blockIdx.y * blockDim.y;
    if(midx >= m || nidx >= n) {
        return;
    }

    Type acc_a  = Type(0);
    Type acc_b  = Type(0);
    Type acc_ab = Type(0);

    for(int i = 0; i < k; ++i) {
        auto a = x[i + midx * k];
        auto b = y[i + nidx * k];

        acc_a  += a * a;
        acc_b  += b * b;
        acc_ab += a * b;
    }

    out[midx * n + nidx] = acc_ab / (sqrt(acc_a) * sqrt(acc_b));
}

template <typename Type>
void naiveDistance(Type* out, const Type* x, const Type* y, int m, int n, int k,
                   DistanceType type) {
    static const dim3 TPB(16, 32, 1);
    dim3 nblks(ceildiv(m, (int)TPB.x), ceildiv(n, (int)TPB.y), 1);

    switch (type) {
        case EucUnexpandedL1:
            naiveL1DistanceKernel<Type><<<nblks,TPB>>>(out, x, y, m, n, k);
            break;
        case EucUnexpandedL2Sqrt:
        case EucUnexpandedL2:
        case EucExpandedL2Sqrt:
        case EucExpandedL2:
            naiveDistanceKernel<Type><<<nblks,TPB>>>(out, x, y, m, n, k, type);
            break;
        case EucExpandedCosine:
            naiveCosineDistanceKernel<Type><<<nblks,TPB>>>(out, x, y, m, n, k);
            break;
        default:
            FAIL() << "should be here\n";
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

enum PartitionScheme {
	XOnly,
	YOnly,
	Both
};

template <typename T>
struct OpgDistanceInputs {
    T tolerance;
    int m, n, k;
    DistanceType type;
    unsigned long long int seed;
		int n_gpu;
		PartitionScheme scheme;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const OpgDistanceInputs<T>& dims) {
    return os;
}

template <typename T>
struct Task {
	T* d_X;
	T* d_Y;
	T* dist;
	DistanceType type;
	int m, n, k;
};


template <typename T>
class Worker {
  public:
    typedef cutlass::Shape<8, 128, 128> OutputTile_t;
	  Worker(int device_id);
		~Worker();
		void enqueue(Task<T>* t);
		void execute();
		Task<T>* dequeue();
		::testing::AssertionResult verify(T tolerance);
		bool empty() const;
		int getDeviceId() const;
	private:
		int device_id_;
		T* workspace_;
		size_t worksize_;
    DistanceType type_;
		std::list<Task<T>*> active_queue_;
};


template <typename T>
Worker<T>::Worker(int device_id)
		: device_id_(device_id),
			workspace_(nullptr),
			worksize_(0) { 
}

template <typename T>
Worker<T>::~Worker() {
	if(workspace_)
		CUDA_CHECK(cudaFree(workspace_));
}

template <typename T>
void Worker<T>::enqueue(Task<T>* t) {
	ASSERT(t != nullptr, "t == nullptr");

	int current_device_id;
	CUDA_CHECK(cudaGetDevice(&current_device_id));

	size_t new_worksize = 0;

	CUDA_CHECK(cudaSetDevice(device_id_));

	distance<T,T,Task<T>,Task<T>, OutputTile_t >(t->d_X, t->d_Y, t->m, t->n, t->k,
                                               *t, *t, t->type,
                                               nullptr, new_worksize);
	if (new_worksize != 0) {
			if(new_worksize > worksize_) {
				if(worksize_ != 0)
					CUDA_CHECK(cudaFree(workspace_));
				worksize_ = new_worksize;
				workspace_ = nullptr;
				allocate(workspace_, worksize_);
			}
	}

	active_queue_.push_back(t);

	CUDA_CHECK(cudaSetDevice(current_device_id));
}

template <typename T>
void Worker<T>::execute() {
	int current_device_id;
	CUDA_CHECK(cudaGetDevice(&current_device_id));

	CUDA_CHECK(cudaSetDevice(device_id_));

	for(auto it = active_queue_.begin(); it != active_queue_.end(); ++it) {
		Task<T>* t = *it;
    distance<T,T,Task<T>,Task<T>, OutputTile_t >(t->d_X, t->d_Y, t->m, t->n, t->k,
                                                 *t, *t, t->type,
                                                 (void*)workspace_, worksize_);
	}

	CUDA_CHECK(cudaSetDevice(current_device_id));
}

template <typename T>
Task<T>* Worker<T>::dequeue() {
	if(empty())
		return nullptr;

	Task<T>* t = active_queue_.front();
	active_queue_.pop_front();
	return t;
}

template <typename T>
::testing::AssertionResult Worker<T>::verify(T tolerance) {
	int current_device_id;
	CUDA_CHECK(cudaGetDevice(&current_device_id));

	CUDA_CHECK(cudaSetDevice(device_id_));

	auto ret = ::testing::AssertionSuccess();
	for(auto it = active_queue_.begin(); it != active_queue_.end(); ++it) {
		Task<T>* t = *it;
		T* dist_ref = nullptr;
		allocate(dist_ref, t->m*t->n);
		naiveDistance(dist_ref, t->d_X, t->d_Y, t->m, t->n, t->k, t->type);
		auto ret = devArrMatch(dist_ref, t->dist, t->m, t->n, CompareApprox<T>(tolerance));
		CUDA_CHECK(cudaFree(dist_ref));
		if(ret != ::testing::AssertionSuccess())
			break;
	}

	CUDA_CHECK(cudaSetDevice(current_device_id));

	return ret;
}

template <typename T>
bool Worker<T>::empty() const {
	return active_queue_.empty();
}

template <typename T>
int Worker<T>::getDeviceId() const {
	return device_id_;
}

void getNumberOfTiles(PartitionScheme scheme,
		int m, int n, int k, int n_gpu, int& n_vertical_tiles, int& n_horizontal_tiles) {
	switch(scheme) {
	case XOnly:
		n_vertical_tiles = n_gpu;
		n_horizontal_tiles = 1;
		break;
	case YOnly:
		n_vertical_tiles = 1;
		n_horizontal_tiles = n_gpu;
		break;
	case Both:
		n_vertical_tiles = std::max(1, m / 4096);
		n_horizontal_tiles = std::max(1, n / 4096);
		break;
	default:
		ASSERT(false, "Invalid PartitionScheme '%d'!", scheme);
	}
}


template <typename T>
void assignTasks(std::vector<Worker<T>*>& workers, int n_gpu,
		int m, int n, int k, DistanceType type, PartitionScheme scheme, unsigned long long int seed) {
		ASSERT(workers.size() == n_gpu, "# workers(%d) != # GPUs(%d)", workers.size(), n_gpu);

		int current_device_id;
		CUDA_CHECK(cudaGetDevice(&current_device_id));

		int n_vertical_tiles = 0, n_horizontal_tiles = 0;
		getNumberOfTiles(scheme, m, n, k, n_gpu, n_vertical_tiles, n_horizontal_tiles);

		for(int y=0; y<n_vertical_tiles; y++) {
			for(int x=0; x<n_horizontal_tiles; x++) {
				int id = (x + (y * n_horizontal_tiles)) % n_gpu;
			  Worker<T>* worker = workers[id];
				ASSERT(id == worker->getDeviceId(), "id(%d) != deviceId(%d)", id, worker->getDeviceId());
				CUDA_CHECK(cudaSetDevice(worker->getDeviceId()));

				Task<T>* task = new Task<T>;
				task->m = m / n_vertical_tiles;
				task->n = n / n_horizontal_tiles;
				task->k = k;
				task->type = type;
				int x_len = task->m*task->k;
				int y_len = task->n*task->k;
				int dist_len = task->m*task->n;
        allocate(task->d_X, x_len);
        allocate(task->d_Y, y_len);
        allocate(task->dist, dist_len);
        Random::Rng<T> r(seed);
        r.uniform(task->d_X, x_len, T(-1.0), T(1.0));
        r.uniform(task->d_Y, y_len, T(-1.0), T(1.0));
				worker->enqueue(task);
			}
		}

		CUDA_CHECK(cudaSetDevice(current_device_id));
}

template <typename T>
void finalizeTasks(std::vector<Worker<T>*>& workers) {
	int current_device_id;
	CUDA_CHECK(cudaGetDevice(&current_device_id));

	while(!workers.empty()) {
		Worker<T>* worker = workers.back();
		workers.pop_back();
		CUDA_CHECK(cudaSetDevice(worker->getDeviceId()));
		while(!worker->empty()) {
			Task<T>* task = worker->dequeue();
			CUDA_CHECK(cudaFree(task->d_X));
			CUDA_CHECK(cudaFree(task->d_Y));
			CUDA_CHECK(cudaFree(task->dist));
			delete task;
		}
		delete worker;
	}

	CUDA_CHECK(cudaSetDevice(current_device_id));
}

void syncAll(int n_gpu) {
	int current_device_id;
	CUDA_CHECK(cudaGetDevice(&current_device_id));

	for(int i=0; i<n_gpu; i++) {
		CUDA_CHECK(cudaSetDevice(i));
		cudaDeviceSynchronize();
	}

	CUDA_CHECK(cudaSetDevice(current_device_id));
}

template <typename T>
class OpgDistanceTest: public ::testing::TestWithParam<OpgDistanceInputs<T> > {
protected:
    void SetUp() override {
        // Get the parameters
        params = ::testing::TestWithParam<OpgDistanceInputs<T>>::GetParam();
				int n_gpu = params.n_gpu;

        // Skip the test if # available GPUs is less than the specified one.
				int avail_gpu;
				CUDA_CHECK(cudaGetDeviceCount(&avail_gpu));
        if(avail_gpu < n_gpu)
          GTEST_SKIP();

				ASSERT(params.m > n_gpu, "Invalid m(%d)", params.m);

        // Initialize all GPU workers and assign tasks to them
				for(int i=0; i<n_gpu; i++)
					workers.push_back(new Worker<T>(i));
				assignTasks(workers, n_gpu,
						params.m, params.n, params.k, params.type, params.scheme, params.seed);


        int n_rep = 1;
	      float elapsed = 0;
				double time_min = 1e100;

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

        for(int r=0; r<n_rep; r++) {
					syncAll(n_gpu);

					cudaEventRecord(start);

#pragma omp parallel for num_threads(n_gpu)
					for(int i=0; i<n_gpu; i++)
						workers[i]->execute();

					syncAll(n_gpu);

					cudaEventRecord(stop);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&elapsed, start, stop);
					double time = (double)elapsed / 1000.;
					time_min = std::min(time, time_min);
				}

				cudaEventDestroy(start);
				cudaEventDestroy(stop);
    }

    void TearDown() override {
			finalizeTasks(workers);
    }

protected:
    OpgDistanceInputs<T> params;
		std::vector<Worker<T>* > workers;
};

const std::vector<OpgDistanceInputs<float> > inputsf = {
    {0.001f, 1024,   1024,  1024, EucExpandedL2, 1234ULL, 8, XOnly},
    {0.001f, 2048,   2048,  2048, EucExpandedL2, 1234ULL, 8, XOnly},
    {0.001f, 4096,   4096,  4096, EucExpandedL2, 1234ULL, 8, XOnly},
    {0.001f, 8192,   8192,  8192, EucExpandedL2, 1234ULL, 8, XOnly},
    {0.001f, 16384, 16384, 16384, EucExpandedL2, 1234ULL, 8, XOnly},

    {0.001f, 1024,   1024,  1024, EucExpandedL2, 1234ULL, 8, YOnly},
    {0.001f, 2048,   2048,  2048, EucExpandedL2, 1234ULL, 8, YOnly},
    {0.001f, 4096,   4096,  4096, EucExpandedL2, 1234ULL, 8, YOnly},
    {0.001f, 8192,   8192,  8192, EucExpandedL2, 1234ULL, 8, YOnly},
    {0.001f, 16384, 16384, 16384, EucExpandedL2, 1234ULL, 8, YOnly},

    {0.001f, 1024,   1024,  1024, EucExpandedL2, 1234ULL, 8, Both},
    {0.001f, 2048,   2048,  2048, EucExpandedL2, 1234ULL, 8, Both},
    {0.001f, 4096,   4096,  4096, EucExpandedL2, 1234ULL, 8, Both},
    {0.001f, 8192,   8192,  8192, EucExpandedL2, 1234ULL, 8, Both},
    {0.001f, 16384, 16384, 16384, EucExpandedL2, 1234ULL, 8, Both},
};

typedef OpgDistanceTest<float> TestF;
TEST_P(TestF, Result) {
		// verify the result
		for(int i=0; i<params.n_gpu; i++) {
				ASSERT_TRUE(workers[i]->verify(params.tolerance));
		}
}

INSTANTIATE_TEST_CASE_P(OpgDistanceTests, TestF, ::testing::ValuesIn(inputsf));

} // end namespace Distance
} // end namespace MLCommon
