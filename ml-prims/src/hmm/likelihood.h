#include <random/rng.h>
#include <linalg/cublas_wrappers.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>


#define IDX2C(i,j,ld) (j*ld + i)

using namespace MLCommon::LinAlg;

namespace MLCommon {
namespace HMM {

template <typename T>
struct _gmm_likelihood_functor
{
        const T *data, *mus, *sigmas;
        T* rhos;
        const int n_classes, dim;
        const bool is_log;

        _likelihood_functor(T *data, T *mus, T *sigmas, T *rhos,
                            int n_classes, int dim, bool is_log){
                this->data = data;
                this->mus = mus;
                this->sigmas = sigmas;
                this->rhos = rhos;
                this.n_classes = n_classes;
                this.dim = dim;
                this.is_log = is_log;
        }

        __host__ __device__
        T operator()(int sample_id, int class_id)
        {
                return (T) lhd_gaussian(x + dim * sample_id,
                                        mus + dim * class_id,
                                        sigmas + dim * dim * class_id,
                                        rhos + n_classes * sample_id, is_log);
        }
};

template <typename T>
T lhd_gaussian(T* x, T* mu, T* sigma, int dim, bool is_log){
        T logl = 0
                 logl += 0.5 * std::log(2 * std::pi);
        T determinant = 0.;

        // Compute the squared sum
        T* temp;
        allocate(temp, dim);

        // x - mu
        T scalar = -0.5;
        subtract(temp, x, mu, dim);
        LinAlg::scalarMultiply(temp, temp, scalar, dim);

        // sigma * (x - mu)
        bilinear(sigma, dim, temp, cublas_h, result);

        logl += determinant;
        return logl;
}


template <typename T>
struct entropy_functor
{
        __host__ __device__
        T operator()(T& x, T& p)
        {
                return (T) x * std::log(p);
        }
};

template <typename T>
T ll_multinomial(T* x, T* p, int dim){
        T logl = 0;
        entropy_functor<T> entropy_op;
        thrust::plus<T> plus_op;

        thrust::transform_reduce(thrust::device_pointer_cast(x),
                                 thrust::device_pointer_cast(x+dim),
                                 thrust::device_pointer_cast(p),
                                 thrust::device_pointer_cast(p+dim),
                                 entropy_op, logl, plus_op);
        cudaCheckError();
        return logl;
}


template <typename T>
T ll_gmm(T* x, T* p, T* mus, T* sigmas, int n_samples){
        T logl = 0;
        gmm_functor<T>(dim) gmm_op;
        thrust::plus<T> plus_op;

        thrust::device_vector<int>  samples_v(n_samples);
        thrust::device_vector<int> classes_v(gmm.n_classes);
        first = thrust::make_zip_iterator(thrust::make_tuple(samples_v.begin(), classes_v.begin()));
        last  = thrust::make_zip_iterator(thrust::make_tuple(samples_v.end(),   classes_v.end()));

        thrust::for_each(thrust::device, first, last, gmm_likelihood);
        MLCommon::HMM::normalize_matrix(out_rhos, gmm.dim_x, gmm.n_classes);

        thrust::transform_reduce(first, last, gmm_op, logl, plus_op);
        return logl;
}

}
}
