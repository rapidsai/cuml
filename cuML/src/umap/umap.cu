#include "umap.h"
#include "runner.h"
#include "umapparams.h"

#include <iostream>

namespace ML {

    static const int TPB_X = 64;

    UMAP::UMAP(UMAPParams *params): params(params){
        knn = nullptr;
    };

    void UMAP::fit(float *X, int n, int d, kNN *knn, float *embeddings) {
        this->knn = knn;
        UMAPAlgo::_fit<float, TPB_X>(X, n, d, knn, get_params(), embeddings);

        std::cout << "n_neighbors=" << this->params->n_neighbors << std::endl;
    }

    void UMAP::transform(float *X, int n, int d,
            float *embedding, int embedding_n,
            kNN *knn,
            float *out) {
        UMAPAlgo::_transform<float, TPB_X>(X, n, d,
                embedding, embedding_n, knn,
                get_params(), out);
    }

    UMAPParams* UMAP::get_params()  { return this->params; }
}
