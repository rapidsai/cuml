#include "umap.h"
#include "runner.h"
#include "umapparams.h"

#include <iostream>

namespace ML {

    static const int TPB_X = 32;

    UMAP_API::UMAP_API(UMAPParams *params): params(params){
        knn = nullptr;
    };

    UMAP_API::~UMAP_API() {
        delete knn;
    }



    void UMAP_API::fit(float *X, int n, int d, float *embeddings) {
        this->knn = new kNN(d);
        UMAPAlgo::_fit<float, TPB_X>(X, n, d, knn, get_params(), embeddings);
    }

    void UMAP_API::transform(float *X, int n, int d,
            float *embedding, int embedding_n,
            float *out) {
        UMAPAlgo::_transform<float, TPB_X>(X, n, d,
                embedding, embedding_n, knn,
                get_params(), out);
    }

    UMAPParams* UMAP_API::get_params()  { return this->params; }
}
