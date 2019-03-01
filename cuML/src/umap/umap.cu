#include "umap.h"
#include "runner.h"

namespace ML {


    void UMAP::fit(const float *X, int n, int d, kNN *knn, float *embeddings) {
        this->knn = knn;
        UMAPAlgo::_fit(X, n, d, knn, get_params(), embeddings);
    }

    void UMAP::transform(const float *X, int n, int d,
            float *embedding, int embedding_n,
            kNN *knn,
            float *out) {
        UMAPAlgo::_transform<float, 256>(X, n, d,
                embedding, embedding_n, knn,
                get_params(), out);
    }

    UMAPParams* UMAP::get_params()  { return this->params; }
}
