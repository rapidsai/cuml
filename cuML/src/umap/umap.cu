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

    /**
     * Fits a UMAP model
     * @param X
     *        pointer to an array in row-major format (note: this will be col-major soon)
     * @param n
     *        n_samples in X
     * @param d
     *        d_features in X
     * @param embeddings
     *        an array to return the output embeddings of size (n_samples, n_components)
     */
    void UMAP_API::fit(float *X, int n, int d, float *embeddings) {
        this->knn = new kNN(d);
        UMAPAlgo::_fit<float, TPB_X>(X, n, d, knn, get_params(), embeddings);
    }

    /**
     * Project a set of X vectors into the embedding space.
     * @param X
     *        pointer to an array in row-major format (note: this will be col-major soon)
     * @param n
     *        n_samples in X
     * @param d
     *        d_features in X
     * @param embedding
     *        pointer to embedding array of size (embedding_n, n_components) that has been created with fit()
     * @param embedding_n
     *        n_samples in embedding array
     * @param out
     *        pointer to array for storing output embeddings (n, n_components)
     */
    void UMAP_API::transform(float *X, int n, int d,
            float *embedding, int embedding_n,
            float *out) {
        UMAPAlgo::_transform<float, TPB_X>(X, n, d,
                embedding, embedding_n, knn,
                get_params(), out);
    }

    /**
     * Get the UMAPParams instance
     */
    UMAPParams* UMAP_API::get_params()  { return this->params; }
}
