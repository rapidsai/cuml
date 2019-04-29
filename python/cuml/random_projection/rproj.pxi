cdef extern from "random_projection/rproj_c.h" namespace "ML":

    # Structure holding random projection hyperparameters
    cdef struct paramsRPROJ:
        int n_samples           # number of samples
        int n_features          # number of features (original dimension)
        int n_components        # number of components (target dimension)
        double eps              # error tolerance according to Johnson-Lindenstrauss lemma
        bool gaussian_method    # toggle Gaussian or Sparse random projection methods
        double density		# ratio of non-zero component in the random projection matrix (used for sparse random projection)
        bool dense_output       # toggle random projection's transformation as a dense or sparse matrix
        int random_state        # seed used by random generator

    # Structure describing random matrix
    cdef cppclass rand_mat[T]:
        rand_mat() except +     # random matrix structure constructor (set all to nullptr)
        T *dense_data           # dense random matrix data
        int *indices            # sparse CSC random matrix indices
        int *indptr             # sparse CSC random matrix indptr
        T *sparse_data          # sparse CSC random matrix data
        size_t sparse_data_size # sparse CSC random matrix number of non-zero elements

    # Method used to fit the model
    cdef void RPROJfit[T](rand_mat[T] *random_matrix, paramsRPROJ* params)
    
    # Method used to apply data transformation
    cdef void RPROJtransform[T](T *input, rand_mat[T] *random_matrix,
                                T *output, paramsRPROJ* params)
