namespace MLCommon {

    namespace Sparse {

        template <typename T>
        struct COOInputs {
          int m, n, nnz;
          unsigned long long int seed;
        };

        template <typename T>
        ::std::ostream &operator<<(::std::ostream &os, const COOInputs<T> &dims) {
          return os;
        }
    }
}
