namespace MLCommon {

    namespace Sparse {

        template <typename T>
        struct CSRInputs {
          int m, n, nnz;
          unsigned long long int seed;
        };

        template <typename T>
        ::std::ostream &operator<<(::std::ostream &os, const CSRInputs<T> &dims) {
          return os;
        }
    }
}
