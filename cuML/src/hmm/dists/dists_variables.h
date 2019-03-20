# pragma once

namespace multinomial {
template <typename T>
struct Multinomial {
        T *dPis, *dLlhd;
        int nCl;
};
}
