#pragma once

namespace MLCommon {
template <typename T>
struct bilinearHandle_t {
        T **dT_array;
};

template <typename T>
struct determinantHandle_t {
        int **dipiv_array, *info_array;
        T **dA_array_cpy; // U and L are stored here after getrf
};

template <typename T>
struct inverseHandle_t {
        int **dipiv_array, *info_array;
        T **dA_array_cpy;
};
}
