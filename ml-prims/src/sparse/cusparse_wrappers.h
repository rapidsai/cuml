#pragma once

#include <cusparse_v2.h>

namespace MLCommon {

    namespace Sparse {
        #define CUSPARSE_CHECK(call)  \
            do { \
                cusparseStatus_t status = call; \
                ASSERT(status == CUSPARSE_STATUS_SUCCESS, "FAIL: call='%s'\n", #call); \
            } while(0)


    }
}
