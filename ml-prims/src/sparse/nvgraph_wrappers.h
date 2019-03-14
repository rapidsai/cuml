#pragma once

#include <nvgraph.h>

namespace MLCommon {

    namespace Sparse {
        #define NVGRAPH_CHECK(call)  \
            do { \
                nvgraphStatus_t status = call; \
                ASSERT(status == NVGRAPH_STATUS_SUCCESS, "FAIL: call='%s'\n", #call); \
            } while(0)
    }
}
