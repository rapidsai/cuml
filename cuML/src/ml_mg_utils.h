#include <cuda_runtime.h>

#ifndef ML_MG_UTILS_H_
#define ML_MG_UTILS_H_

namespace ML {

namespace MLCommon {

	int get_device(void *ptr) {
		cudaPointerAttributes att;
		cudaPointerGetAttributes(&att, ptr);
		return att.device;
	}
}
};




#endif /* ML_MG_UTILS_H_ */
