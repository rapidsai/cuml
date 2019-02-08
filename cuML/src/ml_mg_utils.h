#include <cuda_runtime.h>
#include <iostream>

#pragma once
namespace ML {

	int get_device(const void *ptr) {

		std::cout << "Pointer:  " << ptr << std::endl;

		cudaPointerAttributes att;
		cudaError_t err = cudaPointerGetAttributes(&att, ptr);

		std::cout << "Error: " << err << std::endl;
		std::cout << "Device #: " << att.device << std::endl;
		std::cout << "Device Type: " << att.type << std::endl;

		return att.device;
}
};

