
#include <cuml/linear_model/glm_api.h>

void test_glm() {

   cumlHandle_t handle = 0;
   cumlError_t response = CUML_SUCCESS;

   response = cumlSpQnFit(handle, NULL, NULL, 0, 1, 2, false, 1.0f, 2.0f, 3, 3.0f, 4, 5, 6, NULL, NULL, NULL, true, 7);

   response = cumlDpQnFit(handle, NULL, NULL, 0, 1, 2, false, 1.0f, 2.0f, 3, 3.0f, 4, 5, 6, NULL, NULL, NULL, true, 7);

}