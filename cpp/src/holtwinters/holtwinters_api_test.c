
#include <cuml/tsa/holtwinters_api.h>

void test_holtwinters() {

   cumlHandle_t handle = 0;
   cumlError_t response = CUML_SUCCESS;

   response = cumlHoltWinters_buffer_size(0, 1, 2, NULL, NULL, NULL, NULL, NULL, NULL);

   response = cumlHoltWintersSp_fit(handle, 0, 1, 2, 3, ADDITIVE, 1.0f, NULL, NULL, NULL, NULL, NULL);

   response = cumlHoltWintersDp_fit(handle, 0, 1, 2, 3, ADDITIVE, 1.0f, NULL, NULL, NULL, NULL, NULL);

   response = cumlHoltWintersSp_forecast(handle, 0, 1, 2, 3, ADDITIVE, NULL, NULL, NULL, NULL);

   response = cumlHoltWintersDp_forecast(handle, 0, 1, 2, 3, ADDITIVE, NULL, NULL, NULL, NULL);
}