
#include <cuml/cluster/dbscan_api.h>

void test_dbscan() {

   cumlHandle_t handle = 0;
   cumlError_t response = CUML_SUCCESS;

   response = cumlSpDbscanFit(handle, NULL, 0, 1, 1.0f, 2, NULL, NULL, 10, 1);

   response = cumlDpDbscanFit(handle, NULL, 0, 1, 1.0, 2, NULL, NULL, 10, 1);
}
