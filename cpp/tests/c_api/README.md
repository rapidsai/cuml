# C-API Test Folder

The purpose of this folder and CMake target is to verify it's possible to build an executable/library using the C-API. Since the C-API is compiled using C++ and `extern "C"`, this verifies there are no additional oversights that would prevent users from consuming the C-API. For example, including any of the C++ standard library headers would not prevent compiling `libcuml.so`, but would cause errors when using that library.

This test works by simply `#include`'ing each of the `*_api.h` header files and calling each method with dummy arguments. This ensures the functions are properly imported into a C application and linked without needing to actually call the methods.

## Adding New Headers or Functions

Any changes to the C-API need to be reflected in these tests. If a new `*_api.h` header is added or a new function appended to existing headers, this folder should be updated to reflect the change.

## Macro Definitions

To help prevent accidentally including the C-API files when compiling `libcuml++.so`, two new defines have been created: `CUML_C_API` and `CUML_CPP_API`. Each is set when compiling their respective libraries and can be used to prevent accidentally including the wrong header files. For example, in `cuml_api.h`, the following section will raise an error during compilation if included into the C++ API:

```cpp
#ifdef CUML_CPP_API
#error \
  "This header is only for the C-API and should not be included from the C++ API."
#endif
```
