[WIP] Addition of CumlArray Decorators and Descriptors to Ensure Consistent Usage In Library

Due to the large number of input and output array types that cuml supports, the library needs to be flexible in how array data is handed when entering and exiting the API. This has led to some inconsistent usage of `CumlArray`, unnecessary conversions and memcpys, and adds repeated work for the developers. 

This PR helps with these issues by adding a new object `CumlArrayDescriptor` and a set of decorators in `cuml.internals` that make working with array data easier for developers and more consistent for users while still being flexible enough to handle any scenario. Finally, this PR updates the majority of the code base to be consistent and use the new features.

A future developer guide will do a deep dive into how the new features should be used, but for now, 