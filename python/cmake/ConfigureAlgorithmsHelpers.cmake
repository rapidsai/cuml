function(add_module_gpu_default FILENAME)
    set (extra_args ${ARGN})
    list(LENGTH extra_args extra_count)
    if (${extra_count} GREATER 0 OR
        ${CUML_UNIVERSAL} OR
        ${CUML_GPU})
      list(APPEND cython_sources
           ${FILENAME})
      set (cython_sources ${cython_sources} PARENT_SCOPE)
    endif()
endfunction()
