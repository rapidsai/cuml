# Expects clang-format to be inside this repo itself!
# This has been done to avoid spurious code changes due to version mismatch
# Once we tackle this issue, we can go back to PATH-based finding of clang-format
find_program(ClangFormat_EXE
  NAMES clang-format
  PATHS ${MLPRIMS_DIR}/scripts
  DOC "path to clang-format exe"
  NO_DEFAULT_PATH)
find_program(ClangFormat_PY
  NAMES run-clang-format.py
  PATHS ${MLPRIMS_DIR}/scripts
  DOC "path to run-clang-format python script")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ClangFormat DEFAULT_MSG
  ClangFormat_EXE ClangFormat_PY)

include(CMakeParseArguments)

# clang formatting as a target in the final build stage
function(add_clang_format)
  if(ClangFormat_FOUND)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs TARGETS SRCS)
    cmake_parse_arguments(cf "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    foreach(cf_TARGET ${cf_TARGETS})
      if(NOT TARGET ${cf_TARGET})
        message(FATAL_ERROR "add_clang_format: '${cf_TARGET}' is not a target")
      endif()
    endforeach()
    set(dummy_file clang_format_output)
    add_custom_command(OUTPUT ${dummy_file}
      COMMENT "Clang-Format ${cf_TARGET}"
      COMMAND ${ClangFormat_PY}
        -bindir ${CMAKE_SOURCE_DIR}
        -exe ${ClangFormat_EXE}
        -srcdir ${CMAKE_SOURCE_DIR} ${cf_SRCS})
    # add the dependency on this dummy target
    # So, if the main source file has been modified, clang-format will
    # automatically be run on it!
    add_custom_target(clang_format
      SOURCES ${dummy_file}
      COMMENT "Clang-Format for target ${_target}")
    foreach(cf_TARGET ${cf_TARGETS})
      add_dependencies(${cf_TARGET} clang_format)
    endforeach()
  else()
    message("add_clang_format: clang-format exe not found")
  endif()
endfunction(add_clang_format)
