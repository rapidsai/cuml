# Finds clang-tidy exe based on the PATH env variable
string(REPLACE ":" ";" EnvPath $ENV{PATH})
find_program(ClangTidy_EXE
  NAMES clang-tidy
  PATHS EnvPath
  DOC "path to clang-tidy exe")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ClangTidy DEFAULT_MSG
  ClangTidy_EXE)
