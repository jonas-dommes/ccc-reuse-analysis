cmake_minimum_required(VERSION 3.17)

set(CudaCoarseningCompiler_VERSION 0.1)
project(cuda-coarsening-compiler VERSION ${CudaCoarseningCompiler_VERSION} LANGUAGES C CXX CUDA)

find_package(LLVM CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
message(STATUS "Found LLVM in ${LLVM_DIR}")

add_subdirectory(llvm-rpc-passes)
add_subdirectory(rpc-runtime)

include(CMakePackageConfigHelpers)
install(
  TARGETS LLVMCUDACoarsening rpc-runtime
  EXPORT install_exports
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cuda-coarsening-compiler-config-version.cmake"
  VERSION ${CudaCoarseningCompiler_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(
  EXPORT install_exports
  FILE "coarsening-targets.cmake"
  DESTINATION lib/cmake
)

#set(COARSENING_INSTALL_LOCATION ${CMAKE_INSTALL_PREFIX})
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/cuda-coarsening-compiler-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cuda-coarsening-compiler-config.cmake
  @ONLY
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cuda-coarsening-compiler-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cuda-coarsening-compiler-config-version.cmake
  DESTINATION lib/cmake
)

# To apply coarsening within this project, use:
#
include("${CMAKE_SOURCE_DIR}/cmake/cuda-coarsening-compiler-config.cmake.in")
coarsening_compile(TARGET mt SOURCES test-kernels/matrix-transpose/matrix-transpose.cu)
coarsening_compile(TARGET copy SOURCES test-kernels/copy/copy.cu)
coarsening_compile(TARGET kernel_a SOURCES test-kernels/microkernels/kernel_a.cu)
