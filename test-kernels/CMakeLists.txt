cmake_minimum_required (VERSION 3.17)

if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    find_package(CUDA)
    set(CUDA_ARCH_LIST Auto CACHE STRING
        "List of CUDA architectures (e.g. Pascal, Amphere, Volta, etc.) or \
        compute capability version (6.1, 7.0, etc.) to generate code for. \
        Set to Auto for automatic detection (default).")
    cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
endif()

# include_directories(common ../common)

# set(exe_files
#     ${CMAKE_CURRENT_LIST_DIR}/common/main.cu
# )


add_subdirectory(microkernels)
# add_subdirectory(benchmarks)
