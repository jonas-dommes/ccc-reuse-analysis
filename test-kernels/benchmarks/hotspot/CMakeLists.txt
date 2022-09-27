set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_ARCHITECTURES 75)
# set (CUDA_TOOLKIT_ROOT_DIR /usr/lib/cuda)
enable_language(CUDA C CXX)

find_package(CUDAToolkit)

# FILE(GLOB Sources *.c *.cu )
FILE(GLOB sources *.c *.cu)
# file( GLOB LIB_HEADERS *.cuh utility.h )
# add_library( common ${LIB_SOURCES} ${LIB_HEADERS} )
# set_target_properties(common PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(hotspot ${sources})
# target_compile_options(hotspot PRIVATE -no-pie)
# set_target_properties(hotspot PROPERTIES CMAKE_POSITION_INDEPENDENT_CODE OFF)
# set_target_properties(kmeans PROPERTIES CUDA_SEPARABLE_COMPILATION ON CMAKE_POSITION_INDEPENDENT_CODE OFF)
# target_link_libraries(microkernels common ${CUDA_LIBRARIES} rt lcuda cuda)
