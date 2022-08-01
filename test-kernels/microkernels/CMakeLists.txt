set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_ARCHITECTURES 75)
# set (CUDA_TOOLKIT_ROOT_DIR /usr/lib/cuda)
enable_language(C CXX CUDA)

FILE(GLOB Sources *.c *.cu )

add_executable(microkernels ${Sources})
target_link_libraries(microkernels ${CUDA_LIBRARY} rt lcuda cuda)
