project(CudaCCoarseningCompiler LANGUAGES C CXX CUDA)

cmake_minimum_required(VERSION 3.17)
message(STATUS "Building CudaCCoarseningCompiler")
find_package(LLVM CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
message(STATUS "Found LLVM in ${LLVM_DIR}")

add_subdirectory(llvm-rpc-passes)
add_subdirectory(rpc-runtime)

set(CUDA_PATH /usr/local/cuda)
set(RPC_DEVICE_ARCH sm_52)
set(RPC_COMPUTE_ARCH compute_52)
set(OPT -O3)


# Generate a small test file. This is used to grep one of clang's internal commands
# Alternatively, use a real test file:
# set(TEST_FILE "${CMAKE_SOURCE_DIR}/test-kernels/matrix-transpose/matrix-transpose.cu")
set(TEST_FILE "coarsening_test_compile.cu")
execute_process(
  COMMAND echo " __global__ void foo(){} int main(int argc, char **argv){return 0;}"
  OUTPUT_FILE coarsening_test_compile.cu
  WORKING_DIRECTORY /tmp
)
execute_process(
  COMMAND clang++ ${TEST_FILE} -o mt --cuda-gpu-arch=sm_52 -L/usr/local/cuda/lib64 -lcudart -ldl -lrt -v
  ERROR_VARIABLE test_compile_cmd
  WORKING_DIRECTORY /tmp
)

string(REGEX MATCH "[^\n]* -cc1 -triple x86_64-unknown-linux-gnu[^\n]*" raw_cc1_call ${test_compile_cmd})
if(raw_cc1_call STREQUAL "")
  message(FATAL_ERROR "Failed to identify required internal command from test compile."
                      "Check if command can be identified differently."
                      "Aborting...")
endif()

# Setting up command for later use
string(REGEX REPLACE "-fcuda-include-gpubinary [^((?<!\\)\ )]*.fatbin(\")?" "" raw_cc1_call_1 ${raw_cc1_call})
string(REGEX REPLACE "-fdebug-compilation-dir /tmp" "" raw_cc1_call_2 ${raw_cc1_call_1})
string(REGEX REPLACE "-o /tmp/[^((?<!\\)\ )]*.o" "" raw_cc1_call_3 ${raw_cc1_call_2})
string(REGEX REPLACE "-main-file-name [^((?<!\\)\ )]*.cu" "" raw_cc1_call_4 ${raw_cc1_call_3})
string(REGEX REPLACE " -v " " " raw_cc1_call_5 ${raw_cc1_call_4})
string(REGEX REPLACE "-x cuda ${TEST_FILE}" "" raw_cc1_call_6 ${raw_cc1_call_5})
string(REGEX REPLACE " -emit-obj " " -emit-llvm " cc1_call_str ${raw_cc1_call_6})
separate_arguments(cc1_call UNIX_COMMAND ${cc1_call_str})


# This function will work with the standard cmake include mechanism
function(coarsening_compile)
  set(options)
  set(one_value_args TARGET)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(COARSENING
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (cc1_call STREQUAL "")
    message(FATAL_ERROR "This should never happen")
  endif()

  set(target_build_dir "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${COARSENING_TARGET}.dir")
  file(MAKE_DIRECTORY ${target_build_dir})
  set(link_file_list "")
  
  foreach(src_file ${COARSENING_SOURCES})
    get_filename_component(build_name_short "${src_file}" NAME_WLE)
    set(build_name_full "${target_build_dir}/${build_name_short}")
    get_filename_component(src_file_base_name "${src_file}" NAME)

    add_custom_command(
      OUTPUT
        "${build_name_full}_device.bc"
      COMMAND
        clang++
        -x cuda
	-c
	-emit-llvm
        ${OPT}
        "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}"
        "-I$<JOIN:$<TARGET_PROPERTY:${COARSENING_TARGET},INCLUDE_DIRECTORIES>, -I>"
        -Xclang -disable-O0-optnone
        #--cuda-path="${CUDA_PATH}"
        --cuda-gpu-arch="${RPC_DEVICE_ARCH}"
        --cuda-device-only
        -o "${build_name_short}_device.bc"
      WORKING_DIRECTORY
        "${target_build_dir}"
    )

    add_custom_command(
      OUTPUT
        "${build_name_full}_host.bc"
      COMMAND
        clang++
        -x cuda
	-c
	-emit-llvm
        ${OPT}
        "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}"
        "-I$<JOIN:$<TARGET_PROPERTY:${COARSENING_TARGET},INCLUDE_DIRECTORIES>, -I>"
        #--cuda-path="${CUDA_PATH}"
        --cuda-gpu-arch="${RPC_DEVICE_ARCH}"
        --cuda-host-only
        -o "${build_name_short}_host.bc"
      WORKING_DIRECTORY
        "${target_build_dir}"
    )

    add_custom_command(
      OUTPUT
        "${build_name_full}_device_coarsened.bc"
      COMMAND
        opt
        -load "$<TARGET_FILE:LLVMCUDACoarsening>"
        -mem2reg -indvars -structurizecfg -be
        -cuda-coarsening-pass
        -coarsened-kernel all
        -coarsening-mode dynamic
        -o "${build_name_short}_device_coarsened.bc"
         < "${build_name_short}_device.bc"
      WORKING_DIRECTORY
        "${target_build_dir}"
      DEPENDS
        "${build_name_full}_device.bc"
        LLVMCUDACoarsening
    )

    add_custom_command(
      OUTPUT
        "${build_name_full}_device_coarsened.ptx"
      COMMAND
        llc
        ${OPT}
        -mcpu=${RPC_DEVICE_ARCH}
        -o "${build_name_short}_device_coarsened.ptx"
        "${build_name_short}_device_coarsened.bc"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${build_name_full}_device_coarsened.bc"
    )

    add_custom_command(
      OUTPUT
        "${build_name_full}_device_coarsened.cubin"
      COMMAND
        ptxas
        -m64
        --gpu-name="${RPC_DEVICE_ARCH}"
        "${build_name_short}_device_coarsened.ptx"
        --output-file "${build_name_short}_device_coarsened.cubin"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${build_name_full}_device_coarsened.ptx"
    )

    add_custom_command(
      OUTPUT
        "${build_name_full}_device_coarsened.fatbin"
      COMMAND
        fatbinary
        -64
        --create "${build_name_short}_device_coarsened.fatbin"
        "--image=profile=${RPC_DEVICE_ARCH},file=${build_name_short}_device_coarsened.cubin"
        "--image=profile=${RPC_COMPUTE_ARCH},file=${build_name_short}_device_coarsened.ptx"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${build_name_full}_device_coarsened.ptx"
        "${build_name_full}_device_coarsened.cubin"
    )
   
    add_custom_command(
      OUTPUT
        "${build_name_full}_combined.bc"
      COMMAND
        ${cc1_call}
        -fcuda-include-gpubinary "${build_name_short}_device_coarsened.fatbin"
        -fdebug-compilation-dir "${target_build_dir}"
        -o "${build_name_full}_combined.bc"
        -main-file-name "${src_file_base_name}"
        -x cuda "${CMAKE_SOURCE_DIR}/${src_file}"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${build_name_full}_device_coarsened.fatbin"
        "${build_name_full}_host.bc"
    )

    add_custom_command(
      OUTPUT
        "${build_name_full}_combined_coarsened.bc"
      COMMAND
        opt
        -load "$<TARGET_FILE:LLVMCUDACoarsening>"
        -cuda-coarsening-pass
        -coarsened-kernel all
        -coarsening-mode dynamic
        -o "${build_name_short}_combined_coarsened.bc"
        < "${build_name_short}_combined.bc"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${build_name_full}_combined.bc"
        LLVMCUDACoarsening
    )

    add_custom_command(
      OUTPUT
        "${build_name_full}_coarsened.o"
      COMMAND
        llc
        ${OPT}
        -filetype=obj
        "${build_name_short}_combined_coarsened.bc"
        -o "${build_name_short}_coarsened.o"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${build_name_full}_combined_coarsened.bc"
    )

    list(APPEND link_file_list "${build_name_full}_coarsened.o")

  endforeach()

  add_custom_target(
    "${COARSENING_TARGET}"
    ALL
    clang
      "-L${CUDA_PATH}/lib64"
      -lcudart
      -ldl
      -lm
      -lstdc++
      "$<TARGET_OBJECTS:rpc-runtime>"
      ${link_file_list}
      -o "${CMAKE_CURRENT_BINARY_DIR}/${COARSENING_TARGET}"
    WORKING_DIRECTORY
       "${target_build_dir}"
    DEPENDS
      rpc-runtime
      ${link_file_list}
  )

endfunction()

coarsening_compile(TARGET mt SOURCES test-kernels/matrix-transpose/matrix-transpose.cu)
