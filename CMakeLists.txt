project(CudaCCoarseningCompiler LANGUAGES C CXX CUDA)

cmake_minimum_required(VERSION 3.17)
find_package(LLVM CONFIG)
  message(STATUS "Building CudaCCoarseningCompiler against LLVM configured from ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_subdirectory(llvm-rpc-passes)
add_subdirectory(rpc-runtime)

message(STATUS "CUDA compiler: ${CUDACXX}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
set(CUDA_PATH /usr/local/cuda)
set(RPC_DEVICE_ARCH sm_52)
set(RPC_COMPUTE_ARCH compute_52)
set(OPT -O3)

set(TEST_FILE "${CMAKE_SOURCE_DIR}/test-kernels/matrix-transpose/matrix-transpose.cu")
execute_process(
  COMMAND clang++ ${TEST_FILE} -o mt --cuda-gpu-arch=sm_52 -L/usr/local/cuda/lib64 -lcudart -ldl -lrt -v
  ERROR_VARIABLE test_compile_output
  WORKING_DIRECTORY /tmp
)

string(REGEX MATCH "[^\n]* -cc1 -triple x86_64-unknown-linux-gnu[^\n]*" raw_cc1_call ${test_compile_output})
#-fdebug-compilation-dir /tmp
#-fcuda-include-gpubinary /tmp/matrix-transpose-627796.fatbin
#-faddrsig
#-o /tmp/matrix-transpose-bf037f.o
#-x cuda /data/nico/code/llvm-cuda-coarsening/test-kernels/matrix-transpose/matrix-transpose.cu

#message(STATUS "RAW_CC1_CALL: ${raw_cc1_call}")
# string(REGEX REPLACE "-fcuda-include-gpubinary [^((?<!\\)\ )]*.fatbin(\")?" "-fcuda-include-gpubinary FATBIN_PLACEHOLDER" raw_cc1_call_1 ${raw_cc1_call})
# string(REGEX REPLACE "-fdebug-compilation-dir /tmp" "-fdebug-compilation-dir WORKDIR_PLACEHOLDER" raw_cc1_call_2 ${raw_cc1_call_1})
# string(REGEX REPLACE "-o /tmp/[^((?<!\\)\ )]*.o" "-o OUTFILE_PLACEHOLDER" raw_cc1_call_3 ${raw_cc1_call_2})
# string(REGEX REPLACE "-x cuda ${TEST_FILE}" "-x cuda CUDA_FILE_PLACEHOLDER" cc1_call_template ${raw_cc1_call_3})
string(REGEX REPLACE "-fcuda-include-gpubinary [^((?<!\\)\ )]*.fatbin(\")?" "" raw_cc1_call_1 ${raw_cc1_call})
string(REGEX REPLACE "-fdebug-compilation-dir /tmp" "" raw_cc1_call_2 ${raw_cc1_call_1})
string(REGEX REPLACE "-o /tmp/[^((?<!\\)\ )]*.o" "" raw_cc1_call_3 ${raw_cc1_call_2})
string(REGEX REPLACE "-main-file-name [^((?<!\\)\ )]*.cu" "" raw_cc1_call_4 ${raw_cc1_call_3})
string(REGEX REPLACE " -v " " " raw_cc1_call_5 ${raw_cc1_call_4})
string(REGEX REPLACE "-x cuda ${TEST_FILE}" "" raw_cc1_call_6 ${raw_cc1_call_5})
string(REGEX REPLACE " -emit-obj " " -emit-llvm " cc1_call_str ${raw_cc1_call_6})
message(STATUS "CC1_CALL: ${cc1_call_str}")
separate_arguments(cc1_call UNIX_COMMAND ${cc1_call_str})


#add_library(rpc-runtime OBJECT rpc-runtime/dynamic.cpp)
#target_link_libraries(rpc-runtime PUBLIC my_cpps)

#function(add_coarsening source_dir target suffix props)
#  add_executable(${target}
#end_function(add_coarsening)


#function(add_celerity_to_target)
#  set(options)
#  set(one_value_args TARGET)
#  set(multi_value_args SOURCES)
#  cmake_parse_arguments(ADD_CELERITY
#    "${options}"
#    "${one_value_args}"
#    "${multi_value_args}"
#    ${ARGN}
#  )
#
#  set_property(
#    TARGET ${ADD_CELERITY_TARGET}
#    APPEND PROPERTY LINK_LIBRARIES Celerity::celerity_runtime
#  )
#
#  add_sycl_to_target(
#    TARGET ${ADD_CELERITY_TARGET}
#    SOURCES ${ADD_CELERITY_SOURCES}
#  )
#endfunction()
include_directories(
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
)
function(add_coarsening_to_target)
  set(options)
  set(one_value_args TARGET)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(ADD_COARSENING
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  set(target_build_dir "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${ADD_COARSENING_TARGET}.dir")
  file(MAKE_DIRECTORY ${target_build_dir})
  set(link_file_list "")
  
  foreach(src_file ${ADD_COARSENING_SOURCES})
    message(STATUS "${src_file}")
    get_filename_component(src_file_wle "${src_file}" NAME_WLE)
    set(target_base_name "${src_file_wle}")
    message(STATUS "${target_base_name}")
    get_filename_component(src_file_base_name "${src_file}" NAME)
    message(STATUS "FILE BASE NAME: ${src_file_base_name}")
    
    add_custom_target("${target_base_name}_device.bc" ALL
      clang++
        -x cuda -c -emit-llvm
        ${OPT}
        "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}"
        "-I$<JOIN:$<TARGET_PROPERTY:${target_base_name}_device.bc,INCLUDE_DIRECTORIES>, -I>"
        -Xclang -disable-O0-optnone
        #--cuda-path="${CUDA_PATH}"
        --cuda-gpu-arch="${RPC_DEVICE_ARCH}"
        --cuda-device-only
        -o "${target_base_name}_device.bc"
      WORKING_DIRECTORY
        "${target_build_dir}"
    )
    add_custom_target("${target_base_name}_host.bc" ALL
      clang++
        -x cuda -c -emit-llvm
         ${OPT}
         "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}"
        "-I$<JOIN:$<TARGET_PROPERTY:${target_base_name}_host.bc,INCLUDE_DIRECTORIES>, -I>"
        #--cuda-path="${CUDA_PATH}"
        --cuda-gpu-arch="${RPC_DEVICE_ARCH}"
        --cuda-host-only
        -o "${target_base_name}_host.bc"
      WORKING_DIRECTORY
        "${target_build_dir}"
    )
    add_custom_target("${target_base_name}_device_coarsened.bc" ALL
      opt -load "$<TARGET_FILE:LLVMCUDACoarsening>"
        -mem2reg -indvars -structurizecfg -be
        -cuda-coarsening-pass
        -coarsened-kernel all
        -coarsening-mode dynamic
        -o "${target_base_name}_device_coarsened.bc"
         < "${target_base_name}_device.bc"
      WORKING_DIRECTORY
        "${target_build_dir}"
      DEPENDS
        "${target_base_name}_device.bc"
        LLVMCUDACoarsening
    )
    add_custom_target("${target_base_name}_device_coarsened.ptx" ALL
      llc
        ${OPT}
        -mcpu=${RPC_DEVICE_ARCH}
        -o "${target_base_name}_device_coarsened.ptx"
        "${target_base_name}_device_coarsened.bc"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${target_base_name}_device_coarsened.bc"
    )
    add_custom_target("${target_base_name}_device_coarsened.cubin" ALL
      ptxas
        -m64
        --gpu-name="${RPC_DEVICE_ARCH}"
        "${target_base_name}_device_coarsened.ptx"
        --output-file "${target_base_name}_device_coarsened.cubin"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${target_base_name}_device_coarsened.ptx"
    )
    add_custom_target("${target_base_name}_device_coarsened.fatbin" ALL
      fatbinary
        -64
        --create "${target_base_name}_device_coarsened.fatbin"
        "--image=profile=${RPC_DEVICE_ARCH},file=${target_base_name}_device_coarsened.cubin"
        "--image=profile=${RPC_COMPUTE_ARCH},file=${target_base_name}_device_coarsened.ptx"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${target_base_name}_device_coarsened.ptx"
        "${target_base_name}_device_coarsened.cubin"
    )
   
    message(STATUS
      "${cc1_call}
        -fcuda-include-gpubinary ${target_base_name}_device_coarsened.fatbin
        -fdebug-compilation-dir ${target_build_dir}
        -o ${target_base_name}_combined.bc
        -main-file-name ${src_file_base_name}
        -x cuda ${CMAKE_SOURCE_DIR}/${src_file}"
    )
    add_custom_target("${target_base_name}_combined.bc" ALL
      ${cc1_call}
        -fcuda-include-gpubinary "${target_base_name}_device_coarsened.fatbin"
        -fdebug-compilation-dir "${target_build_dir}"
        -o "${target_base_name}_combined.bc"
        -main-file-name "${src_file_base_name}"
        -x cuda "${CMAKE_SOURCE_DIR}/${src_file}" -v
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${target_base_name}_device_coarsened.fatbin"
        "${target_base_name}_host.bc"
    )

    add_custom_target("${target_base_name}_combined_coarsened.bc" ALL
      opt
        -load "$<TARGET_FILE:LLVMCUDACoarsening>"
        -cuda-coarsening-pass
        -coarsened-kernel all
        -coarsening-mode dynamic
        -o "${target_base_name}_combined_coarsened.bc"
        < "${target_base_name}_combined.bc"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${target_base_name}_combined.bc"
        LLVMCUDACoarsening
    )

    add_custom_target("${target_base_name}_out" ALL
      llc ${OPT}
        -filetype=obj
        "${target_base_name}_combined_coarsened.bc"
        -o "${target_base_name}_out"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${target_base_name}_combined_coarsened.bc"
    )

    list(APPEND link_file_list "${target_base_name}_out")

  endforeach()

  add_custom_target("${ADD_COARSENING_TARGET}" ALL
    clang
      -L$CUDA_PATH/lib64
      -lcudart
      -ldl
      -lm
      -lstdc++
      "$<TARGET_OBJECTS:rpc-runtime>"
      ${link_file_list}
      -o "${CMAKE_CURRENT_BINARY_DIR}/${ADD_COARSENING_TARGET}"
    WORKING_DIRECTORY
       "${target_build_dir}"
    DEPENDS
      rpc-runtime
      ${link_file_list}
  )

endfunction()

add_coarsening_to_target(TARGET my_mt SOURCES test-kernels/matrix-transpose/matrix-transpose.cu)

