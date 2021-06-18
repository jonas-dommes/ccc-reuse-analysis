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
set(RPC_DEVICE_ARCH sm_61)
set(RPC_COMPUTE_ARCH compute_61)


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
  
  foreach(src_file ${ADD_COARSENING_SOURCES})
    message(STATUS "${src_file}")
    string(REPLACE "/" "_" target_base_name "${src_file}")
    message(STATUS "${target_base_name}")
    
    get_directory_property (inc_dirs INCLUDE_DIRECTORIES)
    #list(TRANSFORM inc_dirs PREPEND "-I")
    #list(JOIN inc_dirs " " INCLUDE_STRING)
    foreach(dir ${inc_dirs})
      string(APPEND INCLUDE_COMPILER_STRING "-I${dir} ")
    endforeach()

    add_custom_target("${target_base_name}_device.bc" ALL
      clang++
        -x cuda -c -emit-llvm
        ${OPT}
        "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}"
        "-I$<JOIN:$<TARGET_PROPERTY:${target_base_name}_device.bc,INCLUDE_DIRECTORIES>, -I>"
        -Xclang -disable-O0-optnone
        --cuda-path="${CUDA_PATH}"
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
        --cuda-path="${CUDA_PATH}"
        --cuda-gpu-arch="${RPC_DEVICE_ARCH}"
        --cuda-host-only
        -o "${target_base_name}_host.bc"
      WORKING_DIRECTORY
        "${target_build_dir}"
    )
    add_custom_target("${target_base_name}_device_coarsened.bc" ALL
      echo -load "$<TARGET_FILE:LLVMCUDACoarsening>"
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
      echo llc
        $OPT
        -mcpu=${RPC_DEVICE_ARCH}
        -o "${target_base_name}_device_coarsened.ptx"
        "${target_base_name}_device_coarsened.bc"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${target_base_name}_device_coarsened.bc"
    )
    add_custom_target("${target_base_name}_device_coarsened.cubin" ALL
      echo $CUDA_PATH/bin/ptxas
        -m64
        --gpu-name=$RPC_DEVICE_ARCH
        "${target_base_name}_device_coarsened.ptx"
        --output-file "${target_base_name}_device_coarsened.cubin"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${target_base_name}_device_coarsened.ptx"
    )
    add_custom_target("${target_base_name}_device_coarsened.fatbin" ALL
      echo $CUDA_PATH/bin/fatbinary
        -64
        --create"${target_base_name}_device_coarsened.fatbin"
        "--image=profile=${RPC_DEVICE_ARCH},file=${target_base_name}_device_coarsened.cubin"
        "--image=profile=${RPC_COMPUTE_ARCH},file=${target_base_name}_device_coarsened.ptx"
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${target_base_name}_device_coarsened.ptx"
        "${target_base_name}_device_coarsened.cubin"
    )
    
    add_custom_target("${target_base_name}_combined.ll" ALL
      echo clang
        -cc1 $OPT -emit-llvm -triple x86_64-unknown-linux-gnu
        -x cuda
        -target-sdk-version=$CUDA_VERSION
        -aux-triple nvptx64-nvidia-cuda -mrelax-all
        -disable-free -main-file-name $INPUT_FILE
        -mrelocation-model static -mthread-model posix
        -mdisable-fp-elim -fmath-errno -masm-verbose
        -mconstructor-aliases -munwind-tables -fuse-init-array
        -target-cpu x86-64 -dwarf-column-info
        -debugger-tuning=gdb
        -resource-dir $RPC_LLVM_BUILD_DIR/lib/clang/9.0.0
        -internal-isystem $INCLUDE_DIR -internal-externc-isystem $INCLUDE_DIR
        -internal-isystem $RPC_LLVM_BUILD_DIR/lib/clang/9.0.0/include/cuda_wrappers
        -internal-isystem $CUDA_PATH/include
        -include __clang_cuda_runtime_wrapper.h
        -internal-isystem /usr/include/c++/8.3.0/
        -internal-isystem /usr/include/c++/8.3.0/x86_64-pc-linux-gnu
        -internal-isystem /usr/include/c++/8.3.0/backward
        -internal-isystem /usr/local/include
        -internal-isystem $RPC_LLVM_BUILD_DIR/lib/clang/9.0.0/include
        -internal-externc-isystem /usr/include
        -internal-isystem /usr/local/include
        -fdeprecated-macro
        -fdebug-compilation-dir $BUILD_DIR
        -ferror-limit 19 -fmessage-length 0 -fobjc-runtime=gcc
        -fcxx-exceptions -fexceptions -fdiagnostics-show-option
        -o $BUILD_DIR/rpc_combined.ll $INPUT_FILE
        -fcuda-include-gpubinary $BUILD_DIR/rpc_device.fatbin
        -faddrsig
      WORKING_DIRECTORY
         "${target_build_dir}"
      DEPENDS
        "${target_base_name}_device_coarsened.fatbin"
        "${target_base_name}_host.bc"
      )

    get_target_property(my_sys_dirs "${target_base_name}_combined.ll" INTERFACE_INCLUDE_DIRECTORIES)
    foreach(dir ${my_sys_dirs})
      string(APPEND INCLUDE_SYS_COMPILER_STRING "-I${dir} ")
    endforeach()
    message(STATUS "BAR ${my_sys_dirs}")
  endforeach() 
endfunction()

add_coarsening_to_target(TARGET my_mt SOURCES test-kernels/matrix-transpose/matrix-transpose.cu)

