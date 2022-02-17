if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

add_llvm_library( MemoryAccessAnalysis MODULE BUILDTREE_ONLY
  MemoryAccessAnalysis.cpp

  DEPENDS
  intrinsics_gen
  PLUGIN_TOOL
  opt
  )

target_include_directories(MemoryAccessAnalysis PUBLIC ${LLVM_INCLUDE_DIRS})
