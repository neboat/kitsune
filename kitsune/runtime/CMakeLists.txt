# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#
cmake_minimum_required(VERSION 3.20.0)

project(kitrt C CXX)

# -----------------------------------------------------------------------------
#
# This file cannot use any LLVM_* or CLANG_* or KITSUNE_* variables because this
# project is built as an external project and does not have any access to
# variables set in the main LLVM project. Everything needed here, including any
# configure time options must be passed explicitly to the ExternalProject_Add
# command in the top-level kitsune/CMakeLists.txt file in ../CMakeLists.txt.
#
# -----------------------------------------------------------------------------

# Since this is built as an external project, we have to explicitly find
# libLLVM. This is guaranteed to have been built, but we need to be careful that
# we don't pick up anything from the system.
#
# NOTE: We have removed the dependence on libLLVM for now, but we keep this
# here for now. There is a lot of unused code in the runtime, most of which
# should probably be deleted but is still kept around in case we decide to
# resurrect something. If we do, some of it might require libLLVM.
find_library(LIB_LLVM
  NAMES LLVM
  LLVM-${LLVM_VERSION_MAJOR}
  LLVM-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}
  REQUIRED
  PATHS ${LLVM_LIBRARY_OUTPUT_INTDIR}
  NO_DEFAULT_PATH)

# ${KITSUNE_LIBNAME} is passed in by cmake when the runtime is configured.
set(KITSUNE_OBJLIB ${KITSUNE_LIBNAME}_objlib)

add_library(${KITSUNE_OBJLIB} OBJECT
  kitrt.cpp
  memory.cpp
  memory_map.cpp)

# The KITSUNE_CUDA_* variables are defined in kitsune/CMakeLists.txt. They are
# passed in via CMake because the Kitsune runtime is treated as an external
# project.
if (KITSUNE_CUDA_ENABLED)
  target_sources(${KITSUNE_OBJLIB} PUBLIC
    cuda/kitcuda.cpp
    cuda/dylib_support.cpp
    cuda/launching.cpp
    cuda/memory.cpp
    cuda/streams.cpp)

  # FIXME: The only use of KITRT_CUDA_ENABLED seem to be some runtime
  # initialization code that is probably obsolete.
  target_compile_definitions(${KITSUNE_OBJLIB} PUBLIC
    KITRT_CUDA_ENABLED)

  target_include_directories(${KITSUNE_OBJLIB} SYSTEM PUBLIC
    ${KITSUNE_CUDA_INCLUDE_DIR})

  target_include_directories(${KITSUNE_OBJLIB} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

  target_link_directories(${KITSUNE_OBJLIB} PUBLIC
    ${KITSUNE_CUDA_LIB_DIRS})

  target_link_libraries(${KITSUNE_OBJLIB} PUBLIC
    ${KITSUNE_CUDA_LIB_LIBS})

  if (KITCUDA_ENABLE_NVTX)

    target_compile_definitions(${KITSUNE_OBJLIB} PUBLIC
      KITCUDA_ENABLE_NVTX)

  endif()

  find_program(LLVM_LINK llvm-link PATHS ${LLVM_BINARY_DIR}/bin NO_DEFAULT_PATH)
  find_program(LLVM_CLANGXX clang++ PATHS ${LLVM_BINARY_DIR}/bin NO_DEFAULT_PATH)

  set(KITCUDA_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/kitcuda.bc)
  set(KITCUDA_BC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/scan_inlined.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/reduce_inlined.cu
  )
  set(KITCUDA_INDIVIDUAL_BC_FILES "")
  # Loop through each source file and generate bitcode
  foreach(CU_SOURCE_FILE ${KITCUDA_BC_SOURCES})
    get_filename_component(CU_FILENAME ${CU_SOURCE_FILE} NAME_WE) # Get base name without extension
    set(CU_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/${CU_FILENAME}-cuda-nvptx64-nvidia-cuda-${KITSUNE_CUDA_ARCH_DEFAULT}.bc)
    add_custom_command(
      OUTPUT ${CU_BC_FILE}
      COMMAND ${LLVM_CLANGXX} -x cuda -O2 ${CU_SOURCE_FILE} -DGEN_BITCODE
                              --cuda-gpu-arch=${KITSUNE_CUDA_ARCH_DEFAULT} -c -emit-llvm
                              --no-offload-new-driver
                               -gline-directives-only
      DEPENDS ${CU_SOURCE_FILE}
      COMMENT "Generating CUDA bitcode for ${CU_FILENAME}.cu"
      VERBATIM
    )
    list(APPEND KITCUDA_INDIVIDUAL_BC_FILES ${CU_BC_FILE})
  endforeach()
  set(KITCUDA_BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/kitcuda.bc)
  add_custom_command(
    OUTPUT ${KITCUDA_BC_FILE}
    COMMAND ${LLVM_LINK} -o ${KITCUDA_BC_FILE} ${KITCUDA_INDIVIDUAL_BC_FILES}
    DEPENDS ${KITCUDA_INDIVIDUAL_BC_FILES} # Depends on all intermediate files
    COMMENT "Linking CUDA bitcode modules into kitcuda.bc"
    VERBATIM
  )
  add_custom_target(kitcuda_bc_target ALL DEPENDS ${KITCUDA_BC_FILE})
  add_dependencies(${KITSUNE_OBJLIB} kitcuda_bc_target)
  install(FILES ${KITCUDA_BC_FILE} DESTINATION ${CLANG_RESOURCE_SUBDIR}/lib)
endif()

# The KITSUNE_HIP_* variables are defined in kitsune/CMakeLists.txt. They are
# passed in via CMake because the Kitsune runtime is treated as an external
# project.
#
# Note on AMD HIP CMake details...  When is HIP not HIP?  When it is HIP...
# It is easy to confuse requirements for HIP as the HIP "language" vs. the
# HIP runtime. Unfortunately, these details seem to be a moving target with
# each release of rocm/hip.
if (KITSUNE_HIP_ENABLED)
  target_sources(${KITSUNE_OBJLIB} PUBLIC
    hip/kithip.cpp
    hip/dylib_support.cpp
    hip/launching.cpp
    hip/memory.cpp
    hip/streams.cpp)

  target_compile_definitions(${KITSUNE_OBJLIB} PUBLIC
    # FIXME: The only use of KITRT_HIP_ENABLED seem to be some runtime
    # initialization code that is probably obsolete.
    KITRT_HIP_ENABLED

    # FIXME: It looks like __HIP_PLATFORM_AMD__ is defined in the kitsune
    # runtime headers for hip. This is probably unnecessary. Alternatively, we
    # could remove it from those headers and only include it here if is required
    # to be defined before any hip headers are included.
    __HIP_PLATFORM_AMD__)

  target_include_directories(${KITSUNE_OBJLIB} PUBLIC
    ${KITSUNE_HIP_INCLUDE_DIR})

  target_link_directories(${KITSUNE_OBJLIB} PUBLIC
    ${KITSUNE_HIP_LIB_DIRS})

  target_link_libraries(${KITSUNE_OBJLIB} PUBLIC
    ${KITSUNE_HIP_LIB_LIBS})
endif()

# The pthreads tapir target is always enabled
target_sources(${KITSUNE_OBJLIB} PUBLIC
  pthreads/kitpthr.cpp)

set_target_properties(${KITSUNE_OBJLIB} PROPERTIES
  POSITION_INDEPENDENT_CODE 1)

target_include_directories(${KITSUNE_OBJLIB} PUBLIC
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIR}
  ${KITSUNE_SOURCE_DIR}/include
  ${KITSUNE_RUNTIME_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})

target_link_directories(${KITSUNE_OBJLIB} PUBLIC
  ${KITSUNE_COMMON_LIB_DIRS})

target_link_libraries(${KITSUNE_OBJLIB} PUBLIC
  ${KITSUNE_COMMON_LIB_LIBS})

# Create the static library
add_library(${KITSUNE_LIBNAME_STATIC} STATIC
  $<TARGET_OBJECTS:${KITSUNE_OBJLIB}>)

set_target_properties(${KITSUNE_LIBNAME_STATIC} PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CLANG_RESOURCE_INTDIR}/lib)

# Create the shared library
add_library(${KITSUNE_LIBNAME} SHARED
  $<TARGET_OBJECTS:${KITSUNE_OBJLIB}>)

set_target_properties(${KITSUNE_LIBNAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CLANG_RESOURCE_INTDIR}/lib)

get_target_property(KITSUNE_LINK_LIBRARIES ${KITSUNE_OBJLIB} LINK_LIBRARIES)
foreach (LIB IN LISTS KITSUNE_LINK_LIBRARIES)
  target_link_libraries(${KITSUNE_LIBNAME} PUBLIC ${LIB})
endforeach()

# $ORIGIN is the location of libkitrt.so. This is
#
#     <prefix>/lib/clang/${LLVM_VERSION_MAJOR}/lib
#
# where <prefix> is either the build or the install prefix. Right now, there is
# no dependence on libLLVM.so, but in case that is needed in the future, add an
# rpath to the top-level lib directory, i.e. <prefix>/lib. This should be safe
# in all cases.
set_property(TARGET ${KITSUNE_LIBNAME} APPEND
  PROPERTY BUILD_RPATH "$ORIGIN/../../../"
  PROPERTY INSTALL_RPATH "$ORIGIN/../../../")

# Install the libraries into a subdirectory of CMAKE_INSTALL_PREFIX. This
# subdirectory reflects the subdirectory of the top-level LLVM's install prefix
# into which the libraries will eventually be installed. This makes it easy to
# copy these into the corresponding build directory root which allows us to
# run everything from the build directory. The top-level CMakeLists.txt for
# Kitsune will ensure that everything is installed to the correct subdirectory
# of LLVM's install prefix.
install(TARGETS ${KITSUNE_LIBNAME} ${KITSUNE_LIBNAME_STATIC}
  DESTINATION ${CLANG_RESOURCE_SUBDIR}/lib)
