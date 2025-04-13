cmake_minimum_required(VERSION 3.20.0)
project(Vortex LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17)

include(CMakeDependentOption)

#-------------------------------------------------------------------------------
# Options and definitions
#-------------------------------------------------------------------------------

option(VORTEX_BUILD_EMBEDDED "Build Vortex as part of another project" ON)
option(VORTEX_TOSA_USE_EIGEN "Enables use of Eigen library for some TOSA Ops." OFF)
option(VORTEX_INCLUDE_TESTS "Generate build targets for the MLIR Vortex unit tests." OFF)
cmake_dependent_option(VORTEX_TOSA_TEST_EIGEN "Enables testing of Eigen library for some TOSA Ops." ON "VORTEX_INCLUDE_TESTS;VORTEX_TOSA_USE_EIGEN" OFF)

#-------------------------------------------------------------------------------
# MLIR/LLVM configuration
#-------------------------------------------------------------------------------

find_package(HDF5 REQUIRED)
target_link_libraries(Vortex PRIVATE HDF5::HDF5)

find_package(MLIR REQUIRED CONFIG)


message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")


include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# TODO: With LLVM_ENABLE_WARNINGS set to ON in HandleLLVMOptions, the
# compilation of googletest fails. As a workaround, we remove the
# `-Wcovered-switch-default` flag.
string(REPLACE "-Wcovered-switch-default" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(VORTEX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VORTEX_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(VORTEX_MAIN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(VORTEX_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${VORTEX_MAIN_INCLUDE_DIR})
include_directories(${VORTEX_INCLUDE_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# Dependent projects and dependencies
#-------------------------------------------------------------------------------

# Optional Eigen dependency for some TOSA Ops
if(VORTEX_TOSA_USE_EIGEN)
  find_package(Eigen3 3.3.1 NO_MODULE)
  if(NOT TARGET Eigen3::Eigen)
    message(FATAL_ERROR "Should build with Eigen, but Eigen was not found.")
  endif()
endif()

# Dependency on GoogleTest. Used to unit test the reference implementation.
if(VORTEX_INCLUDE_TESTS)
  include(third_party/cmake-scripts/code-coverage.cmake)

  # With `LLVM_INSTALL_UTILS` set to `ON`, the LLVM/MLIR build dir contains a
  # gtest, added in llvm/utils/unittest/CMakeLists.txt. However, LLVM ships
  # gooletest version 1.8, whereas we require a more recent version. Therefore,
  # we require to use an installed MLIR if unittests should be build.
  if(TARGET gtest)
    message(FATAL_ERROR
      "Unitests can only be build if MLIR is installed, but MLIR_DIR seems to "
      "point to an MLIR build directory.")
  endif()
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)