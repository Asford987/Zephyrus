cmake_minimum_required(VERSION 3.20)
project(zephyrus LANGUAGES C CXX)

# Compiler config
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enforce static linking where possible
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})

# Find packages
find_package(MLIR REQUIRED CONFIG)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(nlohmann_json REQUIRED)

message(STATUS "MLIR dir: ${MLIR_DIR}")
message(STATUS "LLVM dir: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(HandleLLVMOptions)
include(LLVMConfig)
include(AddLLVM)
include(AddMLIR)

include(AddMLIR)
include(TableGen)
include(HandleLLVMOptions)

# Static linking hints
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

# Add your code
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(src)

# # Optional tests
# option(ZEPHYRUS_INCLUDE_TESTS "Build tests" OFF)
# if(ZEPHYRUS_INCLUDE_TESTS)
#   enable_testing()
#   add_subdirectory(test)
# endif()

# Packaging
set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME "zephyrus")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_CONTACT "fabricio.asfora.001@gmail.com")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Fabricio Asfora")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6")  # Minimal runtime dependency
set(CPACK_PACKAGE_DESCRIPTION "Zephyrus MLIR-based Neural Network Compiler")
set(CPACK_DEBIAN_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}_${CPACK_DEBIAN_ARCHITECTURE}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "zephyrus")

include(CPack)

# Install targets (compiler + headers)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS zephyrus DESTINATION bin)
