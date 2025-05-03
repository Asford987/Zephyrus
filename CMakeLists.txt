cmake_minimum_required(VERSION 3.21)
project(zephyrus LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH "/opt/llvm/lib/cmake")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM  REQUIRED CONFIG)
execute_process(COMMAND llvm-config --libdir
                OUTPUT_VARIABLE LLVM_LIBDIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
link_directories(${LLVM_LIBDIR})
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_LIBRARY_DIRS})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX HL)

add_subdirectory(lib)
add_subdirectory(tools/zephyrus)

install(
  DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)


set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_NAME             "zephyrus")
set(CPACK_PACKAGE_VERSION          "0.1.0")
set(CPACK_PACKAGE_VENDOR           "Zephyrus Project")
set(CPACK_PACKAGE_CONTACT          "fabricio.asfora.001@gmail.com")
set(CPACK_PACKAGE_DESCRIPTION
    "MLIR-based neural-network ahead-of-time compiler for NAOv6 robots")
set(CPACK_DEBIAN_PACKAGE_SECTION   "devel")
set(CPACK_DEBIAN_FILE_NAME         DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

# Explicit runtime deps that dpkg-shlibdeps might miss (header-only JSON doesn’t need one)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libhdf5-103")

include(CPack)
