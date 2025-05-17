cmake_minimum_required(VERSION 3.16)
project(zephyrus LANGUAGES C CXX)

list(APPEND CMAKE_PREFIX_PATH "/opt/llvm2/lib/cmake")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX HL)

add_subdirectory(lib)
add_subdirectory(tools/zephyrus)

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h")

# set(CPACK_GENERATOR DEB)
# set(CPACK_PACKAGE_NAME           "zephyrus")
# set(CPACK_PACKAGE_VERSION        "0.1.0")
# set(CPACK_PACKAGE_CONTACT        "you@example.com")
# set(CPACK_DEBIAN_PACKAGE_SECTION "devel")

# set(CPACK_DEBIAN_PACKAGE_DEPENDS
#     "llvm-13-dev (>= 1:13), mlir-13-dev (>= 1:13), libhdf5-dev, libjsoncpp-dev")

# set(CMAKE_INSTALL_RPATH "/opt/llvm/lib")

# include(CPack)
