cmake_minimum_required(VERSION 3.21)
project(zephyrus LANGUAGES CXX)

# ---------- LLVM / MLIR -------------------------------------------------
# If LLVM lives in /opt/llvm, expose its CMake configs:
list(APPEND CMAKE_PREFIX_PATH "/opt/llvm/lib/cmake")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM  REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_LIBRARY_DIRS})

# Export compile_commands.json for clangd / VSCode
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use C++17 everywhere
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# ----- external deps ----------------------------------------------------
find_package(nlohmann_json 3.2.0 REQUIRED)      # imports nlohmann_json::nlohmann_json
find_package(HDF5 REQUIRED COMPONENTS CXX HL)  # HDF5::HDF5, etc.


# ---------- Sub-dirs ----------------------------------------------------
add_subdirectory(lib)            # builds libzephyrus*
add_subdirectory(tools/zephyrus) # builds the CLI

# ---------- Install headers only once ----------------------------------
install(
  DIRECTORY include/
  DESTINATION include            # -> /opt/llvm/include/…
  FILES_MATCHING PATTERN "*.h"
)


# -----------------------------------------------------------------------
#  DEB package
# -----------------------------------------------------------------------
set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_NAME             "zephyrus")
set(CPACK_PACKAGE_VERSION          "0.1.0")           # <— bump on every tag
set(CPACK_PACKAGE_VENDOR           "Zephyrus Project")
set(CPACK_PACKAGE_CONTACT          "you@example.com")
set(CPACK_PACKAGE_DESCRIPTION
    "MLIR-based neural-network ahead-of-time compiler for NAOv6 robots")
set(CPACK_DEBIAN_PACKAGE_SECTION   "devel")
set(CPACK_DEBIAN_FILE_NAME         DEB-DEFAULT)       # zephyrus_0.1.0_amd64.deb
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)                # auto-detect shared-lib deps

# Explicit runtime deps that dpkg-shlibdeps might miss (header-only JSON doesn’t need one)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libhdf5-103")

include(CPack)     # <-- MUST be last!
