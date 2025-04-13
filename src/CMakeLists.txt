# src/CMakeLists.txt

# Find all .cpp files under src/
file(GLOB_RECURSE SRC_FILES "*.cpp")

# Build the compiler binary
add_executable(zephyrus ${SRC_FILES})

# Include directories
target_include_directories(zephyrus
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Link libraries (Zephyrus passes + MLIR + HDF5 + JSON)
target_link_libraries(zephyrus
  PRIVATE
    ZephyrusPasses
    MLIRSupport
    MLIRIR
    MLIRParser
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRArithToLLVM
    MLIRLLVMIR
    MLIRTransforms
    MLIRTosa
    MLIRTosaToLinalg
    MLIRExecutionEngine
    HDF5::HDF5
    nlohmann_json::nlohmann_json
)

# Force static linking (if supported)
target_link_options(zephyrus PRIVATE -static)

# Install the compiler binary to /usr/bin (via `make install` or CPack)
install(TARGETS zephyrus DESTINATION bin)
