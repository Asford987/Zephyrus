# Collect all source files from the Passes directory
file(GLOB_RECURSE ZEPHYRUS_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Passes/*.cpp")

# Create a static library (shared could work too if needed)
add_library(zephyrus_core STATIC ${ZEPHYRUS_SRC})

# Add include paths
target_include_directories(zephyrus_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS})

    get_property(MLIR_DIALECT_LIBS     GLOBAL PROPERTY MLIR_DIALECT_LIBS)     # dialect impls【turn12search1】【turn12search0】
    get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS) # e.g. MLIRTargetLLVMIRExport【turn2search0】
    get_property(MLIR_CONVERSION_LIBS  GLOBAL PROPERTY MLIR_CONVERSION_LIBS)  # e.g. MemRefToLLVM
    
    # ---------------------------------------------------------------------------
    # 4. Link everything together
    # ---------------------------------------------------------------------------
    target_link_libraries(zephyrus_core
      PUBLIC
        ${MLIR_DIALECT_LIBS}
        ${MLIR_TRANSLATION_LIBS}
        ${MLIR_CONVERSION_LIBS}
        LLVM                         # single shared lib produced by LLVM_LINK_LLVM_DYLIB
        HDF5::HDF5
        nlohmann_json::nlohmann_json
    )
    

# Install only the archive (static library)
install(TARGETS zephyrus_core ARCHIVE DESTINATION lib)
