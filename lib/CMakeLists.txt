file(GLOB_RECURSE ZEPHYRUS_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Passes/*.cpp")

add_library(zephyrus_core STATIC ${ZEPHYRUS_SRC})

target_include_directories(zephyrus_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS})

    get_property(MLIR_DIALECT_LIBS     GLOBAL PROPERTY MLIR_DIALECT_LIBS)
    get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
    get_property(MLIR_CONVERSION_LIBS  GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
    
    target_link_libraries(zephyrus_core
      PUBLIC
        ${MLIR_DIALECT_LIBS}
        ${MLIR_TRANSLATION_LIBS}
        ${MLIR_CONVERSION_LIBS}
        LLVM
        HDF5::HDF5
        nlohmann_json::nlohmann_json
    )
    

install(TARGETS zephyrus_core ARCHIVE DESTINATION lib)
