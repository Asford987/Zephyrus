file(GLOB_RECURSE ZEPHYRUS_SRC
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/Passes/*.cpp")

add_library(zephyrus_core STATIC ${ZEPHYRUS_SRC})

target_include_directories(zephyrus_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS})

target_link_libraries(zephyrus_core
  PRIVATE
    MLIRBufferizationToMemRef
    MLIRFuncToLLVM
    MLIRLinalgToLLVM
    MLIRMemRefToLLVM
    MLIRTosaToLinalg
    MLIRArithToLLVM
    MLIRSCFToControlFlow
    MLIRTosaTransforms
    MLIRConversionPasses
    MLIRLowerToLLVM
    MLIRLinalgTransforms
    MLIRTosaOps
    MLIRSupport
    MLIRIR
    LLVMSupport
    ${HDF5_C_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_HL_LIBRARIES})

set_target_properties(zephyrus_core PROPERTIES
  LINK_SEARCH_START_STATIC TRUE
  LINK_SEARCH_END_STATIC   TRUE
)

install(TARGETS zephyrus_core
          ARCHIVE DESTINATION lib)