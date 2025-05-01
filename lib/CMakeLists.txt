# lib/CMakeLists.txt  – replace the whole file with this

# Collect every *.cpp under lib/ (pass & handler implementations)
file(GLOB_RECURSE ZEPHYRUS_SRC
     CONFIGURE_DEPENDS                 # re-run CMake if new files appear
     "${CMAKE_CURRENT_SOURCE_DIR}/Passes/*.cpp")

add_library(zephyrus_core STATIC ${ZEPHYRUS_SRC})

target_include_directories(zephyrus_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS})          # ← headers

target_link_libraries(zephyrus_core
  PUBLIC
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
    ${HDF5_C_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_HL_LIBRARIES})
