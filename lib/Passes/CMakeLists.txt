file(GLOB_RECURSE SOURCE_FILES "*.cpp")

add_mlir_library(ZephyrusPasses
  ${SOURCE_FILES}

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRArithToLLVM
    MLIRLinalgToLoops
    MLIRTensorToMemRef
    MLIRSCFToControlFlow
    MLIRTosa
    MLIRTosaToLinalg
    MLIRPass
    MLIRTransforms
    MLIRLLVMIR
)