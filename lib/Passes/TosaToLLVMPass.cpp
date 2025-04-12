#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/LinalgToLoops/LinalgToLoops.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/TensorToMemRef/TensorToMemRef.h"
#include "mlir/Conversion/LLVMCommon/LowerToLLVM.h"

#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"


using namespace mlir;


namespace vortex {

  mlir::LogicalResult lowerTosaToLLVM(mlir::ModuleOp module){
    PassManager pm(module.getContext());
  
    // TOSA → Linalg
    pm.addPass(mlir::tosa::createTosaToLinalgPass());
  
    // Lowering boilerplate (only what's strictly required)
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createConvertTensorToMemrefPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
  
    // To LLVM
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertMemRefToLLVMPass());
    pm.addPass(mlir::createConvertArithToLLVMPass());
    pm.addPass(mlir::createLowerToLLVMPass());
  
    return pm.run(module);
  }
  

} // namespace vortex
