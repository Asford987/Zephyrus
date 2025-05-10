#include "Passes/Passes.h"
#include "Passes/HDF5ToTosaPass.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/TosaToStandard/TosaToStandard.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"


using namespace mlir;

namespace zephyrus {

static llvm::cl::opt<std::string> inputFilename(
  llvm::cl::Positional,
  llvm::cl::desc("<input file>"),
  llvm::cl::value_desc("filename"));

static PassRegistration<HDF5ToTosaPass> hdf5Reg(
  []() -> std::unique_ptr<mlir::Pass> { return createHDF5ToTosaPass(inputFilename); });

void buildHDF5ToLLVMPipeline(OpPassManager &pm, llvm::StringRef file) {
  pm.addPass(createHDF5ToTosaPass(file.str()));
  OpPassManager &fpm = pm.nest<FuncOp>();
  
  fpm.addPass(tosa::createTosaInferShapesPass());
  fpm.addPass(createCanonicalizerPass());
  fpm.addPass(tosa::createTosaToLinalgNamed());
  fpm.addPass(tosa::createTosaToLinalg());
  fpm.addPass(tosa::createTosaToStandard());
  fpm.addPass(tosa::createTosaToSCF());
  fpm.addPass(createCanonicalizerPass());
  
  fpm.addPass(createLinalgBufferizePass());
  fpm.addPass(createTensorBufferizePass());
  fpm.addPass(bufferization::createBufferDeallocationPass());
  fpm.addPass(bufferization::createFinalizingBufferizePass());

  fpm.addPass(createBufferizationToMemRefPass());

  fpm.addPass(createLowerAffinePass());
  fpm.addPass(createLowerToCFGPass());
  fpm.addPass(createCanonicalizerPass());
  
  pm.addPass(createConvertLinalgToStandardPass());

  OpPassManager &cleanup = pm.nest<FuncOp>();
  cleanup.addPass(createBufferizationToMemRefPass()); 
  cleanup.addPass(bufferization::createFinalizingBufferizePass());
  cleanup.addPass(createCanonicalizerPass()); 
  cleanup.addPass(arith::createConvertArithmeticToLLVMPass());
  
  pm.addPass(createConvertVectorToLLVMPass());

  OpPassManager &last = pm.nest<FuncOp>();
  last.addPass(createCanonicalizerPass());
  last.addPass(arith::createConvertArithmeticToLLVMPass());

  pm.addPass(createLowerToLLVMPass());
}

LogicalResult lowerHDF5ToLLVM(ModuleOp module, llvm::StringRef file) {
  PassManager pm(module.getContext());
  buildHDF5ToLLVMPipeline(pm, file);
  return pm.run(module);
}

} // namespace zephyrus
