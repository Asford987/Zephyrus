#include "Passes/Passes.h"
#include "Passes/HDF5ToTosaPass.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"


using namespace mlir;

namespace zephyrus {

static llvm::cl::opt<std::string> inputFilename(
  llvm::cl::Positional,
  llvm::cl::desc("<input file>"),
  llvm::cl::value_desc("filename"));

static PassRegistration<HDF5ToTosaPass> hdf5Reg(
  []() -> std::unique_ptr<mlir::Pass> { return createHDF5ToTosaPass(inputFilename); });

void buildHDF5ToLLVMPipeline(OpPassManager &pm, llvm::StringRef hdf5Path) {
  pm.addPass(createHDF5ToTosaPass(hdf5Path.str()));

  pm.addNestedPass<func::FuncOp>(tosa::createTosaToTensor());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalgNamed());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToArith());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToSCF());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

  bufferization::OneShotBufferizationOptions opts;
  opts.bufferizeFunctionBoundaries = true;
  opts.allowReturnAllocs           = true;
  pm.addPass(bufferization::createOneShotBufferizePass(opts));

  pm.addNestedPass<func::FuncOp>(bufferization::createFinalizingBufferizePass());
  pm.addPass(createBufferizationToMemRefPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(memref::createExpandStridedMetadataPass()); 
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertVectorToLLVMPass()); 
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

}


LogicalResult lowerHDF5ToLLVM(ModuleOp module, llvm::StringRef file) {
  PassManager pm(module.getContext());
  buildHDF5ToLLVMPipeline(pm, file);
  return pm.run(module);
}

} // namespace zephyrus
