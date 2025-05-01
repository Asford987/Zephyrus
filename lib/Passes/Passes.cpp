#include "Passes/Passes.h"
#include "Passes/HDF5ToTosaPass.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"



using namespace mlir;

namespace zephyrus {

//— --------------------------------------------------------------------------
// 1.  Pass registration (new API = zero-arg)
//— --------------------------------------------------------------------------

  // Define a command-line option for the input file.
  static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input file>"),
    llvm::cl::value_desc("filename"));

  // Update the registration to pass the file name from the command-line.
  static PassRegistration<HDF5ToTosaPass> hdf5Reg(
    []() -> std::unique_ptr<mlir::Pass> { return createHDF5ToTosaPass(inputFilename); });
//— --------------------------------------------------------------------------
// 2.  Pipeline builder
//— --------------------------------------------------------------------------
void buildHDF5ToLLVMPipeline(OpPassManager &pm, llvm::StringRef file) {
  pm.addPass(createHDF5ToTosaPass(file.str()));

  pm.addPass(tosa::createTosaToLinalg());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

//— --------------------------------------------------------------------------
// 3.  Driver convenience wrapper
//— --------------------------------------------------------------------------
LogicalResult lowerHDF5ToLLVM(ModuleOp module, llvm::StringRef file) {
  PassManager pm(module.getContext());
  buildHDF5ToLLVMPipeline(pm, file);
  return pm.run(module);
}

} // namespace zephyrus
