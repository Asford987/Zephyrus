#include "Passes/Passes.h"
#include "Passes/HDF5ToTosaPass.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"



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

  pm.addPass(tosa::createTosaToLinalgOnTensors());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createLowerToCFGPass());
  pm.addPass(createLowerToLLVMPass());  
}

LogicalResult lowerHDF5ToLLVM(ModuleOp module, llvm::StringRef file) {
  PassManager pm(module.getContext());
  buildHDF5ToLLVMPipeline(pm, file);
  return pm.run(module);
}

} // namespace zephyrus
