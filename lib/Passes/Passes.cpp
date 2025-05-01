#include "Passes.h"
#include "Passes/HDF5ToTosaPass.h"
#include "Passes/TosaToLLVMPass.h"

namespace zephyrus {

void registerAllPasses() {
  // If your HDF5 pass is registered manually
  PassRegistration<HDF5ToTosaPass>(
    "hdf5-to-tosa", "Import a Keras HDF5 model into TOSA"
  );
}

void buildHDF5ToLLVMPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createHDF5ToTosaPass(/*modelFile=*/"TODO"));
  // Can also inline logic here:
  pm.addPass(mlir::createTosaToLinalgPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::createConvertTensorToMemrefPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createConvertMemRefToLLVMPass());
  pm.addPass(mlir::createConvertArithToLLVMPass());
  pm.addPass(mlir::createLowerToLLVMPass());
}

LogicalResult lowerHDF5ToLLVM(ModuleOp module, llvm::StringRef file) {
  PassManager pm(module.getContext());
  buildHDF5ToLLVMPipeline(pm, file);
  return pm.run(module);
}

} // namespace zephyrus
