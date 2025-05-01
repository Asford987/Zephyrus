#pragma once
#include "mlir/Pass/Pass.h"

namespace zephyrus {

// Registers *your* passes
void registerAllPasses();

// Your custom pipelines
void buildHDF5ToLLVMPipeline(mlir::OpPassManager &pm);

mlir::LogicalResult lowerHDF5ToLLVM(mlir::ModuleOp module, llvm::StringRef hdf5File);

} // namespace zephyrus
