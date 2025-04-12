#pragma once
#include "mlir/Pass/Pass.h"

namespace zephyrus {

// Registers *your* passes
void registerAllPasses();

// Your custom pipelines
void buildHDF5ToLLVMPipeline(mlir::OpPassManager &pm);

} // namespace zephyrus
