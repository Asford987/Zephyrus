#pragma once
#include "mlir/IR/BuiltinOps.h"          // mlir::ModuleOp
#include "mlir/Pass/Pass.h"              // mlir::LogicalResult / OpPassManager
#include "llvm/ADT/StringRef.h"

namespace zephyrus {

/// Registers every Zephyrus pass with MLIR.
void registerAllPasses();

/// Builds the full HDF5→LLVM lowering pipeline.
void buildHDF5ToLLVMPipeline(mlir::OpPassManager &pm,
                             llvm::StringRef hdf5File);

/// Runs that pipeline on \p module.
mlir::LogicalResult lowerHDF5ToLLVM(mlir::ModuleOp  module,
                                    llvm::StringRef hdf5File);
} // namespace zephyrus
