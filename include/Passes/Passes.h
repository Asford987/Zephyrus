#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace zephyrus {

void registerAllPasses();

void buildHDF5ToLLVMPipeline(mlir::OpPassManager &pm,
                             llvm::StringRef hdf5File);

mlir::LogicalResult lowerHDF5ToLLVM(mlir::ModuleOp  module,
                                    llvm::StringRef hdf5File);
} // namespace zephyrus
