#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace zephyrus {

/// Lowers a module containing TOSA dialect ops to LLVM dialect IR.
/// This includes the required intermediate passes (TOSA → Linalg → Loops → LLVM).
mlir::LogicalResult lowerTosaToLLVM(mlir::ModuleOp module);

} // namespace zephyrus
