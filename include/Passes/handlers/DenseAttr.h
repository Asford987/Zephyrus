#pragma once
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <llvm/ADT/APFloat.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <vector>

using namespace mlir;

namespace zephyrus{
  DenseElementsAttr convertToDenseAttr(Builder &builder, RankedTensorType type, const std::vector<float> &values);
} // namespace zephyrus