#pragma once
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <llvm/ADT/APFloat.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <vector>


using namespace mlir;

namespace zephyrus{
  // Converts std::vector<float> to DenseElementsAttr
  DenseElementsAttr convertToDenseAttr(Builder &builder, RankedTensorType type, const std::vector<float> &values){
    SmallVector<APFloat, 4> apValues;
    for (float v : values) {
        apValues.push_back(APFloat(v));
    }
    return DenseElementsAttr::get(type, apValues);
  }

} // namespace zephyrus