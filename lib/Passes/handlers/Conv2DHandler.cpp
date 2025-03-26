#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <llvm/ADT/APFloat.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <nlohmann/json.hpp>
#include <vector>
#include "Passes/handlers/Conv2DHandler.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace vortex{
  // Converts std::vector<float> to DenseElementsAttr
  auto convertToConv2DAttr(Builder &builder, RankedTensorType type, const std::vector<float> &values) {
    SmallVector<APFloat, 4> apValues;
    for (float v : values) {
        apValues.push_back(APFloat(v));
    }
    return DenseElementsAttr::get(type, apValues);
  }

  void Conv2DHandler::handleLayer(OpBuilder& builder, FuncOp& funcOp, json& layer){

  }

} // namespace vortex