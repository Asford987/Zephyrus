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
#include "Passes/handlers/LayerHandler.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace vortex{
  // Converts std::vector<float> to DenseElementsAttr
  auto convertToReLUAttr(Builder &builder, RankedTensorType type, const std::vector<float> &values);

  class ReLUHandler : public LayerHandler {
    public:
      void handleLayer(Builder& builder, FuncOp& funcOp, json& layer);
  };


} // namespace vortex