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
#include "Passes/handlers/DenseAttr.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace vortex{
  void Conv2DHandler::handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t>& inputShape, mlir::Value& lastOutput) {
    setup(layer, builder);

    // Create constant ops for weights and biases.
    Value weightTensor = builder.create<tosa::ConstOp>(funcOp.getLoc(), weightType, weightAttr);
    Value biasTensor = builder.create<tosa::ConstOp>(funcOp.getLoc(), biasType, biasAttr);
    
  }

} // namespace vortex