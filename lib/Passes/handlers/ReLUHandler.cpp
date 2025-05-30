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
#include "Passes/handlers/ReLUHandler.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace zephyrus{
  void ReLUHandler::handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t>& inputShape, mlir::Value& lastOutput){
    Location loc = funcOp.getLoc();
    auto f32 = builder.getF32Type();
    auto inputType = RankedTensorType::get(inputShape, f32);

    auto zero = builder.getF32FloatAttr(0.0f);
    auto large = builder.getF32FloatAttr(std::numeric_limits<float>::max());

    lastOutput = builder.create<tosa::ClampOp>(loc, inputType, lastOutput, zero, large);

  }

} // namespace zephyrus