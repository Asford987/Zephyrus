#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <nlohmann/json.hpp>
#include <vector>
#include "Passes/handlers/DenseHandler.h"
#include "Passes/handlers/DenseAttr.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace zephyrus{
  void DenseHandler::handleLayer(OpBuilder &builder,
        FuncOp       &funcOp,
        const json   &layer,
        std::vector<int64_t> &inputShape,
        Value        &lastOutput) {
    setup(layer, builder);

    Location loc = funcOp.getLoc();
    auto f32     = builder.getF32Type();

    Value weightTensor = builder.create<tosa::ConstOp>(loc, weightType,
                              weightAttr);
    Value biasTensor   = builder.create<tosa::ConstOp>(loc, biasType,
                              biasAttr);

    std::vector<int64_t> outShape = inputShape;
    outShape.back() = units;
    auto outType = RankedTensorType::get(outShape, f32);

    lastOutput = builder.create<tosa::FullyConnectedOp>(
    loc,
    outType,
    lastOutput,
    weightTensor,
    biasTensor);

    inputShape = outShape;
  }

} // namespace zephyrus