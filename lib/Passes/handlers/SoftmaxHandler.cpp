#include <limits>
#include <vector>

#include <nlohmann/json.hpp>

#include "Passes/handlers/SoftmaxHandler.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <llvm/ADT/APFloat.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace zephyrus{
  void SoftmaxHandler::handleLayer(OpBuilder &builder, FuncOp &funcOp, const json  &layer, std::vector<int64_t> &inputShape, Value &lastOutput) {
    Location loc = funcOp.getLoc();
    auto f32 = builder.getF32Type();
    auto inputType   = RankedTensorType::get(inputShape, f32);

    int axis = static_cast<int>(inputShape.size()) - 1;

    std::vector<int64_t> reducedShape = inputShape;
    reducedShape[axis] = 1;
    auto reducedType = RankedTensorType::get(reducedShape, f32);

    Value maxVals = builder.create<tosa::ReduceMaxOp>(
    loc, reducedType, lastOutput,
    builder.getI64IntegerAttr(axis));

    Value maxValsBroadcast = builder.create<tosa::ReshapeOp>(
    loc, inputType, maxVals,
    builder.getDenseI64ArrayAttr(reducedShape));

    Value centered = builder.create<tosa::SubOp>(
    loc, inputType, lastOutput, maxValsBroadcast);

    Value expTensor = builder.create<tosa::ExpOp>(loc, inputType, centered);

    Value sumExp = builder.create<tosa::ReduceSumOp>(
    loc, reducedType, expTensor, builder.getI64IntegerAttr(axis));

    Value sumExpBroadcast = builder.create<tosa::ReshapeOp>(
    loc, inputType, sumExp,
    builder.getDenseI64ArrayAttr(reducedShape));

    Value reciprocal = builder.create<tosa::ReciprocalOp>(
    loc, inputType, sumExpBroadcast);

    Value softmaxOut = builder.create<tosa::MulOp>(
    loc, inputType, expTensor, reciprocal, 0u);

    lastOutput = softmaxOut;
  }


} // namespace zephyrus