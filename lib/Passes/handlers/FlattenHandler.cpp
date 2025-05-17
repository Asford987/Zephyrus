#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <llvm/ADT/APFloat.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <nlohmann/json.hpp>
#include <vector>
#include "Passes/handlers/FlattenHandler.h"
#include "Passes/handlers/DenseAttr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace zephyrus{
  void FlattenHandler::handleLayer(OpBuilder &builder, FuncOp &funcOp, const json &layer, std::vector<int64_t> &inputShape, Value &lastOutput) {
    Location loc = funcOp.getLoc();

    int64_t batch = inputShape[0];
    int64_t flat  = 1;
    for (size_t i = 1; i < inputShape.size(); ++i)
    flat *= inputShape[i];

    std::vector<int64_t> newShape = {batch, flat};

    auto outType = RankedTensorType::get(newShape, builder.getF32Type());

    lastOutput = builder.create<tosa::ReshapeOp>(
    loc,
    outType,
    lastOutput,
    builder.getDenseI64ArrayAttr(newShape));

    inputShape = newShape;
}


} // namespace zephyrus