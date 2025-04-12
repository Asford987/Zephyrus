#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <nlohmann/json.hpp>
#include <vector>
#include "Passes/handlers/DenseHandler.h"
#include "Passes/handlers/DenseAttr.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace zephyrus{
  void DenseHandler::handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t> &inputShape, mlir::Value &lastOutput) {
    setup(layer, builder);

    // Create constant ops for weights and biases.
    Value weightTensor = builder.create<tosa::ConstOp>(funcOp.getLoc(), weightType, weightAttr);
    Value biasTensor = builder.create<tosa::ConstOp>(funcOp.getLoc(), biasType, biasAttr);

    // Create TOSA MatMul and Add ops.
    // For this example, assume the output of matmul replaces the last dimension with "units".
    std::vector<int64_t> matmulOutputShape = inputShape;
    matmulOutputShape.back() = units;
    auto matmulOutputType = RankedTensorType::get(matmulOutputShape, builder.getF32Type());

    Value matmulOutput = builder.create<tosa::MatMulOp>(
        funcOp.getLoc(), matmulOutputType, lastOutput, weightTensor);
    lastOutput = builder.create<tosa::AddOp>(
        funcOp.getLoc(), matmulOutputType, matmulOutput, biasTensor);
    inputShape = matmulOutputShape;
  }

} // namespace zephyrus