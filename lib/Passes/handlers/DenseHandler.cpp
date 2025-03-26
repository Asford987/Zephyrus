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
#include "Passes/handlers/DenseHandler.h"
#include "DenseHandler.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace vortex{
  // Converts std::vector<float> to DenseElementsAttr
  auto convertToDenseAttr(Builder &builder, RankedTensorType type, const std::vector<float> &values) {
    SmallVector<APFloat, 4> apValues;
    for (float v : values) {
        apValues.push_back(APFloat(v));
    }
    return DenseElementsAttr::get(type, apValues);
  }

  void DenseHandler::handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t> &inputShape, mlir::Value &lastOutput) {
    // Extract weight and bias data.
    std::vector<float> weightData, biasData;
    if (layer.contains("weights") && layer["weights"].contains("data"))
      weightData = layer["weights"]["data"].get<std::vector<float>>();
    if (layer.contains("biases") && layer["biases"].contains("data"))
      biasData = layer["biases"]["data"].get<std::vector<float>>();

    // Retrieve Dense layer configuration.
    int units = layer["config"]["units"].get<int>();
    int inputFeatures = inputShape.back();
    // Define the weight and bias tensor shapes.
    std::vector<int64_t> weightShape = {inputFeatures, units};
    std::vector<int64_t> biasShape = {units};

    auto weightType = RankedTensorType::get(weightShape, builder.getF32Type());
    auto biasType = RankedTensorType::get(biasShape, builder.getF32Type());

    auto weightAttr = convertToDenseAttr(builder, weightType, weightData);
    auto biasAttr = convertToDenseAttr(builder, biasType, biasData);

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
  }

} // namespace vortex