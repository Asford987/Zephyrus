#pragma once
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <nlohmann/json.hpp>
#include <vector>
#include "Passes/handlers/DenseAttr.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace zephyrus {
  class LayerHandler {
    protected:
      std::vector<float> weightData;
      std::vector<int64_t> weightDim;
      std::vector<float> biasData;
      std::vector<int64_t> biasDim;
      RankedTensorType weightType;
      RankedTensorType biasType;
      int units;
      DenseElementsAttr weightAttr;
      DenseElementsAttr biasAttr;
      std::string layerName;

    public:
    void setup(const json& layer, OpBuilder& builder) {
            // Extract weight and bias data.
      layerName = layer["config"]["name"].get<std::string>();
      if (layer.contains("weights") && !layer["weights"].is_null()){
        weightData = layer["weights"]["sequential"][layerName]["kernel"]["data"].get<std::vector<float>>();
        weightDim = layer["weights"]["sequential"][layerName]["kernel"]["dims"].get<std::vector<int64_t>>();
        biasData = layer["weights"]["sequential"][layerName]["bias"].get<std::vector<float>>();
        biasDim = layer["weights"]["sequential"][layerName]["bias"]["dims"].get<std::vector<int64_t>>();
        units = layer["config"]["units"].get<int>();
        weightType = RankedTensorType::get(weightDim, builder.getF32Type());
        biasType = RankedTensorType::get(biasDim, builder.getF32Type());
        weightAttr = convertToDenseAttr(builder, weightType, weightData);
        biasAttr = convertToDenseAttr(builder, biasType, biasData);
      }
    }

    virtual void handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t>& inputShape, mlir::Value& lastOutput) = 0;
  };

} // namespace zephyrus