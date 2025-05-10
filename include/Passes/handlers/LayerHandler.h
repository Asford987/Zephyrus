#pragma once
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <nlohmann/json.hpp>
#include <vector>
#include "Passes/handlers/DenseAttr.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::FuncOp;

namespace zephyrus {
  inline mlir::DenseIntElementsAttr makeI64DenseAttr(mlir::OpBuilder &b, llvm::ArrayRef<int64_t> vals) {
    auto vecTy = mlir::RankedTensorType::get(
        {static_cast<int64_t>(vals.size())}, b.getI64Type());
    return mlir::DenseIntElementsAttr::get(vecTy, vals);
  }
  inline mlir::ArrayAttr makeI64ArrayAttr(mlir::OpBuilder &b, llvm::ArrayRef<int64_t> vals) {
    llvm::SmallVector<mlir::Attribute> eltAttrs;
    eltAttrs.reserve(vals.size());
    for (int64_t v : vals)
        eltAttrs.push_back(b.getI64IntegerAttr(v));
    return b.getArrayAttr(eltAttrs);
  }

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
    void setup(const json &layer, OpBuilder &builder) {
      layerName = layer["config"]["name"].get<std::string>();
      if (!layer.contains("weights") || layer["weights"].is_null())
        return;                                   // layer is untrained
    
      const json &w =
          layer["weights"].contains("sequential") &&                 // usual Keras
          !layer["weights"]["sequential"].is_null() &&
          layer["weights"]["sequential"].contains(layerName)
              ? layer["weights"]["sequential"][layerName]
              : layer["weights"][layerName];                         // flat variant
    
      auto fetchTensor =
          [&](const json &obj, std::vector<float>  &data,
              std::vector<int64_t> &dims) -> bool {
            /*  schema v1  */ const char *dKey = "data",   *sKey = "dims";
            /*  schema v2  */ const char *vKey = "value",  *shKey = "shape";
    
            const json *d = obj.contains(dKey) ? &obj[dKey]
                         : obj.contains(vKey) ? &obj[vKey] : nullptr;
            const json *s = obj.contains(sKey) ? &obj[sKey]
                         : obj.contains(shKey)? &obj[shKey]: nullptr;
            if (!d || !s) return false;
    
            data = d->get<std::vector<float>>();
            dims = s->get<std::vector<int64_t>>();
            return true;
          };
    
      const json &kernelObj = w["kernel"];
      const json &biasObj   = w["bias"];
    
      if (!fetchTensor(kernelObj, weightData, weightDim) ||
          !fetchTensor(biasObj,   biasData,  biasDim)) {
        llvm::errs() << "zephyrus: malformed weights for layer \""
                     << layerName << "\"\n";
        return;
      }
    
      units      = layer["config"].value("units",
                    layer["config"].value("filters", 0));
    
      auto f32   = builder.getF32Type();
      weightType = RankedTensorType::get(weightDim, f32);
      biasType   = RankedTensorType::get(biasDim,   f32);
    
      weightAttr = convertToDenseAttr(builder, weightType, weightData);
      biasAttr   = convertToDenseAttr(builder, biasType,   biasData);
    }
    
    virtual void handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t>& inputShape, mlir::Value& lastOutput) = 0;
  };

} // namespace zephyrus