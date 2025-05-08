#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <llvm/ADT/APFloat.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <nlohmann/json.hpp>
#include <vector>
#include "Passes/handlers/Conv2DHandler.h"
#include "Passes/handlers/DenseAttr.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::FuncOp;

namespace zephyrus{
  void Conv2DHandler::handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t>& inputShape, mlir::Value& lastOutput) {
    setup(layer, builder);

    Location loc = funcOp.getLoc();
    auto f32 = builder.getF32Type();
  
    auto config = layer["config"];
    std::vector<int64_t> strides = config["strides"].get<std::vector<int64_t>>();
    std::string paddingType = config["padding"].get<std::string>();
  
    int64_t sh = strides[0];
    int64_t sw = strides[1];
    int64_t dh = 1, dw = 1;
  
    int64_t batch = inputShape[0];
    int64_t inH = inputShape[1];
    int64_t inW = inputShape[2];
    int64_t inC = inputShape[3];
  
    int64_t kh = weightDim[0];
    int64_t kw = weightDim[1];
    int64_t outC = weightDim[3];
  
    int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
    if (paddingType == "same") {
      int64_t outH = (inH + sh - 1) / sh;
      int64_t outW = (inW + sw - 1) / sw;
      int64_t padH = std::max((outH - 1) * sh + kh - inH, int64_t(0));
      int64_t padW = std::max((outW - 1) * sw + kw - inW, int64_t(0));
      padTop = padH / 2;
      padBottom = padH - padTop;
      padLeft = padW / 2;
      padRight = padW - padLeft;
    }
  
    int64_t outH = (inH + padTop + padBottom - kh) / sh + 1;
    int64_t outW = (inW + padLeft + padRight - kw) / sw + 1;
    std::vector<int64_t> outputShape = {batch, outH, outW, outC};
    auto outputType = RankedTensorType::get(outputShape, f32);
  
    Value weightTensor = builder.create<tosa::ConstOp>(loc, weightType, weightAttr);
    Value biasTensor = builder.create<tosa::ConstOp>(loc, biasType, biasAttr);
    
    auto padAttr      = builder.getI64ArrayAttr({padTop, padBottom,
                                                 padLeft, padRight});
    auto strideAttr   = builder.getI64ArrayAttr({sh, sw});
    auto dilationAttr = builder.getI64ArrayAttr({dh, dw});
    lastOutput = builder.create<tosa::Conv2DOp>(
        loc,
        outputType,
        lastOutput, weightTensor, biasTensor,
        padAttr, strideAttr, dilationAttr);
      
  
    inputShape = outputShape;
  }

} // namespace zephyrus