#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <llvm/ADT/APFloat.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <nlohmann/json.hpp>
#include <vector>
#include "Passes/handlers/MaxPool2DHandler.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace zephyrus{
  void MaxPool2DHandler::handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t>& inputShape, mlir::Value& lastOutput){
    Location loc = funcOp.getLoc();
    auto f32 = builder.getF32Type();
  
    auto config = layer["config"];
    std::vector<int64_t> poolSize = config["pool_size"].get<std::vector<int64_t>>();
    std::vector<int64_t> strides = config["strides"].get<std::vector<int64_t>>();
    std::string paddingType = config["padding"].get<std::string>(); // "valid" or "same"
  
    int64_t batch = inputShape[0];
    int64_t inH = inputShape[1];
    int64_t inW = inputShape[2];
    int64_t channels = inputShape[3];
  
    int64_t kh = poolSize[0];
    int64_t kw = poolSize[1];
    int64_t sh = strides[0];
    int64_t sw = strides[1];
  
    int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
    if (paddingType == "same") {
      auto outH = (inH + sh - 1) / sh;
      auto outW = (inW + sw - 1) / sw;
      int64_t totalPadH = std::max((outH - 1) * sh + kh - inH, int64_t(0));
      int64_t totalPadW = std::max((outW - 1) * sw + kw - inW, int64_t(0));
      padTop = totalPadH / 2;
      padBottom = totalPadH - padTop;
      padLeft = totalPadW / 2;
      padRight = totalPadW - padLeft;
    }
  
    int64_t outH = (inH + padTop + padBottom - kh) / sh + 1;
    int64_t outW = (inW + padLeft + padRight - kw) / sw + 1;
    std::vector<int64_t> outputShape = {batch, outH, outW, channels};
  
    auto outputType = RankedTensorType::get(outputShape, f32);

    auto kernelAttr = builder.getDenseI64ArrayAttr({kh, kw});
    auto strideAttr = builder.getDenseI64ArrayAttr({sh, sw});
    auto padAttr    = builder.getDenseI64ArrayAttr({padTop, padBottom,
                                              padLeft, padRight});

    lastOutput = builder.create<tosa::MaxPool2dOp>(
        loc,
        outputType,
        lastOutput,
        kernelAttr,
        strideAttr,
        padAttr);

    inputShape = outputShape;  
  }

} // namespace zephyrus