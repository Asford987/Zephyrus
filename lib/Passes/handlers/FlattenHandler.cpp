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
#include "Passes/handlers/FlattenHandler.h"
#include "Passes/handlers/DenseAttr.h"


using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace zephyrus{
  void FlattenHandler::handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t>& inputShape, mlir::Value& lastOutput){
    // setup(layer, builder);
    // Flatten all dimensions after batch
    Location loc = funcOp.getLoc();

    int64_t batchSize = inputShape[0];
    int64_t flatSize = 1;
    for (size_t i = 1; i < inputShape.size(); ++i) {
        flatSize *= inputShape[i];
    }

    std::vector<int64_t> newShape = {batchSize, flatSize};
    auto outputType = RankedTensorType::get(newShape, builder.getF32Type());
    
    auto shapeType  = RankedTensorType::get(
          {static_cast<int64_t>(newShape.size())}, builder.getI64Type());

    llvm::ArrayRef<int64_t> shapeRef(newShape);                 // ← no makeArrayRef
    DenseElementsAttr shapeAttr = DenseElementsAttr::get(shapeType, shapeRef);
    Value shapeTensor = builder.create<tosa::ConstOp>(loc, shapeType, shapeAttr);
    
    lastOutput = builder.create<tosa::ReshapeOp>(
      loc,
      outputType,
      lastOutput,
      shapeTensor
  );
      
    // lastOutput = builder.create<tosa::ReshapeOp>(
    //     funcOp.getLoc(), outputType, lastOutput,
    //     builder.getI64ArrayAttr(newShape));

    inputShape = newShape;    
  }

} // namespace zephyrus