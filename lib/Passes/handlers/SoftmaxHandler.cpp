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
#include "Passes/handlers/SoftmaxHandler.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace zephyrus{
  void SoftmaxHandler::handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t>& inputShape, mlir::Value& lastOutput){
      Location loc = funcOp.getLoc();
      auto f32 = builder.getF32Type();
      auto inputType = RankedTensorType::get(inputShape, f32);
      int axis = inputShape.size() - 1;
  
      std::vector<int64_t> reducedShape = inputShape;
      reducedShape[axis] = 1;
      auto reducedType = RankedTensorType::get(reducedShape, f32);
      auto shapeType  = RankedTensorType::get(
          {static_cast<int64_t>(reducedShape.size())}, builder.getI64Type());
      DenseElementsAttr shapeAttr =
          DenseElementsAttr::get(shapeType,
                                 llvm::ArrayRef<int64_t>(reducedShape));
  
      Value maxVals = builder.create<tosa::ReduceMaxOp>(
          loc, reducedType, lastOutput, builder.getI64IntegerAttr(axis));
  
      Value shapeTensor =
          builder.create<tosa::ConstOp>(loc, shapeType, shapeAttr);
      Value maxValsBroadcast = builder.create<tosa::ReshapeOp>(
          loc, inputType, maxVals, shapeTensor);
  
      Value centered = builder.create<tosa::SubOp>(loc, inputType, lastOutput, maxValsBroadcast);
  
      Value expTensor = builder.create<tosa::ExpOp>(loc, inputType, centered);
  
      Value sumExp = builder.create<tosa::ReduceSumOp>(
          loc, reducedType, expTensor, builder.getI64IntegerAttr(axis));
  
      Value sumExpBroadcast = builder.create<tosa::ReshapeOp>(
          loc, inputType, sumExp, shapeTensor);
      
      auto shiftType  = RankedTensorType::get({}, builder.getIntegerType(32));
      auto zeroShift  = builder.create<tosa::ConstOp>(
          loc, shiftType,
          DenseElementsAttr::get(shiftType, 0));
      
      Value reciprocal = builder.create<tosa::ReciprocalOp>(loc, inputType, sumExpBroadcast);
      Value softmaxOut = builder.create<tosa::MulOp>(
          loc, inputType, expTensor, reciprocal, zeroShift);

      lastOutput = softmaxOut;
  }


} // namespace zephyrus