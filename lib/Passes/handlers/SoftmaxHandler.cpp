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

namespace vortex{
  void SoftmaxHandler::handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t>& inputShape, mlir::Value& lastOutput){
      Location loc = funcOp.getLoc();
      auto f32 = builder.getF32Type();
      auto inputType = RankedTensorType::get(inputShape, f32);
      int axis = inputShape.size() - 1;
  
      // 1. reduce_max(x)
      std::vector<int64_t> reducedShape = inputShape;
      reducedShape[axis] = 1;
      auto reducedType = RankedTensorType::get(reducedShape, f32);
  
      Value maxVals = builder.create<tosa::ReduceMaxOp>(
          loc, reducedType, lastOutput, builder.getI64IntegerAttr(axis));
  
      // 2. reshape max to broadcast shape
      Value maxValsBroadcast = builder.create<tosa::ReshapeOp>(
          loc, inputType, maxVals, builder.getI64ArrayAttr(reducedShape));
  
      // 3. subtract max: x - max(x)
      Value centered = builder.create<tosa::SubOp>(loc, inputType, lastOutput, maxValsBroadcast);
  
      // 4. exp(x - max)
      Value expTensor = builder.create<tosa::ExpOp>(loc, inputType, centered);
  
      // 5. reduce_sum(exp(...))
      Value sumExp = builder.create<tosa::ReduceSumOp>(
          loc, reducedType, expTensor, builder.getI64IntegerAttr(axis));
  
      // 6. broadcast sum back
      Value sumExpBroadcast = builder.create<tosa::ReshapeOp>(
          loc, inputType, sumExp, builder.getI64ArrayAttr(reducedShape));
  
      // 7. divide exp / sum
      Value reciprocal = builder.create<tosa::ReciprocalOp>(loc, inputType, sumExpBroadcast);
      Value softmaxOut = builder.create<tosa::MulOp>(loc, inputType, expTensor, reciprocal);
  
      // Final output
      lastOutput = softmaxOut;
  }


} // namespace vortex