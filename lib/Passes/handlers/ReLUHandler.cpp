#include <vector>
#include <limits>

#include "Passes/handlers/ReLUHandler.h"

#include <nlohmann/json.hpp>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <llvm/ADT/APFloat.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>


using json = nlohmann::json;
using namespace mlir;
using mlir::FuncOp;

namespace zephyrus{
  void ReLUHandler::handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t>& inputShape, mlir::Value& lastOutput){
    Location loc = funcOp.getLoc();
    auto f32 = builder.getF32Type();
    auto inputType = RankedTensorType::get(inputShape, f32);

    auto zeroFP  = builder.getF32FloatAttr(0.0f);
    auto largeFP = builder.getF32FloatAttr(std::numeric_limits<float>::max());
    
    // integer limits – ignored for FP tensors but required by the op
    auto zeroI64 = builder.getI64IntegerAttr(0);
    
    lastOutput = builder.create<tosa::ClampOp>(loc, inputType, lastOutput, zeroI64, zeroI64, zeroFP, largeFP);

  }

} // namespace zephyrus