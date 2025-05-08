#pragma once
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <llvm/ADT/APFloat.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <nlohmann/json.hpp>
#include <vector>
#include "Passes/handlers/LayerHandler.h"

using json = nlohmann::json;
using namespace mlir;
using mlir::FuncOp;

namespace zephyrus{
  class MaxPool2DHandler : public LayerHandler {
    public:
    void handleLayer(OpBuilder& builder, FuncOp& funcOp, const json& layer, std::vector<int64_t>& inputShape, mlir::Value& lastOutput);
  };


} // namespace zephyrus