#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <nlohmann/json.hpp>
#include <vector>

using json = nlohmann::json;
using namespace mlir;
using mlir::func::FuncOp;

namespace vortex {
  class LayerHandler {
    public:
      virtual void handleLayer(Builder& builder, FuncOp& funcOp, json& layer) = 0;
  };

} // namespace vortex