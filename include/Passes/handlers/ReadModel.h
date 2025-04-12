#include "HDF5ToTosaPass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "H5Cpp.h"
#include "llvm/ADT/APFloat.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <H5Opublic.h>

namespace vortex {
  using json = nlohmann::json;
  json readModelConfig(const std::string &filePath);
  json readGroup(H5::Group &group);
  json readModelWeights(const std::string &filePath);
  std::vector<json> parseLayers(const json &modelConfig);
  void assignWeightsToLayers(std::vector<json> &layers, const json &weightsJson);
} // namespace vortex