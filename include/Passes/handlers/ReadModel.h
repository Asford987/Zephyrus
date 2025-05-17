#include "H5Cpp.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <H5Opublic.h>

namespace zephyrus {
  using json = nlohmann::json;
  json readModelConfig(const std::string &filePath);
  json readGroup(H5::Group &group);
  json readModelWeights(const std::string &filePath);
  void assignWeightsToLayers(std::vector<json> &layers, const json &weightsJson);
} // namespace zephyrus