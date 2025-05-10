#include "Passes/HDF5ToTosaPass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "H5Cpp.h"
#include "llvm/ADT/APFloat.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include <H5Opublic.h>
#include "Passes/handlers/ReadModel.h"


using json = nlohmann::json;

namespace zephyrus {
  json readModelConfig(const std::string &filePath) {
  try {
      H5::H5File file(filePath, H5F_ACC_RDONLY);
      H5::Attribute modelConfigAttr = file.openAttribute("model_config");
      H5::StrType strType = modelConfigAttr.getStrType();
      std::string modelConfigStr;

      if (strType.isVariableStr()) {
          char* buffer = nullptr;
          modelConfigAttr.read(strType, &buffer);
          if (buffer) {
              modelConfigStr = std::string(buffer);
              free(buffer);
          }
      } else {
          size_t size = strType.getSize();
          std::vector<char> buffer(size + 1, '\0');
          modelConfigAttr.read(strType, buffer.data());
          modelConfigStr = std::string(buffer.data());
      }
      return json::parse(modelConfigStr);
  } catch (H5::Exception &e) {
      std::cerr << "Error reading model_config from HDF5: " << e.getDetailMsg() << std::endl;
      return json();
  }
  }

  json readGroup(H5::Group &group)
  {
    json out;
  
    const hsize_t nObjs = group.getNumObjs();
    for (hsize_t i = 0; i < nObjs; ++i) {
      std::string name = group.getObjnameByIdx(i);
  
      H5G_obj_t kind = H5G_UNKNOWN;
      try {
        kind = static_cast<H5G_obj_t>(group.getObjTypeByIdx(i));
      } catch (const H5::Exception &e) {
        std::cerr << "Skipping child '" << name
                  << "' (cannot determine type – " << e.getDetailMsg() << ")\n";
        continue;
      }
  
      if (kind == H5G_DATASET) {
        H5::DataSet ds     = group.openDataSet(name);
        H5::DataSpace space = ds.getSpace();
  
        int rank = space.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        space.getSimpleExtentDims(dims.data());
  
        size_t nElem = 1;
        for (auto d : dims) nElem *= d;
  
        std::vector<float> buffer(nElem);
        ds.read(buffer.data(), H5::PredType::NATIVE_FLOAT);
  
        out[name] = {
            {"dims", dims},
            {"data", buffer}
        };
      }
      else if (kind == H5G_GROUP) {
        H5::Group sub = group.openGroup(name);
        out[name] = readGroup(sub);
      }
  
      else {
        std::cerr << "Skipping object '" << name
                  << "' (unsupported HDF5 type " << kind << ")\n";
      }
    }
    return out;
  }

  json readModelWeights(const std::string &filePath) {
    json weightsJson;
    try {
      H5::H5File file(filePath, H5F_ACC_RDONLY);
      H5::Group weightsGroup = file.openGroup("model_weights");
      hsize_t nLayers = weightsGroup.getNumObjs();
      for (hsize_t i = 0; i < nLayers; i++) {
        std::string layerName = weightsGroup.getObjnameByIdx(i);
        H5::Group layerGroup = weightsGroup.openGroup(layerName);
        weightsJson[layerName] = readGroup(layerGroup);
      }
    } catch (H5::Exception &e) {
        std::cerr << "Error reading model_weights from HDF5: " 
                    << e.getDetailMsg() << std::endl;
    }
    return weightsJson;
  }

  std::vector<json> parseLayers(const json &modelConfig) {
  std::vector<json> layers;
  std::vector<int64_t> currentShape;

  if (!modelConfig.contains("config") || !modelConfig["config"].contains("layers")) {
      std::cerr << "Invalid model_config JSON structure!\n";
      return layers;
  }

  for (auto dim : modelConfig["config"]["build_input_shape"]) {
      currentShape.push_back(dim.is_null() ? -1 : dim.get<int64_t>());
  }

  for (const auto &layerJson : modelConfig["config"]["layers"]) {
      json newLayer = layerJson;

      bool isActivation = false;
      std::string activationType;

      if (layerJson.contains("class_name") && layerJson["class_name"].get<std::string>() == "Activation") {
          isActivation = true;
          activationType = layerJson["config"]["activation"].get<std::string>();
      } else if (layerJson["config"].contains("activation")) {
          activationType = layerJson["config"]["activation"].get<std::string>();
          if (!activationType.empty() && activationType != "linear") {
              isActivation = true;
          }
      }

      if (isActivation) {
          newLayer["isActivationLayer"] = true;
          newLayer["activationType"] = activationType;
      } else {
          newLayer["isActivationLayer"] = false;
      }

      layers.push_back(newLayer);

      if (!isActivation && layerJson["config"].contains("activation")) {
          std::string act = layerJson["config"]["activation"].get<std::string>();
          if (!act.empty() && act != "linear") {
              json activationLayer;
              activationLayer["name"] = newLayer["config"]["name"].get<std::string>() + "_activation";
              activationLayer["type"] = "Activation";
              activationLayer["activationType"] = act;
              activationLayer["isActivationLayer"] = true;

              layers.push_back(activationLayer);
          }
      }
  }
  return layers;
  }

  void assignWeightsToLayers(std::vector<json> &layers, const json &weightsJson) {
    for (auto &layer : layers) {
      std::string layerName;
      if (layer.contains("config") && layer["config"].contains("name")) {
        layerName = layer["config"]["name"].get<std::string>();
      } else if (layer.contains("name")) {
        layerName = layer["name"].get<std::string>();
      }
      if (!layerName.empty() && weightsJson.contains(layerName)) {
        layer["weights"] = weightsJson[layerName];
      }
    }
  }
} // namespace zephyrus