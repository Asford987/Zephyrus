#include "Passes/HDF5ToTosaPass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "hdf5/serial/H5Cpp.h"
#include "hdf5/serial/H5Opublic.h"
#include "llvm/ADT/APFloat.h"
#include <iostream>
#include <nlohmann/json.hpp>
#include "Passes/handlers/ReadModel.h"
#include "Passes/handlers/DenseHandler.h"
#include "Passes/handlers/Conv2DHandler.h"
#include "Passes/handlers/SoftmaxHandler.h"
#include "Passes/handlers/MaxPool2DHandler.h"
#include "Passes/handlers/FlattenHandler.h"
#include "Passes/handlers/ReLUHandler.h"

using namespace mlir;

namespace zephyrus {
struct HDF5ToTosaPass : public PassWrapper<HDF5ToTosaPass, OperationPass<ModuleOp>> {
    std::string modelFile;

    explicit HDF5ToTosaPass(const std::string &file) : modelFile(file) {}

    void runOnOperation() override {
      ModuleOp module = getOperation();
      OpBuilder builder(module.getContext());
    
      // Load required dialects.
      module.getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
      module.getContext()->getOrLoadDialect<mlir::tosa::TosaDialect>();
    
      if (modelFile.empty()) {
        module.emitError("No model file provided. Use -model-file=<path>");
        return;
      }
    
      json modelConfig = readModelConfig(modelFile);
      if (modelConfig.empty()) {
        module.emitError("Failed to read model configuration from HDF5 file.");
        return;
      }
    
      std::vector<int64_t> inputShape;
      if (modelConfig.contains("config") && modelConfig["config"].contains("build_input_shape")) {
        for (auto dim : modelConfig["config"]["build_input_shape"]) {
          inputShape.push_back(dim.is_null() ? -1 : dim.get<int64_t>());
        }
      } else {
        module.emitError("Invalid model_config JSON: missing build_input_shape");
        return;
      }
    
      std::vector<json> layers = parseLayers(modelConfig);
      if (layers.empty()) {
        module.emitError("Failed to parse HDF5 model.");
        return;
      }
    
      json weightsJson = readModelWeights(modelFile);
      assignWeightsToLayers(layers, weightsJson);
    
      std::vector<int64_t> outputShape = inputShape;
      for (auto &layer : layers) {
        if (layer.contains("class_name") && layer["class_name"].get<std::string>() == "Dense") {
          int units = layer["config"]["units"].get<int>();
          outputShape.back() = units;
          layer["output_shape"] = outputShape;
          break;
        }
      }

      auto tensorInputType = RankedTensorType::get(inputShape, builder.getF32Type());
      auto tensorOutputType = RankedTensorType::get(outputShape, builder.getF32Type());
      auto funcType = builder.getFunctionType({tensorInputType}, {tensorOutputType});
    
      builder.setInsertionPointToEnd(module.getBody());
      auto funcOp = builder.create<mlir::func::FuncOp>(module.getLoc(), "forward", funcType);
      funcOp.addEntryBlock();
      Block &entryBlock = funcOp.getBody().front();
      builder.setInsertionPointToStart(&entryBlock);
    
      Value inputTensor = funcOp.getArgument(0);
      Value lastOutput = inputTensor;
    
      for (const auto &layer : layers) {
        if (layer.contains("class_name") && layer["class_name"].get<std::string>() == "Dense") {
          DenseHandler denseHandler;
          denseHandler.handleLayer(builder, funcOp, layer, inputShape, lastOutput);
        } else if (layer.contains("class_name") && layer["class_name"].get<std::string>() == "Conv2D") {
          Conv2DHandler conv2DHandler;
          conv2DHandler.handleLayer(builder, funcOp, layer, inputShape, lastOutput);
        } else if (layer.contains("class_name") && layer["class_name"].get<std::string>() == "Flatten") {
          FlattenHandler flattenHandler;
          flattenHandler.handleLayer(builder, funcOp, layer, inputShape, lastOutput);
        } else if (layer.contains("class_name") && layer["class_name"].get<std::string>() == "MaxPool2D") {
          MaxPool2DHandler maxPool2DHandler;
          maxPool2DHandler.handleLayer(builder, funcOp, layer, inputShape, lastOutput);
        } else if (layer.contains("class_name") && layer["class_name"].get<std::string>() == "Softmax") {
          SoftmaxHandler softmaxHandler;
          softmaxHandler.handleLayer(builder, funcOp, layer, inputShape, lastOutput);
        } else if (layer.contains("class_name") && layer["class_name"].get<std::string>() == "ReLU") {
          ReLUHandler reluHandler;
          reluHandler.handleLayer(builder, funcOp, layer, inputShape, lastOutput);
        } else {
          module.emitError("Unsupported layer type: " + layer["class_name"].get<std::string>());
          return;
        }
        //   // Extract weight and bias data.
        //   std::vector<float> weightData, biasData;
        //   if (layer.contains("weights") && layer["weights"].contains("data"))
        //     weightData = layer["weights"]["data"].get<std::vector<float>>();
        //   if (layer.contains("biases") && layer["biases"].contains("data"))
        //     biasData = layer["biases"]["data"].get<std::vector<float>>();
    
        //   // Retrieve Dense layer configuration.
        //   int units = layer["config"]["units"].get<int>();
        //   int inputFeatures = inputShape.back();
        //   // Define the weight and bias tensor shapes.
        //   std::vector<int64_t> weightShape = {inputFeatures, units};
        //   std::vector<int64_t> biasShape = {units};
    
        //   auto weightType = RankedTensorType::get(weightShape, builder.getF32Type());
        //   auto biasType = RankedTensorType::get(biasShape, builder.getF32Type());
    
        //   auto weightAttr = convertToDenseAttr(builder, weightType, weightData);
        //   auto biasAttr = convertToDenseAttr(builder, biasType, biasData);
    
        //   // Create constant ops for weights and biases.
        //   Value weightTensor = builder.create<tosa::ConstOp>(funcOp.getLoc(), weightType, weightAttr);
        //   Value biasTensor = builder.create<tosa::ConstOp>(funcOp.getLoc(), biasType, biasAttr);
    
        //   // Create TOSA MatMul and Add ops.
        //   // For this example, assume the output of matmul replaces the last dimension with "units".
        //   std::vector<int64_t> matmulOutputShape = inputShape;
        //   matmulOutputShape.back() = units;
        //   auto matmulOutputType = RankedTensorType::get(matmulOutputShape, builder.getF32Type());
    
        //   Value matmulOutput = builder.create<tosa::MatMulOp>(
        //       funcOp.getLoc(), matmulOutputType, lastOutput, weightTensor);
        //   lastOutput = builder.create<tosa::AddOp>(
        //       funcOp.getLoc(), matmulOutputType, matmulOutput, biasTensor);
        // }
        // // (Additional layer types such as Activation layers can be handled here.)
      }
    
      builder.create<mlir::func::ReturnOp>(funcOp.getLoc(), lastOutput);
    }
      
};

std::unique_ptr<Pass> createHDF5ToTosaPass(const std::string &modelFile) {
    return std::make_unique<HDF5ToTosaPass>(modelFile);
}

} // namespace zephyrus
