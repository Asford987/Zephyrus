#include "HDF5ToTosaPass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "H5Cpp.h"
#include "llvm/ADT/APFloat.h"
#include <iostream>
#include <nlohmann/json.hpp> // Use JSON library for parsing

using json = nlohmann::json;  // Alias for convenience

using namespace mlir;
namespace vortex {

// Converts std::vector<float> to DenseElementsAttr
auto convertToDenseAttr(Builder &builder, RankedTensorType type, const std::vector<float> &values) {
    SmallVector<APFloat, 4> apValues;
    for (float v : values) {
        apValues.push_back(APFloat(v));
    }
    return DenseElementsAttr::get(type, apValues);
}

std::vector<NeuralNetworkLayer> parseHDF5(const std::string &filePath) {
    std::vector<NeuralNetworkLayer> layers;

    try {
        H5::H5File file(filePath, H5F_ACC_RDONLY);

        // Read model_config JSON string correctly
        H5::Attribute modelConfigAttr = file.openAttribute("model_config");
        H5::StrType strType = modelConfigAttr.getStrType();

        if (strType.isVariableStr()) {
            char* buffer = nullptr;
            modelConfigAttr.read(strType, &buffer);

            if (buffer) {
                std::string modelConfigStr(buffer);
                free(buffer);  // Free memory allocated by HDF5

                // Parse JSON
                json modelConfig = json::parse(modelConfigStr);

                // Extract input shape
                std::vector<int64_t> inputShape;
                for (auto dim : modelConfig["config"]["build_input_shape"]) {
                    inputShape.push_back(dim.is_null() ? -1 : dim.get<int64_t>());
                }

                // Extract layers
                for (auto &layer : modelConfig["config"]["layers"]) {
                    NeuralNetworkLayer nnLayer;
                    nnLayer.name = layer["config"]["name"];
                    nnLayer.type = layer["class_name"];
                    nnLayer.inputShape = inputShape;
                    nnLayer.outputShape = inputShape;

                    if (layer["config"].contains("activation")) {
                        nnLayer.activation = layer["config"]["activation"];
                    }
                    if (nnLayer.type == "Activation") {
                        nnLayer.isActivationLayer = true;
                    }
                    layers.push_back(nnLayer);
                }
            }
        } else {
            // Handle fixed-length string attributes
            size_t size = strType.getSize();
            std::vector<char> buffer(size + 1, '\0');  // Ensure null termination
            modelConfigAttr.read(strType, buffer.data());

            std::string modelConfigStr(buffer.data());
            json modelConfig = json::parse(modelConfigStr);

            // Extract input shape
            std::vector<int64_t> inputShape;
            for (auto dim : modelConfig["config"]["build_input_shape"]) {
                inputShape.push_back(dim.is_null() ? -1 : dim.get<int64_t>());
            }

            // Extract layers
            for (auto &layer : modelConfig["config"]["layers"]) {
                NeuralNetworkLayer nnLayer;
                nnLayer.name = layer["config"]["name"];
                nnLayer.type = layer["class_name"];
                nnLayer.inputShape = inputShape;
                nnLayer.outputShape = inputShape;

                if (layer["config"].contains("activation")) {
                    nnLayer.activation = layer["config"]["activation"];
                }
                if (nnLayer.type == "Activation") {
                    nnLayer.isActivationLayer = true;
                }
                layers.push_back(nnLayer);
            }
        }
    } catch (H5::Exception &e) {
        std::cerr << "Error reading HDF5: " << e.getDetailMsg() << std::endl;
    }

    return layers;
}


// MLIR Pass Definition
namespace {
struct HDF5ToTosaPass : public PassWrapper<HDF5ToTosaPass, OperationPass<ModuleOp>> {
    std::string modelFile;  // Store model filename

    explicit HDF5ToTosaPass(const std::string &file) : modelFile(file) {}

    void runOnOperation() override {
        ModuleOp module = getOperation();
        OpBuilder builder(module.getContext());

        module.getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
        module.getContext()->getOrLoadDialect<mlir::tosa::TosaDialect>();

        if (modelFile.empty()) {
            module.emitError("No model file provided. Use -model-file=<path>");
            return;
        }

        // Read HDF5 file and construct AST representation
        std::vector<NeuralNetworkLayer> layers = parseHDF5(modelFile);

        if (layers.empty()) {
            module.emitError("Failed to parse HDF5 model.");
            return;
        }

        // Get input/output tensor shapes
        auto inputShape = layers.front().inputShape;
        auto outputShape = layers.back().outputShape;

        auto tensorInputType = RankedTensorType::get(inputShape, builder.getF32Type());
        auto tensorOutputType = RankedTensorType::get(outputShape, builder.getF32Type());
        auto funcType = builder.getFunctionType({tensorInputType}, {tensorOutputType});

        // Inject run_model function
        builder.setInsertionPointToEnd(module.getBody());
        auto funcOp = builder.create<mlir::func::FuncOp>(module.getLoc(), "run_model", funcType);
        funcOp.addEntryBlock();

        // Insert TOSA operations
        Block &entryBlock = funcOp.getBody().front();
        builder.setInsertionPointToStart(&entryBlock);

        Value inputTensor = funcOp.getArgument(0);
        Value lastOutput = inputTensor;

        for (const auto &layer : layers) {
            if (layer.type == "Dense") {
                auto weightType = RankedTensorType::get(layer.outputShape, builder.getF32Type());
                auto biasType = RankedTensorType::get(layer.outputShape, builder.getF32Type());

                auto weightAttr = convertToDenseAttr(builder, weightType, layer.weights);
                auto biasAttr = convertToDenseAttr(builder, biasType, layer.biases);

                Value weightTensor = builder.create<tosa::ConstOp>(funcOp.getLoc(), weightType, weightAttr);
                Value biasTensor = builder.create<tosa::ConstOp>(funcOp.getLoc(), biasType, biasAttr);

                // Use MatMulOp instead of FullyConnectedOp
                Value matmulOutput = builder.create<tosa::MatMulOp>(
                    funcOp.getLoc(), tensorOutputType, lastOutput, weightTensor);

                lastOutput = builder.create<tosa::AddOp>(
                    funcOp.getLoc(), tensorOutputType, matmulOutput, biasTensor);
            }
        }

        // Return last output
        builder.create<mlir::func::ReturnOp>(funcOp.getLoc(), lastOutput);
    }
};
} // namespace

std::unique_ptr<Pass> createHDF5ToTosaPass(const std::string &modelFile) {
    return std::make_unique<HDF5ToTosaPass>(modelFile);
}

} // namespace vortex
