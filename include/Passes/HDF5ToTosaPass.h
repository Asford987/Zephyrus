#ifndef VORTEX_HDF5_TO_TOSA_PASS_H
#define VORTEX_HDF5_TO_TOSA_PASS_H

#include "mlir/Pass/Pass.h"
#include <vector>
#include <string>

namespace vortex {
using namespace mlir;

// Represents a single layer in the model.
struct NeuralNetworkLayer {
    std::string name;              // Layer name (e.g., "Dense_1")
    std::string type;              // Layer type (e.g., "Dense", "Conv2D", "Activation")
    std::vector<int64_t> inputShape;  // Input tensor shape
    std::vector<int64_t> outputShape; // Output tensor shape
    std::vector<float> weights;    // Weights (for Dense, Conv2D, etc.)
    std::vector<float> biases;     // Bias values
    std::string activation;        // Activation function (e.g., "relu", "sigmoid")
    bool isActivationLayer = false; // True if this is an activation-only layer
};

// Parses an HDF5 model file.
std::vector<NeuralNetworkLayer> parseHDF5(const std::string &filePath);

// Creates an MLIR pass that injects `run_model` and lowers to TOSA.
std::unique_ptr<Pass> createHDF5ToTosaPass(const std::string &modelFile);

} // namespace vortex

#endif // VORTEX_HDF5_TO_TOSA_PASS_H
