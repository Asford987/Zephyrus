#ifndef VORTEX_HDF5_TO_TOSA_PASS_H
#define VORTEX_HDF5_TO_TOSA_PASS_H

#include "mlir/Pass/Pass.h"
#include <vector>
#include <string>

namespace vortex {
using namespace mlir;

/// Represents a single layer in a neural network model.
struct NeuralNetworkLayer {
    std::string name;
    std::string type;  // e.g., "Dense", "Conv2D"
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    std::vector<float> weights;
    std::vector<float> biases;
};

/// Reads an HDF5 file and constructs an AST-like representation of the model.
std::vector<NeuralNetworkLayer> parseHDF5(const std::string &filePath);

/// Creates the MLIR pass that injects `run_model` and lowers to TOSA.
std::unique_ptr<Pass> createHDF5ToTosaPass();

} // namespace vortex

#endif // VORTEX_HDF5_TO_TOSA_PASS_H
