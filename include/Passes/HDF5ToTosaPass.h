#ifndef ZEPHYRUS_HDF5_TO_TOSA_PASS_H
#define ZEPHYRUS_HDF5_TO_TOSA_PASS_H

#include "mlir/Pass/Pass.h"
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "hdf5/serial/H5Cpp.h"
#include "hdf5/serial/H5Opublic.h"


namespace zephyrus {
    using namespace mlir;
    using json = nlohmann::json;
    json readModelConfig(const std::string &filePath);
    json readGroup(H5::Group &group);
    json readModelWeights(const std::string &filePath);
    std::vector<json> parseLayers(const json &modelConfig) ;
    void assignWeightsToLayers(std::vector<json> &layers, const json &weightsJson);
    struct HDF5ToTosaPass;

    std::unique_ptr<Pass> createHDF5ToTosaPass(const std::string &modelFile);

} // namespace zephyrus

#endif // ZEPHYRUS_HDF5_TO_TOSA_PASS_H
