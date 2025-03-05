#include "include/Passes/HDF5ToTosaPass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "H5Cpp.h"

using namespace mlir;
namespace vortex {

// Function to read tensor shape from an HDF5 file
std::vector<int64_t> getTensorShape(const std::string &filePath, const std::string &datasetName) {
    try {
        H5::H5File file(filePath, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(datasetName);
        H5::DataSpace dataspace = dataset.getSpace();

        int rank = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        dataspace.getSimpleExtentDims(dims.data(), nullptr);

        return std::vector<int64_t>(dims.begin(), dims.end());
    } catch (H5::Exception &e) {
        llvm::errs() << "Error reading HDF5 file: " << e.getDetailMsg() << "\n";
        return {};
    }
}

// MLIR Pass Definition
namespace {
struct InjectRunModelPass : public PassWrapper<InjectRunModelPass, OperationPass<ModuleOp>> {
    std::string modelFile;

    explicit InjectRunModelPass(const std::string &file) : modelFile(file) {}

    StringRef getArgument() const final { return "inject-run-model"; }
    StringRef getDescription() const final { return "Injects a callable run_model function dynamically."; }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        OpBuilder builder(module.getContext());

        module.getContext()->getOrLoadDialect<mlir::func::FuncDialect>();
        
        // Read HDF5 to determine input/output shapes dynamically
        std::vector<int64_t> inputShape = getTensorShape(modelFile, "input");
        std::vector<int64_t> outputShape = getTensorShape(modelFile, "output");

        if (inputShape.empty() || outputShape.empty()) {
            module.emitError("Failed to read input/output shapes from HDF5.");
            return;
        }

        // Define MLIR tensor types
        auto tensorInputType = RankedTensorType::get(inputShape, builder.getF32Type());
        auto tensorOutputType = RankedTensorType::get(outputShape, builder.getF32Type());
        auto funcType = builder.getFunctionType({tensorInputType}, {tensorOutputType});

        // Inject function into MLIR
        builder.setInsertionPointToEnd(module.getBody());
        auto funcOp = builder.create<func::FuncOp>(module.getLoc(), "run_model", funcType);
        funcOp.addEntryBlock();
    }
};
} // namespace

// Pass Registration
std::unique_ptr<Pass> createInjectRunModelPass(const std::string &modelFile) {
    return std::make_unique<InjectRunModelPass>(modelFile);
}

} // namespace vortex
