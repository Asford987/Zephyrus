#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "Passes/HDF5ToTosaPass.h"
#include "Passes/TosaToLLVMPass.h"
#include "mlir/Support/ToolOutputFile.h" 
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

// === Command-line options ===
static cl::opt<std::string> inputFilename(cl::Positional,
    cl::desc("<HDF5 model file>"), cl::Required);

static cl::opt<std::string> outputFilename("o",
    cl::desc("Output file (LLVM IR or MLIR)"), cl::value_desc("filename"),
    cl::init("-"));

static cl::opt<bool> emitLLVM("emit-llvm",
    cl::desc("Emit LLVM IR instead of MLIR"), cl::init(false));

// === Main ===
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Zephyrus NN Compiler\n");

  MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();

  // Create an empty MLIR module
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));

  // Apply frontend pass (HDF5 → TOSA MLIR)
  PassManager pm(&context);
  pm.addPass(zephyrus::createHDF5ToTosaPass(inputFilename));

  if (failed(pm.run(module.get()))) {
    errs() << "Failed to convert HDF5 model to TOSA\n";
    return 1;
  }

  // Optionally lower to LLVM
  if (emitLLVM) {
    if (failed(zephyrus::lowerTosaToLLVM(*module))) {
      errs() << "Lowering to LLVM failed\n";
      return 1;
    }
  }

  // Write IR to file or stdout
  std::string errorMessage;
  std::unique_ptr<ToolOutputFile> output =
      mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    errs() << errorMessage << "\n";
    return 1;
  }

  module->print(output->os());
  output->keep();

  return 0;
}
