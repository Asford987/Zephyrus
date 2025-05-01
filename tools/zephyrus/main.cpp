#include "Passes/Passes.h"
#include "zephyrus/Zephyrus.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> inputModel(cl::Positional,
    cl::desc("<model.h5>"), cl::Required);

static cl::opt<std::string> outputFile("o",
    cl::desc("Output file (ELF object or LLVM IR)"),
    cl::init("model.o"));

static cl::opt<bool> emitLLVM("emit-llvm",
    cl::desc("Emit textual LLVM IR instead of object code"), cl::init(false));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Zephyrus – NN AOT compiler\n");

  zephyrus::CompileOptions opts;
  opts.emitLLVMIR = emitLLVM;
  opts.emitObject = !emitLLVM;

  std::string buffer;
  if (zephyrus::compileModel(inputModel, buffer, opts)) {
    errs() << "zephyrus: compilation failed\n";
    return 1;
  }

  // Write buffer to disk
  std::string error;
  auto outfile = mlir::openOutputFile(outputFile, &error);
  if (!outfile) {
    errs() << error << '\n';
    return 1;
  }
  outfile->os().write(buffer.data(), buffer.size());
  outfile->keep();

  outs() << "Wrote " << buffer.size() << " bytes → " << outputFile << '\n';
  return 0;
}
