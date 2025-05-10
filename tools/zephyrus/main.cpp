#include "Passes/Passes.h"
#include "zephyrus/Zephyrus.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h" 
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Target/TargetMachine.h"

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
      
namespace zephyrus {
  bool compileModel(StringRef hdf5, std::string &out, const CompileOptions &opt) {
    DialectRegistry reg; registerAllDialects(reg);
    MLIRContext ctx(reg);  ctx.loadAllAvailableDialects();
    OwningOpRef<ModuleOp> mod(ModuleOp::create(UnknownLoc::get(&ctx)));
  
    mlir::PassManager pm(&ctx);
    buildHDF5ToLLVMPipeline(pm, hdf5);
    if (failed(pm.run(*mod))) return true;
  
    LLVMContext llvmCtx;
    auto llvmMod = translateModuleToLLVMIR(*mod, llvmCtx);
    if (!llvmMod) return true;
  
    if (opt.emitLLVMIR) {
      std::string tmp; raw_string_ostream os(tmp);
      llvmMod->print(os, nullptr);  out.swap(tmp);  return false;
    }
  
    InitializeNativeTarget(); InitializeNativeTargetAsmPrinter();
    std::string err;
    auto *T = TargetRegistry::lookupTarget(sys::getDefaultTargetTriple(), err);
    if (!T) { errs() << err << '\n'; return true; }
  
    auto TM = std::unique_ptr<TargetMachine>(
        T->createTargetMachine(sys::getDefaultTargetTriple(), "generic", "",
                               TargetOptions(), Reloc::PIC_, CodeModel::Small));
  
    SmallVector<char, 0> obj; raw_svector_ostream objOS(obj);
    legacy::PassManager cg;
    if (TM->addPassesToEmitFile(cg, objOS, nullptr, CGFT_ObjectFile)) return true;
    cg.run(*llvmMod);
  
    out.assign(obj.begin(), obj.end());
    return false;
  }
} // namespace zephyrus

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
