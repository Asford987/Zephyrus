#include "Passes/Passes.h"
#include "zephyrus/Zephyrus.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Target/LLVMIR/Export.h" 
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
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
    DialectRegistry reg;
    registerAllDialects(reg);
    mlir::registerBuiltinDialectTranslation(reg);
    mlir::registerLLVMDialectTranslation(reg);

    MLIRContext ctx(reg);
    ctx.loadAllAvailableDialects();

    OwningOpRef<ModuleOp> mod(ModuleOp::create(UnknownLoc::get(&ctx)));

    mlir::PassManager pm(&ctx);
    buildHDF5ToLLVMPipeline(pm, hdf5);
    if (failed(pm.run(*mod))) return true;

    LLVMContext llvmCtx;
    std::unique_ptr<llvm::Module> llvmMod =
        mlir::translateModuleToLLVMIR(*mod, llvmCtx);
    if (!llvmMod) return true;

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    std::string triple = llvm::sys::getDefaultTargetTriple();
    std::string err;
    const llvm::Target *T = llvm::TargetRegistry::lookupTarget(triple, err);
    if (!T) {
      llvm::errs() << err << '\n';
      return true;
    }

    std::unique_ptr<llvm::TargetMachine> TM(
        T->createTargetMachine(triple, "generic", "", llvm::TargetOptions(),
                              llvm::Reloc::PIC_, llvm::CodeModel::Small));

    llvmMod->setTargetTriple(triple);
    llvmMod->setDataLayout(TM->createDataLayout());

    if (opt.emitLLVMIR) {
      std::string tmp;
      llvm::raw_string_ostream os(tmp);
      llvmMod->print(os, nullptr);
      out.swap(tmp);
      return false;
    }

    llvm::SmallVector<char, 0> obj;
    llvm::raw_svector_ostream objOS(obj);
    llvm::legacy::PassManager cg;
    if (TM->addPassesToEmitFile(cg, objOS, nullptr, llvm::CGFT_ObjectFile))
      return true;
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
