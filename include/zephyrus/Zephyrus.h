#pragma once
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"

namespace zephyrus {

struct CompileOptions {
  bool emitLLVMIR = false;   // textual .ll
  bool emitObject = true;    // default phase-1 output
};

/// Compiles `hdf5Path` into either LLVM-IR or an ELF object, writing the
/// bytes into `outputBuffer`.  Returns `false` on success.
bool compileModel(llvm::StringRef hdf5Path,
                  std::string &outputBuffer,
                  const CompileOptions &opts = {});

} // namespace zephyrus
