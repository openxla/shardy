/* Copyright 2024 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "shardy/common/file_utils.h"

// NOLINTBEGIN: silence `is an unapproved C++11 header`.
#include <memory>
#include <string>
#include <system_error>
// NOLINTEND: silence `is an unapproved C++11 header`.

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "shardy/common/save_module_op.h"

namespace mlir {
namespace sdy {

namespace {

class SaveModuleOpPass
    : public PassWrapper<SaveModuleOpPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SaveModuleOpPass)

  // NOLINTNEXTLINE(clang-diagnostic-shadow-field)
  explicit SaveModuleOpPass(StringRef dumpDirectory, StringRef fileName) {
    this->dumpDirectory = dumpDirectory.str();
    this->fileName = fileName.str();
  }

  SaveModuleOpPass(const SaveModuleOpPass& other) : PassWrapper(other) {}

 private:
  void runOnOperation() final {
    saveModuleOp(getOperation(), dumpDirectory, fileName);
  }

  StringRef getArgument() const override { return "sdy-save-module"; }

  StringRef getDescription() const override {
    return "Saves the module to the specified directory with the specified "
           "name, saving it as a `.mlir` file.";
  }

  Option<std::string> dumpDirectory{*this, "module-dump-directory",
                                    llvm::cl::desc("where to save the module"),
                                    llvm::cl::init("")};

  Option<std::string> fileName{
      *this, "file-name",
      llvm::cl::desc("the name of the file without the `.mlir` extension")};
};

}  // namespace

std::unique_ptr<Pass> createSaveModuleOpPass(StringRef dumpDirectory,
                                             StringRef fileName) {
  return std::make_unique<SaveModuleOpPass>(dumpDirectory, fileName);
}

}  // namespace sdy
}  // namespace mlir
