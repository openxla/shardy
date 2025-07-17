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

#include <memory>
#include <optional>
#include <string>

#include "llvm/Support/FormatVariadic.h"
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
  explicit SaveModuleOpPass(StringRef dumpDirectory, StringRef fileName,
                            std::optional<int> dumpIndex) {
    this->dumpDirectory = dumpDirectory.str();
    this->fileName = fileName.str();
    this->dumpIndex = dumpIndex;
  }

  SaveModuleOpPass(const SaveModuleOpPass& other) : PassWrapper(other) {}

 private:
  void runOnOperation() final {
    saveModuleOp(getOperation(), dumpDirectory, fileName, dumpIndex);
  }

  StringRef getArgument() const override { return "sdy-save-module"; }

  StringRef getDescription() const override {
    return "Saves the module to the specified directory with the specified "
           "name, saving it as a `.mlir` file.";
  }

  // Where to save the module
  std::string dumpDirectory;
  // The name of the file without the `.mlir` extension
  std::string fileName;
  // The index to be included as a prefix in the file name. If set to 0, will be
  // ignored.
  std::optional<int> dumpIndex;
};

}  // namespace

void saveModuleOp(ModuleOp moduleOp, StringRef dumpDirectory,
                  StringRef fileName, std::optional<int> dumpIndex) {
  if (!dumpIndex) {
    return saveModuleOpInternal(moduleOp, dumpDirectory, fileName);
  }
  return saveModuleOpInternal(
      moduleOp, dumpDirectory,
      llvm::formatv("{0:02}.{1}", *dumpIndex, fileName).str());
}

std::unique_ptr<Pass> createSaveModuleOpPass(StringRef dumpDirectory,
                                             StringRef fileName,
                                             std::optional<int> dumpIndex) {
  return std::make_unique<SaveModuleOpPass>(dumpDirectory, fileName, dumpIndex);
}

}  // namespace sdy
}  // namespace mlir
