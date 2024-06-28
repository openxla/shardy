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

#include "shardy/common/save_module_op.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {
namespace sdy {

namespace {
static void fileSavingError(StringRef filePath, StringRef message) {
  llvm::errs() << llvm::formatv("error when writing file {0}: {1}\n", filePath,
                                message);
}
}  // namespace

void saveModuleOp(ModuleOp moduleOp, StringRef dumpDirectory,
                  StringRef fileName) {
  if (dumpDirectory.empty()) {
    return;
  }
  SmallString<128> filePath(dumpDirectory);
  llvm::sys::path::append(filePath, fileName);
  filePath.append(".mlir");

  std::error_code errorCode;
  llvm::raw_fd_ostream fileStream(filePath, errorCode);
  if (errorCode) {
    fileSavingError(filePath.str(), errorCode.message());
    return;
  }
  moduleOp.print(fileStream);
  fileStream.close();
}

}  // namespace sdy
}  // namespace mlir
