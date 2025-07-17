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

#ifndef SHARDY_COMMON_FILE_UTILS_H_
#define SHARDY_COMMON_FILE_UTILS_H_

#include <memory>
#include <optional>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace sdy {

// Saves the `moduleOp` to the given `dumpDirectory` with name `fileName`.
//
// NOTE:
// - if there is an existing file in `dumpDirectory` with the same name, it will
//   be overwritten.
// - if `dumpDirectory` is an empty string, nothing will be saved.
// - if `dumpDirectory` path doesn't exist yet, it will try to create it.
// - if `dumpIndex` is present, it will be included as a prefix in the filename.
// - any error will be logged to standard error.
// - do not include a file extension in `fileName`, `.mlir` will be appended
//   internally.
void saveModuleOp(ModuleOp moduleOp, StringRef dumpDirectory,
                  StringRef fileName,
                  std::optional<int> dumpIndex = std::nullopt);

// Saves the `moduleOp` to the given `dumpDirectory` with name `fileName` and
// `dumpIndex` (if present).
//
// NOTE: see `saveModuleOp` for details of the behavior.
std::unique_ptr<Pass> createSaveModuleOpPass(
    StringRef dumpDirectory, StringRef fileName,
    std::optional<int> dumpIndex = std::nullopt);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_COMMON_FILE_UTILS_H_
