/* Copyright 2025 The MPMD Authors.

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

#include <string>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_VALIDATENOPARAMTRANSFERSPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

// Returns true if the location is a NameLoc and contains the given pattern.
bool LocationContainsPattern(Location loc, StringRef pattern) {
  if (auto name_loc = dyn_cast<NameLoc>(loc)) {
    return name_loc.getName().getValue().contains(pattern);
  }
  return false;
}

class ValidateNoParamTransfersPass
    : public impl::ValidateNoParamTransfersPassBase<
          ValidateNoParamTransfersPass> {
  using ValidateNoParamTransfersPassBase::ValidateNoParamTransfersPassBase;

 protected:
  void runOnFunc(func::FuncOp func) override {
    if (!IsMpmdFunction(func)) {
      return;
    }

    StringRef pattern(paramPattern);
    func.walk([&](TransferOp transfer) {
      if (transfer.isIntraMesh()) {
        return;
      }

      Location operand_loc = transfer.getTensor().getLoc();
      Location transfer_loc = transfer.getLoc();
      if (!LocationContainsPattern(operand_loc, pattern) &&
          !LocationContainsPattern(transfer_loc, pattern)) {
        return;
      }

      auto src_mesh = transfer.getTensor().getType().getMeshName();
      auto dst_mesh = transfer.getResult().getType().getMeshName();

      // Collect only the NameLoc names that match the pattern for a concise
      // error message. Use SetVector to deduplicate while preserving order.
      llvm::SetVector<StringRef> matching_names;
      if (auto name_loc = dyn_cast<NameLoc>(operand_loc)) {
        if (name_loc.getName().getValue().contains(pattern)) {
          matching_names.insert(name_loc.getName().getValue());
        }
      }
      if (auto name_loc = dyn_cast<NameLoc>(transfer_loc)) {
        if (name_loc.getName().getValue().contains(pattern)) {
          matching_names.insert(name_loc.getName().getValue());
        }
      }

      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "Detected cross-mesh transfer of a parameter matching \"" << pattern
         << "\" from mesh \"" << src_mesh << "\" to \"" << dst_mesh
         << "\". JAX locations: [";
      bool first = true;
      for (StringRef name : matching_names) {
        if (!first) os << ", ";
        os << "\"" << name << "\"";
        first = false;
      }
      os << "]";

      SDY_LOG(WARNING) << msg;
      auto diag = failOnParamTransfers ? transfer.emitError()
                                       : transfer.emitWarning();
      diag << msg;
      if (failOnParamTransfers) {
        signalPassFailure();
      }
    });
  }
};

}  // namespace

}  // namespace mlir::mpmd
