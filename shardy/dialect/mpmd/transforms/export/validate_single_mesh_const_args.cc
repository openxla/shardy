/* Copyright 2026 The MPMD Authors.

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

#include <cstdint>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_VALIDATESINGLEMESHCONSTARGSPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

// Validates that closed-over constants hoisted to entry-point function
// arguments (`{jax.const = true}`) are only used on a single mesh. Because
// constants are hoisted to function arguments and dead code is eliminated
// earlier in export, any cross-mesh usage of a constant argument is
// represented as an inter-mesh `mpmd.transfer` directly on that argument.
class ValidateSingleMeshConstArgsPass
    : public impl::ValidateSingleMeshConstArgsPassBase<
          ValidateSingleMeshConstArgsPass> {
  using ValidateSingleMeshConstArgsPassBase::
      ValidateSingleMeshConstArgsPassBase;

 protected:
  void runOnFunc(func::FuncOp func) override {
    if (!IsMpmdFunction(func) || !IsEntryPointFunction(func)) {
      return;
    }

    for (int64_t arg_num = 0; arg_num < func.getNumArguments(); ++arg_num) {
      auto is_const = func.getArgAttrOfType<BoolAttr>(arg_num, kJaxConstAttr);
      if (!is_const || !is_const.getValue()) {
        continue;
      }

      BlockArgument c_arg = func.getArgument(arg_num);
      if (c_arg.use_empty()) {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << "Closed-over constant (argument " << arg_num << ") is unused.";
        SDY_LOG(WARNING) << msg;
        func.emitWarning() << msg;
        continue;
      }

      SmallVector<TransferOp> transfer_worklist;
      for (Operation* user : c_arg.getUsers()) {
        if (auto transfer = dyn_cast<TransferOp>(user);
            transfer && transfer.isInterMesh()) {
          transfer_worklist.push_back(transfer);
        }
      }
      if (transfer_worklist.empty()) {
        continue;
      }

      llvm::SetVector<StringRef> meshes;
      if (auto mesh_tensor = dyn_cast<MeshTensorType>(c_arg.getType())) {
        meshes.insert(mesh_tensor.getMeshName());
      }
      while (!transfer_worklist.empty()) {
        TransferOp transfer = transfer_worklist.pop_back_val();
        meshes.insert(transfer.getTensor().getType().getMeshName());
        meshes.insert(transfer.getResult().getType().getMeshName());
        for (Operation* user : transfer.getResult().getUsers()) {
          if (auto chained = dyn_cast<TransferOp>(user);
              chained && chained.isInterMesh()) {
            transfer_worklist.push_back(chained);
          }
        }
      }

      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "Closed-over constant (argument " << arg_num
         << ") is used across multiple meshes: [";
      llvm::interleaveComma(meshes, os,
                            [&](StringRef mesh) { os << "'" << mesh << "'"; });
      os << "]. Constants cannot be shared across multiple meshes in MPMD. "
            "Please duplicate the constant in each mesh/stage where it is "
            "needed, or avoid closing over constants.";
      func.emitError() << msg;
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::mpmd
