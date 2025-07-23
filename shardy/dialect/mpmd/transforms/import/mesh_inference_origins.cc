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

#include "shardy/dialect/mpmd/transforms/import/mesh_inference_origins.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"

namespace mlir::mpmd {

StringRef TerminalNodesOrigin(Operation* op) {
  if (auto mpmd_op = dyn_cast<mpmd::BroadcastOp>(op)) {
    return kMpmdBroadcastOrigin;
  }
  if (auto mpmd_op = dyn_cast<mpmd::ReduceOp>(op)) {
    return kMpmdReduceOrigin;
  }

  SDY_CHECK(false) << "Unexpected MPMD op: "
                   << op->getName().getStringRef().str();
}

mpmd::MeshWithOriginsAttr UnusedCalleeInputMeshWithOrigin(MLIRContext* context,
                                                          StringRef mesh_name) {
  return mpmd::MeshWithOriginsAttr::get(
      context, mesh_name,
      mpmd::OriginAttr::get(context, kInferredUnusedCalleeInOrigin));
}

mpmd::MeshWithOriginsAttr UnusedCalleeOutputMeshWithOrigin(
    MLIRContext* context, StringRef mesh_name) {
  return mpmd::MeshWithOriginsAttr::get(
      context, mesh_name,
      mpmd::OriginAttr::get(context, kInferredUnusedCalleeOutOrigin));
}

mpmd::MeshWithOriginsAttr TransferMeshWithOrigin(mpmd::TransferOp transfer_op) {
  return mpmd::MeshWithOriginsAttr::get(
      transfer_op->getContext(), transfer_op.getType().getMeshName(),
      mpmd::OriginAttr::get(transfer_op->getContext(), kTransferOrigin));
}

}  // namespace mlir::mpmd
