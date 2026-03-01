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

#include <cstdint>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_VALIDATENOBACKWARDDEPSPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

bool IsAllForward(func::FuncOp func) {
  bool is_all_forward = true;
  func.walk([&](FragmentCallOp fragment_call) {
    for (Attribute attr : fragment_call.getOrigin().getValue()) {
      if (cast<UserOriginAttr>(attr).getTransposeCount() != 0) {
        is_all_forward = false;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return is_all_forward;
}

FragmentCallOp FindProducerFragmentCall(Value operand) {
  if (mlir::isa<BlockArgument>(operand)) {
    return nullptr;
  }
  Operation* op = operand.getDefiningOp();
  if (!op) {
    return nullptr;
  }
  if (auto fragment_call = mlir::dyn_cast<FragmentCallOp>(op)) {
    return fragment_call;
  }
  if (auto transfer = mlir::dyn_cast<TransferOp>(op)) {
    return FindProducerFragmentCall(transfer.getOperand());
  }
  return nullptr;
}

llvm::DenseMap<StringRef, int64_t> BuildMeshIndexMap(func::FuncOp func) {
  llvm::DenseMap<StringRef, int64_t> mesh_index_map;
  ArrayRef<NamedMeshAttr> meshes = GetTopologyMeshes(func);
  for (auto [index, mesh] : llvm::enumerate(meshes)) {
    mesh_index_map[mesh.getName()] = index;
  }
  return mesh_index_map;
}

class ValidateNoBackwardDepsPass
    : public impl::ValidateNoBackwardDepsPassBase<ValidateNoBackwardDepsPass> {
  using ValidateNoBackwardDepsPassBase::ValidateNoBackwardDepsPassBase;

 protected:
  void runOnFunc(func::FuncOp func) override {
    if (!IsMpmdFunction(func)) {
      return;
    }
    if (!IsAllForward(func)) {
      if (failOnBackwardDeps) {
        func.emitError()
            << "Expected forward-only program but found non-forward fragments.";
        signalPassFailure();
      }
      return;
    }

    llvm::DenseMap<StringRef, int64_t> mesh_index_map = BuildMeshIndexMap(func);

    func.walk([&](FragmentCallOp consumer) {
      auto consumer_it = mesh_index_map.find(consumer.getMeshName());
      if (consumer_it == mesh_index_map.end()) {
        return;
      }
      int64_t consumer_mesh_index = consumer_it->second;

      for (Value operand : consumer.getArgOperands()) {
        FragmentCallOp producer = FindProducerFragmentCall(operand);
        if (!producer) {
          continue;
        }

        auto producer_it = mesh_index_map.find(producer.getMeshName());
        if (producer_it == mesh_index_map.end()) {
          continue;
        }
        int64_t producer_mesh_index = producer_it->second;

        if (producer_mesh_index > consumer_mesh_index) {
          auto diag = failOnBackwardDeps ? consumer.emitError()
                                         : consumer.emitWarning();
          diag << "Detected backward dependency but expected forward-only "
                  "pipeline since there are no transpose fragments: "
               << "fragment call on mesh \"" << producer.getMeshName()
               << "\" (mesh index " << producer_mesh_index
               << ") produces a value consumed by fragment call on mesh \""
               << consumer.getMeshName() << "\" (mesh index "
               << consumer_mesh_index
               << "). In a forward-only pipeline, dependencies must go from "
               << "lower mesh indices to higher mesh indices.";
          if (failOnBackwardDeps) {
            signalPassFailure();
          }
        }
      }
    });
  }
};

}  // namespace

}  // namespace mlir::mpmd
