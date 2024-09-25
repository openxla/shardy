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

#include <cassert>
#include <functional>
#include <iterator>
#include <memory>  // IWYU pragma: keep
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_MANUALAXESCLEANUPPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

// Returns whether the new shardings needed to be updated to account for unused
// manual axes, and the new shardings.
std::optional<SmallVector<TensorShardingAttr>> addUnusedManualAxesToReplicatedAxes(
    ArrayRef<TensorShardingAttr> shardings, ArrayRef<StringAttr> manualAxes,
    std::function<bool(AxisRefAttr lhs, AxisRefAttr rhs)> meshComparator,
    StringAttr meshName) {
  SmallVector<TensorShardingAttr> newShardings;
  bool modified = false;
  for (const auto& sharding : shardings) {
    llvm::SmallSet<StringRef, 2> unusedManualAxes;
    unusedManualAxes.insert(manualAxes.begin(), manualAxes.end());
    sharding.forEachAxisRef([&unusedManualAxes](AxisRefAttr axis) {
      unusedManualAxes.erase(axis.getName());
    });

    if (unusedManualAxes.empty()) {
      // Already uses all the manual axes, no need to create a new one.
      newShardings.push_back(sharding);
      continue;
    }

    SmallVector<AxisRefAttr> newReplicatedAxes =
        llvm::to_vector(sharding.getReplicatedAxes());
    llvm::transform(unusedManualAxes, std::back_inserter(newReplicatedAxes),
                    [&](StringRef axis) {
                      return AxisRefAttr::get(sharding.getContext(), axis);
                    });
    llvm::sort(newReplicatedAxes, meshComparator);
    newShardings.push_back(
        TensorShardingAttr::get(sharding.getContext(), meshName,
                                sharding.getDimShardings(), newReplicatedAxes));
    modified = true;
  }
  return modified ? std::make_optional(newShardings) : std::nullopt;
}

// Adds any unused manual axes to the replicated_axes list for each in/out
// sharding
void addUnusedManualAxesToReplicatedAxes(ManualComputationOp op) {
  OpBuilder builder(op);
  std::optional<StringRef> meshName = getCommonMeshName(
      op.getInShardings().getShardings(), op.getOutShardings().getShardings());
  assert(meshName.has_value() &&
         "Expected no input/output ManualComputationOps to be canonicalized");
  MeshAttr mesh = getMeshAttr(op, *meshName);
  assert(mesh && "unknown mesh");

  ArrayRef<StringAttr> manualAxes = op.getManualAxes();
  auto meshComparator = AxisRefAttr::getMeshComparator(mesh);
  SmallVector<TensorShardingAttr> newShardings;
  StringAttr meshNameAttr = builder.getStringAttr(*meshName);

  if (std::optional<SmallVector<TensorShardingAttr>> newShardings =
          addUnusedManualAxesToReplicatedAxes(
              op.getInShardings().getShardings(), manualAxes, meshComparator,
              meshNameAttr)) {
    op.setInShardings(*newShardings);
  }

  if (std::optional<SmallVector<TensorShardingAttr>> newShardings =
          addUnusedManualAxesToReplicatedAxes(
              op.getOutShardings().getShardings(), manualAxes, meshComparator,
              meshNameAttr)) {
    op.setOutShardings(*newShardings);
  }
}

// Sorts the manual axes in mesh axis order.
void sortManualAxes(ManualComputationOp op) {
  std::optional<StringRef> meshName = getCommonMeshName(
      op.getInShardings().getShardings(), op.getOutShardings().getShardings());
  assert(meshName.has_value() &&
         "Expected no input/output ManualComputationOps to be canonicalized");
  MeshAttr mesh = getMeshAttr(op, meshName.value());
  assert(mesh && "unknown mesh");

  ArrayRef<StringAttr> manualAxes = op.getManualAxes();
  auto meshComparator = mesh.getAxisNameComparator();

  if (llvm::is_sorted(manualAxes, meshComparator)) {
    return;
  }
  SmallVector<StringAttr> sortedManualAxes = llvm::to_vector(manualAxes);
  llvm::sort(sortedManualAxes, meshComparator);
  op.setManualAxes(sortedManualAxes);
}

struct ManualAxesCleanupPass
    : public impl::ManualAxesCleanupPassBase<ManualAxesCleanupPass> {
  using ManualAxesCleanupPassBase::ManualAxesCleanupPassBase;

  void runOnOperation() final {
    getOperation()->walk([](ManualComputationOp op) {
      sortManualAxes(op);
      addUnusedManualAxesToReplicatedAxes(op);
    });
  }

 private:
  FrozenRewritePatternSet patterns;
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
