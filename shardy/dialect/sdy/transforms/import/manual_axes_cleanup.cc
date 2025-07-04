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
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_MANUALAXESCLEANUPPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

// Returns whether the new shardings needed to be updated to account for unused
// manual axes, and the new shardings.
//
// If a sharding that needs to be updated is bound to an empty mesh, replaces
// its mesh with `commonMeshOrRef`.
std::optional<SmallVector<TensorShardingAttr>>
addUnusedManualAxesToReplicatedAxes(ArrayRef<TensorShardingAttr> shardings,
                                    TypeRange types,
                                    ArrayRef<AxisRefAttr> manualAxes,
                                    Attribute commonMeshOrRef,
                                    const SymbolTable& symbolTable) {
  SmallVector<TensorShardingAttr> newShardings;
  bool modified = false;
  for (auto [sharding, type] : llvm::zip_equal(shardings, types)) {
    if (!isa<ShapedType>(type)) {
      newShardings.push_back(sharding);
      continue;
    }

    SmallVector<AxisRefAttr> axesInSharding;
    sharding.forEachAxisRef([&](AxisRefAttr axis) {
      axesInSharding.push_back(axis);
    });

    MeshAttr mesh = getMeshOrLookup(symbolTable, commonMeshOrRef);
    SmallVector<AxisRefAttr> unusedManualAxes =
        getAxisSetDiff(manualAxes, axesInSharding, mesh);
    if (unusedManualAxes.empty()) {
      // Already uses all the manual axes, no need to create a new one.
      newShardings.push_back(sharding);
      continue;
    }

    SmallVector<AxisRefAttr> newReplicatedAxes =
        llvm::to_vector(sharding.getReplicatedAxes());
    llvm::append_range(newReplicatedAxes, unusedManualAxes);
    sortAndMergeAxes(newReplicatedAxes, mesh);
    newShardings.push_back(TensorShardingAttr::get(
        sharding.getContext(), commonMeshOrRef, sharding.getDimShardings(),
        newReplicatedAxes, sharding.getUnreducedAxes()));
    modified = true;
  }
  return modified ? std::make_optional(newShardings) : std::nullopt;
}

// Adds any unused manual axes to the replicated_axes list for each in/out
// sharding
void addUnusedManualAxesToReplicatedAxes(ManualComputationOp op, MeshAttr mesh,
                                         Attribute commonMeshOrRef,
                                         const SymbolTable& symbolTable) {
  OpBuilder builder(op);

  SmallVector<AxisRefAttr> manualAxes;
  manualAxes.reserve(op.getManualAxes().size());
  llvm::transform(op.getManualAxes(), std::back_inserter(manualAxes),
                  [&](StringAttr axis) {
                    return AxisRefAttr::get(builder.getContext(), axis);
                  });

  if (std::optional<SmallVector<TensorShardingAttr>> newShardings =
          addUnusedManualAxesToReplicatedAxes(
              op.getInShardings().getShardings(), op->getOperandTypes(),
              manualAxes, commonMeshOrRef, symbolTable)) {
    op.setInShardings(*newShardings);
  }

  if (std::optional<SmallVector<TensorShardingAttr>> newShardings =
          addUnusedManualAxesToReplicatedAxes(
              op.getOutShardings().getShardings(), op->getResultTypes(),
              manualAxes, commonMeshOrRef, symbolTable)) {
    op.setOutShardings(*newShardings);
  }
}

// Sorts the manual axes in mesh axis order.
void sortManualAxes(ManualComputationOp op, MeshAttr mesh) {
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
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    moduleOp->walk([&](ManualComputationOp op) {
      ArrayRef<TensorShardingAttr> inShardings =
          op.getInShardings().getShardings();
      ArrayRef<TensorShardingAttr> outShardings =
          op.getOutShardings().getShardings();
      if (inShardings.empty() && outShardings.empty()) {
        // Nothing to do.
        return;
      }
      Attribute meshOrRef =
          getCommonMeshOrRef(inShardings, outShardings, symbolTable);
      MeshAttr mesh =
          meshOrRef ? getMeshOrLookup(symbolTable, meshOrRef) : nullptr;
      assert(mesh && "expected inputs and outputs to have a common mesh");
      sortManualAxes(op, mesh);
      addUnusedManualAxesToReplicatedAxes(op, mesh, meshOrRef, symbolTable);
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
