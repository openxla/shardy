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

#include "shardy/dialect/sdy/transforms/propagation/user_priority_propagation.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/common/file_utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/auto_partitioner_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/op_priority_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_group_map.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_USERPRIORITYPROPAGATIONPASS
#include "shardy/dialect/sdy/transforms/propagation/passes.h.inc"

namespace {

using func::FuncOp;

// A value and its original sharding.
struct ValueSharding {
  Value value;
  TensorShardingAttr sharding;

  ValueSharding(Value value, TensorShardingAttr sharding)
      : value(value), sharding(sharding) {}
};

// A function result number and its original sharding.
struct FuncResultSharding {
  int64_t resNum;
  TensorShardingAttr sharding;

  FuncResultSharding(int64_t resNum, TensorShardingAttr sharding)
      : resNum(resNum), sharding(sharding) {}
};

struct FuncOpAndResultShardings {
  FuncOp funcOp;
  SmallVector<FuncResultSharding> funcResultShardings;

  explicit FuncOpAndResultShardings(
      FuncOp funcOp, ArrayRef<FuncResultSharding> funcResultShardings = {})
      : funcOp(funcOp), funcResultShardings(funcResultShardings) {}
};

// References to a subset of value and function result shardings of a function.
struct ShardingReferences {
  SmallVector<ValueSharding> valueShardings;
  SmallVector<FuncOpAndResultShardings> funcOpAndResultShardingsVec;
};

using PriorityToShardingReferences =
    llvm::SmallMapVector<int64_t, ShardingReferences, 4>;

using PriorityShardingReferences = std::pair<int64_t, ShardingReferences>;

// Returns an updated sharding of `curSharding` by setting the sharding of each
// dimension whose priority equals `curPriority` to the corresponding dimension
// sharding in `originalSharding`, and removing the sharding axes from the
// replicated set.
TensorShardingAttr getUpdatedShardingForPriority(
    TensorShardingAttr curSharding, TensorShardingAttr originalSharding,
    int64_t curPriority) {
  assert(curSharding);
  SmallVector<DimensionShardingAttr> newDimShardings(
      curSharding.getDimShardings());
  llvm::SmallDenseSet<AxisRefAttr> seenAxesWithCurrentPriority;
  for (auto [newDimSharding, originalDimSharding] :
       llvm::zip_equal(newDimShardings, originalSharding.getDimShardings())) {
    if (originalDimSharding.getPriorityOrDefault() == curPriority) {
      assert(newDimSharding.getIsClosed() && newDimSharding.emptyAxes());
      newDimSharding = originalDimSharding.dropPriority();
      seenAxesWithCurrentPriority.insert(originalDimSharding.axis_begin(),
                                         originalDimSharding.axis_end());
    }
  }
  // No need to sort the new vector since `copy_if` maintains order.
  SmallVector<AxisRefAttr> newReplicatedAxes;
  llvm::copy_if(curSharding.getReplicatedAxes(),
                std::back_inserter(newReplicatedAxes), [&](AxisRefAttr axis) {
                  return !seenAxesWithCurrentPriority.contains(axis);
                });
  return TensorShardingAttr::get(curSharding.getContext(),
                                 curSharding.getMeshOrRef(), newDimShardings,
                                 newReplicatedAxes);
}

// Updates the current sharding of all referenced values and function results in
// `shardingReferences` for the given priority (see
// `getUpdatedShardingForPriority`).
void updateReferencedShardingsForPriority(
    const ShardingReferences& shardingReferences, int64_t curPriority) {
  for (auto [value, originalSharding] : shardingReferences.valueShardings) {
    setSharding(value, getUpdatedShardingForPriority(
                           getSharding(value), originalSharding, curPriority));
  }

  for (const auto& [funcOp, resultShardings] :
       shardingReferences.funcOpAndResultShardingsVec) {
    for (auto [resNum, originalSharding] : resultShardings) {
      setFuncResultSharding(
          funcOp, resNum,
          getUpdatedShardingForPriority(getFuncResultSharding(funcOp, resNum),
                                        originalSharding, curPriority));
    }
  }
}

// Returns an initialized sharding for the first iteration (priority 0) such
// that all dimension shardings in `originalSharding` that have a priority >0
// become empty and closed, and their original sharding axes are moved to the
// replicated set.
//
// Dimension shardings with an explicit priority 0 will be the same except their
// priority is dropped.
TensorShardingAttr getInitializedSharding(TensorShardingAttr originalSharding,
                                          Operation* op) {
  MLIRContext* ctx = originalSharding.getContext();
  SmallVector<DimensionShardingAttr> newDimShardings(
      originalSharding.getDimShardings());
  SmallVector<AxisRefAttr> newReplicatedAxes(
      originalSharding.getReplicatedAxes());
  for (DimensionShardingAttr& dimSharding : newDimShardings) {
    if (dimSharding.getPriorityOrDefault() == 0) {
      dimSharding = dimSharding.dropPriority();
    } else {
      newReplicatedAxes.append(dimSharding.axis_begin(),
                               dimSharding.axis_end());
      dimSharding = DimensionShardingAttr::get(ctx, {}, /*isClosed=*/true);
    }
  }
  MeshAttr mesh = originalSharding.getMesh(op);
  assert(mesh && "unknown mesh");
  llvm::sort(newReplicatedAxes, AxisRefAttr::getMeshComparator(mesh));
  // TODO(tomnatan): we need to merge split axes and split them again when
  // updating? or can we assume we won't see split axes?

  return TensorShardingAttr::get(ctx, originalSharding.getMeshOrRef(),
                                 newDimShardings, newReplicatedAxes);
}

// Clears `priorities` and add all non-zero priorities in `sharding` to it.
void clearAndAddNonZeroPriorities(TensorShardingAttr sharding,
                                  llvm::SmallDenseSet<int64_t>& priorities) {
  priorities.clear();
  for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
    if (int64_t priority = dimSharding.getPriorityOrDefault(); priority > 0) {
      priorities.insert(priority);
    }
  }
}

// If `value` has a sharding, adds that original sharding to
// `priorityToShardingReferences` for every non-zero priority in its dimension
// shardings, and initializes the sharding for the first iteration (see
// `getInitializedSharding`).
void addValueShardingToPriorityMapAndInitialize(
    Value value, PriorityToShardingReferences& priorityToShardingReferences,
    llvm::SmallDenseSet<int64_t>& prioritiesInSharding) {
  TensorShardingAttr sharding = getSharding(value);
  if (!sharding) {
    return;
  }
  clearAndAddNonZeroPriorities(sharding, prioritiesInSharding);
  for (int64_t priority : prioritiesInSharding) {
    priorityToShardingReferences[priority].valueShardings.emplace_back(
        value, sharding);
  }
  setSharding(value, getInitializedSharding(sharding, getOwningOp(value)));
}

// Gets the result shardings of a `funcOp` for a given `priority`. If `funcOp`
// doesn't have a result sharding for `priority`, adds it and initializes it,
// and returns a reference to it.
SmallVector<FuncResultSharding>& getFuncResultShardings(
    SmallVector<FuncOpAndResultShardings>& funcOpAndResultShardingsVec,
    FuncOp funcOp) {
  // Add a new entry for `funcOp` if any of the conditions holds:
  // 1. this is the first time we are adding func shardings for this priority,
  //    so there is no `FuncOp` yet
  // 2. this is a new `funcOp` that this function is running on
  if (funcOpAndResultShardingsVec.empty() ||
      funcOpAndResultShardingsVec.back().funcOp != funcOp) {
    funcOpAndResultShardingsVec.emplace_back(funcOp);
  }
  return funcOpAndResultShardingsVec.back().funcResultShardings;
}

// If `funcOp` has a sharding for result `resNum`, adds that original sharding
// to `priorityToShardingReferences` for every non-zero priority in its
// dimension shardings, and initializes the sharding for the first iteration
// (see `getInitializedSharding`).
void addFuncResultShardingToPriorityMapAndInitialize(
    FuncOp funcOp, int resNum,
    PriorityToShardingReferences& priorityToShardingReferences,
    llvm::SmallDenseSet<int64_t>& prioritiesInSharding) {
  auto sharding = getFuncResultSharding(funcOp, resNum);
  if (!sharding) {
    return;
  }
  clearAndAddNonZeroPriorities(sharding, prioritiesInSharding);
  for (int64_t priority : prioritiesInSharding) {
    getFuncResultShardings(
        priorityToShardingReferences[priority].funcOpAndResultShardingsVec,
        funcOp)
        .emplace_back(resNum, sharding);
  }
  setFuncResultSharding(funcOp, resNum,
                        getInitializedSharding(sharding, funcOp));
}

// Traverses `funcOp` and for each value or func result with a sharding:
//   - Adds {value / result number, sharding} to the sharding references of each
//     non-zero priority in its dimension shardings.
//   - Initializes the tensor's current sharding for the first iteration (see
//     `getInitializedSharding`).
// If a dimension sharding doesn't have a user-defined priority,
// `kDefaultPriority` is used.
//
// Returns a vector of `PriorityShardingReferences` sorted by priority.
SmallVector<PriorityShardingReferences>
getShardingReferencesPerPriorityAndInitialize(ModuleOp moduleOp) {
  PriorityToShardingReferences priorityToValueShardings;
  llvm::SmallDenseSet<int64_t> prioritiesInSharding;
  moduleOp.walk([&](Operation* op) {
    if (isa<FuncOp, ManualComputationOp>(op)) {
      // These ops have block arguments with attached shardings.
      for (Value arg : op->getRegion(0).getArguments()) {
        addValueShardingToPriorityMapAndInitialize(
            arg, priorityToValueShardings, prioritiesInSharding);
      }
    }
    for (Value result : op->getResults()) {
      addValueShardingToPriorityMapAndInitialize(
          result, priorityToValueShardings, prioritiesInSharding);
    }
  });
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    for (int resNum = 0; resNum < funcOp.getNumResults(); resNum++) {
      addFuncResultShardingToPriorityMapAndInitialize(
          funcOp, resNum, priorityToValueShardings, prioritiesInSharding);
    }
  }
  // Finally we take the vector of `PriorityValueShardings` and sort it by
  // priority.
  SmallVector<PriorityShardingReferences> priorityShardingReferencesVec =
      priorityToValueShardings.takeVector();
  llvm::sort(
      priorityShardingReferencesVec,
      [](const PriorityShardingReferences& a,
         const PriorityShardingReferences& b) { return a.first < b.first; });

  return priorityShardingReferencesVec;
}

// The user-priority propagation pass that uses the default implementation of
// `UserPriorityPropagationPassImpl`.
struct UserPriorityPropagationPass
    : public impl::UserPriorityPropagationPassBase<
          UserPriorityPropagationPass> {
  using UserPriorityPropagationPassBase::UserPriorityPropagationPassBase;

  // NOLINTBEGIN(clang-diagnostic-shadow-field)
  explicit UserPriorityPropagationPass(bool keepShardingRules,
                                       StringRef dumpDirectory,
                                       bool conservativePropagation) {
    // NOLINTEND(clang-diagnostic-shadow-field)
    this->keepShardingRules = keepShardingRules;
    this->dumpDirectory = dumpDirectory.str();
    this->conservativePropagation = conservativePropagation;
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    UserPriorityPropagationPassImpl::getDependentDialects(registry);
    // If we have a dynamically loaded pipeline, we need to add their dependency
    // ahead of executing this pass, to avoid multi-threading issues within the
    // pass.
    if (AutoPartitionerRegistry::isRegistered()) {
      AutoPartitionerRegistry::getDependentDialects(registry);
    }
  }
};

void saveModuleOpAfterPriority(ModuleOp moduleOp, StringRef dumpDirectory,
                               int64_t priority) {
  saveModuleOp(
      moduleOp, dumpDirectory,
      llvm::formatv("sdy_module_after_user_priority_{0}", priority).str());
}

}  // namespace

LogicalResult UserPriorityPropagationPassImpl::propagate(
    ModuleOp moduleOp, const SymbolTable& symbolTable,
    const ShardingGroupMap& shardingGroupMap,
    GetDirectionToPropagateFn getDirectionToPropagate) {
  SmallVector<PriorityShardingReferences> shardingReferencesPerPriority =
      getShardingReferencesPerPriorityAndInitialize(moduleOp);
  // We first run the first iteration (priority 0):
  if (failed(OpPriorityPropagationPassImpl::propagate(
          moduleOp, symbolTable, shardingGroupMap, getDirectionToPropagate))) {
    return failure();
  }
  saveModuleOpAfterPriority(moduleOp, dumpDirectory, 0);
  // Then we run the remaining iterations (priority >0):
  for (const auto& [priority, shardingReferences] :
       shardingReferencesPerPriority) {
    updateReferencedShardingsForPriority(shardingReferences, priority);
    if (failed(OpPriorityPropagationPassImpl::propagate(
            moduleOp, symbolTable, shardingGroupMap,
            getDirectionToPropagate))) {
      return failure();
    }
    saveModuleOpAfterPriority(moduleOp, dumpDirectory, priority);
  }

  // Finally we run automatic partitioning if enabled by the user
  if (auto useAutoSpmdPartitioning =
          moduleOp->getAttrOfType<BoolAttr>("mhlo.use_auto_spmd_partitioning");
      useAutoSpmdPartitioning && useAutoSpmdPartitioning.getValue()) {
    PassManager autoPartitionerPm(moduleOp.getContext());
    AutoPartitionerRegistry::addPasses(autoPartitionerPm);
    autoPartitionerPm.addPass(createSaveModuleOpPass(
        dumpDirectory, "sdy_module_after_auto_partitioning"));
    if (failed(runPipeline(autoPartitionerPm, moduleOp))) {
      return failure();
    }
  }

  return success();
}

std::unique_ptr<Pass> createUserPriorityPropagationPass(
    bool keepShardingRules, StringRef dumpDirectory,
    bool conservativePropagation) {
  return std::make_unique<UserPriorityPropagationPass>(
      keepShardingRules, dumpDirectory, conservativePropagation);
}

}  // namespace sdy
}  // namespace mlir
