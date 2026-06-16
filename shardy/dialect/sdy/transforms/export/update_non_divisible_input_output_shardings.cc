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
#include <cstdint>
#include <functional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_UPDATENONDIVISIBLEINPUTOUTPUTSHARDINGSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

void updateValueShardings(
    TypeRange types,
    std::function<TensorShardingAttr(int64_t index)> getSharding,
    std::function<void(int64_t index, TensorShardingAttr sharding)> setSharding,
    const SymbolTable& symbolTable) {
  for (auto [index, type] : llvm::enumerate(types)) {
    TensorShardingAttr sharding = getSharding(index);
    if (auto tensorType = dynCastStaticShapedType(type);
        sharding && tensorType) {
      setSharding(index, getEvenlySharded(sharding, tensorType, symbolTable));
    }
  }
}

void updateValueShardings(
    ValueRange values, ArrayRef<TensorShardingAttr> shardings,
    std::function<void(ArrayRef<TensorShardingAttr> newShardings)> setShardings,
    const SymbolTable& symbolTable) {
  if (shardings.empty()) {
    return;
  }
  SmallVector<TensorShardingAttr> newShardings = llvm::to_vector(shardings);
  for (auto [type, sharding] :
       llvm::zip_equal(values.getTypes(), newShardings)) {
    if (auto tensorType = dynCastStaticShapedType(type)) {
      sharding = getEvenlySharded(sharding, tensorType, symbolTable);
    }
  }
  setShardings(newShardings);
}

struct UpdateNonDivisibleInputOutputShardingsPass
    : public impl::UpdateNonDivisibleInputOutputShardingsPassBase<
          UpdateNonDivisibleInputOutputShardingsPass> {
  using UpdateNonDivisibleInputOutputShardingsPassBase::
      UpdateNonDivisibleInputOutputShardingsPassBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      // Update arguments.
      updateValueShardings(
          funcOp.getArgumentTypes(),
          [&](int64_t index) { return getSharding(funcOp.getArgument(index)); },
          [&](int64_t index, TensorShardingAttr sharding) {
            setSharding(funcOp.getArgument(index), sharding);
          },
          symbolTable);
      // Update results.
      updateValueShardings(
          funcOp.getResultTypes(),
          [&](int64_t index) { return getFuncResultSharding(funcOp, index); },
          [&](int64_t index, TensorShardingAttr sharding) {
            setFuncResultSharding(funcOp, index, sharding);
          },
          symbolTable);
    }
    moduleOp.walk([&](func::CallOp callOp) {
      // Update call results.
      updateValueShardings(
          callOp->getResults(), getShardings(callOp),
          [&](ArrayRef<TensorShardingAttr> shardings) {
            setShardings(callOp, shardings);
          },
          symbolTable);
    });

    // Update edge owner shardings for `ShardableDataFlowOp`s and
    // `ShardingRuleOp`s.
    moduleOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
      TypeSwitch<Operation*>(op)
          .Case([&](ShardableDataFlowOpInterface shardableDataFlowOp) {
            if (shardableDataFlowOp.shouldKeepEdgeOwnerShardingsDivisible()) {
              updateValueShardings(
                  shardableDataFlowOp.getBlockArgumentEdgeOwners(),
                  shardableDataFlowOp.getBlockArgumentEdgeOwnerShardings(),
                  [&](ArrayRef<TensorShardingAttr> shardings) {
                    shardableDataFlowOp.setBlockArgumentEdgeOwnerShardings(
                        shardings);
                  },
                  symbolTable);
              updateValueShardings(
                  shardableDataFlowOp.getOpResultEdgeOwners(),
                  shardableDataFlowOp.getOpResultEdgeOwnerShardings(),
                  [&](ArrayRef<TensorShardingAttr> shardings) {
                    shardableDataFlowOp.setOpResultEdgeOwnerShardings(
                        shardings);
                  },
                  symbolTable);
            }
          })
          .Case(
              [&](ShardingRuleOpInterface shardableRuleOp) {
                if (shardableRuleOp.shouldKeepOutputShardingsDivisible()) {
                  updateValueShardings(
                      shardableRuleOp->getResults(),
                      getShardings(shardableRuleOp),
                      [&](ArrayRef<TensorShardingAttr> shardings) {
                        setShardings(shardableRuleOp, shardings);
                      },
                      symbolTable);
                }
              });
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
