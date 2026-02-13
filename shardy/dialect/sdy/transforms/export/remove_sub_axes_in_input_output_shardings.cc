/* Copyright 2026 The Shardy Authors.

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
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_REMOVESUBAXESININPUTOUTPUTSHARDINGSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

TensorShardingAttr removeSubAxesInDimensionShardings(
    TensorShardingAttr sharding) {
  MLIRContext* ctx = sharding.getContext();
  SmallVector<DimensionShardingAttr> newDimShardings;
  newDimShardings.reserve(sharding.getRank());
  for (DimensionShardingAttr oldDimSharding : sharding.getDimShardings()) {
    SmallVector<AxisRefAttr> newAxes;
    for (AxisRefAttr axis : oldDimSharding.getAxes()) {
      if (axis.getSubAxisInfo()) {
        break;
      }
      newAxes.push_back(axis);
    }
    newDimShardings.push_back(
        DimensionShardingAttr::get(ctx, newAxes, oldDimSharding.getIsClosed(),
                                   oldDimSharding.getPriority()));
  }
  return TensorShardingAttr::get(ctx, sharding.getMeshOrRef(), newDimShardings,
                                 sharding.getReplicatedAxes(),
                                 sharding.getUnreducedAxes());
}

void updateValueShardings(
    int64_t numShardings,
    std::function<TensorShardingAttr(int64_t index)> getSharding,
    std::function<void(int64_t index, TensorShardingAttr sharding)>
        setSharding) {
  for (int64_t index = 0; index < numShardings; index++) {
    if (TensorShardingAttr sharding = getSharding(index)) {
      setSharding(index, removeSubAxesInDimensionShardings(sharding));
    }
  }
}

struct RemoveSubAxesInInputOutputShardingsPass
    : public impl::RemoveSubAxesInInputOutputShardingsPassBase<
          RemoveSubAxesInInputOutputShardingsPass> {
  using RemoveSubAxesInInputOutputShardingsPassBase::
      RemoveSubAxesInInputOutputShardingsPassBase;

  void runOnOperation() final {
    for (auto funcOp : getOperation().getOps<func::FuncOp>()) {
      // Update arguments.
      updateValueShardings(
          funcOp.getNumArguments(),
          [&](int64_t index) { return getSharding(funcOp.getArgument(index)); },
          [&](int64_t index, TensorShardingAttr sharding) {
            setSharding(funcOp.getArgument(index), sharding);
          });
      // Update results.
      updateValueShardings(
          funcOp.getNumResults(),
          [&](int64_t index) { return getFuncResultSharding(funcOp, index); },
          [&](int64_t index, TensorShardingAttr sharding) {
            setFuncResultSharding(funcOp, index, sharding);
          });
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
