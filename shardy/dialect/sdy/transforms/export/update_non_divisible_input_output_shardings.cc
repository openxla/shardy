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
#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/ValueRange.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_UPDATENONDIVISIBLEINPUTOUTPUTSHARDINGSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// Gets the largest prefix sharding of the given `sharding` that evenly divides
// the tensor `type`.
//
// Shardings are updated on a per dimension basis, so if one dimension requires
// replication, other dimensions may not be.
//
// Examples:
// - mesh = <"x"=4, "y"=2>
//   - [{"x"}] : tensor<8xf32> -> [{"x"}] : tensor<8xf32>
//   - [{"x",?}] : tensor<4xf32> -> [{"x",?}] : tensor<4xf32>
//   - [{"x"}, {"y"}] : tensor<2x2xf32> -> [{"x":(1)2}, {"y"}] : tensor<2x2xf32>
//   - [{"x"}] : tensor<3xf32> -> [{}] : tensor<3xf32>
//   - [{"y"}] : tensor<3xf32> -> [{}] : tensor<3xf32>
//   - [{"x","y"}] : tensor<4xf32> -> [{"x"}] : tensor<4xf32>
//   - [{"y","x"}] : tensor<4xf32> -> [{"y","x":(1)2}] : tensor<4xf32>
// See update_non_divisible_input_output_shardings.mlir for more examples.
TensorShardingAttr getEvenlySharded(TensorShardingAttr sharding,
                                    ShapedType type, func::FuncOp funcOp) {
  MeshAttr mesh = sharding.getMesh(funcOp);
  assert(mesh && "unknown mesh");
  MLIRContext* ctx = funcOp->getContext();
  llvm::SmallVector<DimensionShardingAttr> newDimShardings;
  newDimShardings.reserve(sharding.getRank());
  for (auto [dimSharding, dimSize] :
       llvm::zip_equal(sharding.getDimShardings(), type.getShape())) {
    if (dimSize % dimSharding.getShardedSize(mesh) == 0) {
      // Exit early if we can divide the dimension evenly.
      newDimShardings.push_back(dimSharding);
      continue;
    }
    int64_t remainingDimSize = dimSize;
    llvm::ArrayRef<AxisRefAttr> oldAxes = dimSharding.getAxes();
    llvm::SmallVector<AxisRefAttr> newAxes;
    newAxes.reserve(oldAxes.size());
    int64_t newShardedSize = 1;
    while (!oldAxes.empty()) {
      AxisRefAttr currentAxis = oldAxes.front();
      oldAxes = oldAxes.drop_front();
      const int64_t gcd = std::gcd(currentAxis.getSize(mesh), remainingDimSize);
      if (gcd == 1) {
        // If there is no GCD with the current axis, we can't use any more axes.
        // Return current built up sharding.
        break;
      }
      newShardedSize *= gcd;
      if (currentAxis.getSize(mesh) == gcd) {
        // Get full axis. Move on to the next axis.
        newAxes.push_back(currentAxis);
        remainingDimSize /= gcd;
      } else {
        // `currentAxis` is larger than the `gcd`. Need to take a smaller chunk
        // of the current axis.
        newAxes.push_back(AxisRefAttr::get(
            ctx, currentAxis.getName(),
            SubAxisInfoAttr::get(ctx, currentAxis.getSubAxisPreSize(), gcd)));
        // Since we took the largest chunk of the `currentAxis`, with some
        // left over, we can't take anymore from it and thus can't also
        // consider any further/minor axes. Return the current build up
        // sharding.
        break;
      }
    }
    DimensionShardingAttr newDimSharding =
        DimensionShardingAttr::get(ctx, newAxes, dimSharding.getIsClosed());
    if (dimSize % newShardedSize != 0) {
      unreachableFormatv(
          "Failed to make input/output shardings evenly sharded. Started with "
          "tensor sharding {0}, tensor shape {1}, mesh {2}, miscalculated dim "
          "sharding {3}",
          sharding, type, attributeToString(mesh), newDimSharding);
    }
    newDimShardings.push_back(newDimSharding);
  }
  // NOTE: no need to account for replicated axes, since we end with a sharding
  // that covers less-than-or-equal amount of axes than we started with. So no
  // way the final sharding can use an axis/sub-axis from the replicated axes.
  return TensorShardingAttr::get(ctx, sharding.getMeshOrRef(), newDimShardings,
                                 sharding.getReplicatedAxes());
}

void updateValueShardings(
    TypeRange types,
    std::function<TensorShardingAttr(int64_t index)> getSharding,
    std::function<void(int64_t index, TensorShardingAttr sharding)> setSharding,
    func::FuncOp funcOp) {
  for (auto [index, type] : llvm::enumerate(types)) {
    TensorShardingAttr sharding = getSharding(index);
    if (auto tensorType = dynCastStaticShapedType(type);
        sharding && tensorType) {
      setSharding(index, getEvenlySharded(sharding, tensorType, funcOp));
    }
  }
}

void updateValueShardings(
    ValueRange values,
    func::FuncOp funcOp) {
  updateValueShardings(
        values.getTypes(),
        [&](int64_t index) { return getSharding(values[index]); },
        [&](int64_t index, TensorShardingAttr sharding) {
          setSharding(values[index], sharding);
        },
        funcOp);
}


struct UpdateNonDivisibleInputOutputShardingsPass
    : public impl::UpdateNonDivisibleInputOutputShardingsPassBase<
          UpdateNonDivisibleInputOutputShardingsPass> {
  using UpdateNonDivisibleInputOutputShardingsPassBase::
      UpdateNonDivisibleInputOutputShardingsPassBase;

  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    // Update arguments.
    updateValueShardings(
        funcOp.getArgumentTypes(),
        [&](int64_t index) { return getSharding(funcOp.getArgument(index)); },
        [&](int64_t index, TensorShardingAttr sharding) {
          setSharding(funcOp.getArgument(index), sharding);
        },
        funcOp);
    // Update results.
    updateValueShardings(
        funcOp.getResultTypes(),
        [&](int64_t index) { return getFuncResultSharding(funcOp, index); },
        [&](int64_t index, TensorShardingAttr sharding) {
          setFuncResultSharding(funcOp, index, sharding);
        },
        funcOp);

    // Update edge owner shardings for `ShardableDataFlowOp`s and
    // `ShardingRuleOp`s.
    // TODO: b/415294308 - Make this pass more efficient by updating shardings
    // all at once.
    funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
      return TypeSwitch<Operation*, WalkResult>(op)
          .Case<ShardableDataFlowOpInterface>([&](ShardableDataFlowOpInterface
                                                      shardableDataFlowOp)
                                                  -> WalkResult {
            if (shardableDataFlowOp.shouldKeepEdgeOwnerShardingsDivisible()) {
              updateValueShardings(
                  shardableDataFlowOp.getBlockArgumentEdgeOwners(), funcOp);
              updateValueShardings(shardableDataFlowOp.getOpResultEdgeOwners(),
                                   funcOp);
            }
            return WalkResult::skip();
          })
          .Case<ShardingRuleOpInterface>(
              [&](ShardingRuleOpInterface shardableRuleOp) -> WalkResult {
                if (shardableRuleOp.shouldKeepOutputShardingsDivisible()) {
                  updateValueShardings(shardableRuleOp->getResults(), funcOp);
                }
                return WalkResult::skip();
              })
          .Default([&](Operation* op) -> WalkResult {
            return WalkResult::advance();
          });
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
