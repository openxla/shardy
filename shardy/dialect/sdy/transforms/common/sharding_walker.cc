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

#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"

#include <cstdint>
#include <functional>
#include <iterator>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

namespace {

using func::FuncOp;

// Applies `callback` on `sharding` if present, and calls `setShardingFn` on the
// result if `transformShardings` is true.
void processSharding(TensorShardingAttr sharding, bool transformShardings,
                     TransformShardingFn callback,
                     std::function<void(TensorShardingAttr)> setShardingFn) {
  if (!sharding) {
    return;
  }
  TensorShardingAttr newSharding = callback(sharding);
  if (transformShardings && newSharding != sharding) {
    setShardingFn(newSharding);
  }
}

// Applies `callback` on each sharding in `shardings` if present, and calls
// `setShardingFn` on the results if `transformShardings` is true.
void processShardings(
    ArrayRef<TensorShardingAttr> shardings, bool transformShardings,
    TransformShardingFn callback,
    std::function<void(ArrayRef<TensorShardingAttr>)> setShardingsFn) {
  if (shardings.empty() || !transformShardings) {
    llvm::for_each(shardings, callback);
    return;
  }

  SmallVector<TensorShardingAttr> newShardings;
  llvm::transform(shardings, std::back_inserter(newShardings), callback);
  setShardingsFn(newShardings);
}

// Same as above but for `TensorShardingPerValueAttr`.
void processShardings(
    TensorShardingPerValueAttr shardings, bool transformShardings,
    TransformShardingFn callback,
    std::function<void(TensorShardingPerValueAttr)> setShardingsFn) {
  return processShardings(shardings.getShardings(), transformShardings,
                          callback,
                          [&](ArrayRef<TensorShardingAttr> newShardings) {
                            setShardingsFn(TensorShardingPerValueAttr::get(
                                shardings.getContext(), newShardings));
                          });
}

void walkShardings(Operation* rootOp, TransformShardingFn callback,
                   ConsumeOpFn consumeOpFn, bool transformShardings) {
  rootOp->walk([&](Operation* op) {
    consumeOpFn(op);
    TypeSwitch<Operation*, void>(op)
        .Case<FuncOp>([&](FuncOp funcOp) {
          for (BlockArgument arg : funcOp.getArguments()) {
            processSharding(getSharding(arg), transformShardings, callback,
                            [&](TensorShardingAttr newSharding) {
                              setSharding(arg, newSharding);
                            });
          }
          for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
            processSharding(
                getFuncResultSharding(funcOp, resNum), transformShardings,
                callback, [&](TensorShardingAttr newSharding) {
                  setFuncResultSharding(funcOp, resNum, newSharding);
                });
          }
        })
        .Case<ShardableDataFlowOpInterface>(
            [&](ShardableDataFlowOpInterface shardableDataFlowOp) {
              processShardings(
                  shardableDataFlowOp.getBlockArgumentEdgeOwnerShardings(),
                  transformShardings, callback,
                  [&](ArrayRef<TensorShardingAttr> newShardings) {
                    shardableDataFlowOp.setBlockArgumentEdgeOwnerShardings(
                        newShardings);
                  });
              processShardings(
                  shardableDataFlowOp.getOpResultEdgeOwnerShardings(),
                  transformShardings, callback,
                  [&](ArrayRef<TensorShardingAttr> newShardings) {
                    shardableDataFlowOp.setOpResultEdgeOwnerShardings(
                        newShardings);
                  });
            })
        .Case<ManualComputationOp>(
            [&](ManualComputationOp manualComputationOp) {
              processShardings(
                  manualComputationOp.getInShardings(), transformShardings,
                  callback, [&](TensorShardingPerValueAttr newShardings) {
                    manualComputationOp.setInShardingsAttr(newShardings);
                  });
              processShardings(
                  manualComputationOp.getOutShardings(), transformShardings,
                  callback, [&](TensorShardingPerValueAttr newShardings) {
                    manualComputationOp.setOutShardingsAttr(newShardings);
                  });
            })
        .Default([&](Operation* op) {
          if (op->getNumResults() == 1) {
            // For ops with a single result, we use `get/setSharding` instead of
            // `get/setShardings`, since the latter only handle ops with an
            // unregistered sharding attribute, to also handle SDY ops like
            // `ShardingConstraintOp`.
            Value result = op->getResult(0);
            processSharding(getSharding(result), transformShardings, callback,
                            [&](TensorShardingAttr newSharding) {
                              setSharding(result, newSharding);
                            });
          } else {
            processShardings(getShardings(op), transformShardings, callback,
                             [&](ArrayRef<TensorShardingAttr> newShardings) {
                               setShardings(op, newShardings);
                             });
          }
        });
  });
}

}  // namespace

void walkShardings(Operation* rootOp, ConsumeShardingFn consumeFn,
                   ConsumeOpFn consumeOpFn) {
  walkShardings(
      rootOp,
      [consumeFn](TensorShardingAttr sharding) {
        consumeFn(sharding);
        return sharding;
      },
      consumeOpFn,
      /*transformShardings=*/false);
}

void transformShardings(Operation* rootOp, TransformShardingFn transformFn,
                        ConsumeOpFn consumeOpFn) {
  walkShardings(rootOp, transformFn, consumeOpFn, /*transformShardings=*/true);
}

}  // namespace sdy
}  // namespace mlir
