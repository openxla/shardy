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

#include <cassert>
#include <cstdint>
#include <functional>
#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

namespace {

using func::FuncOp;

// Applies `callback` on `sharding` if present, and calls `setShardingFn` on the
// result if `transformShardings` is true.
void processSharding(TensorShardingAttr sharding,
                     const ValueOrFuncResult& valueOrFuncResult,
                     bool transformShardings,
                     TransformShardingForTensorFn callback,
                     std::function<void(TensorShardingAttr)> setShardingFn) {
  if (!sharding) {
    return;
  }
  TensorShardingAttr newSharding = callback(sharding, valueOrFuncResult);
  if (transformShardings && newSharding != sharding) {
    setShardingFn(newSharding);
  }
}

// Applies `callback` on the sharding of `valueOrFuncResult` if present, and
// replaces the sharding with the result if `transformShardings` is true.
void processSharding(const ValueOrFuncResult& valueOrFuncResult,
                     bool transformShardings,
                     TransformShardingForTensorFn callback) {
  if (auto* value = std::get_if<Value>(&valueOrFuncResult)) {
    processSharding(getSharding(*value), valueOrFuncResult, transformShardings,
                    callback, [&](TensorShardingAttr newSharding) {
                      setSharding(*value, newSharding);
                    });
  } else {
    auto [funcOp, resNum] = std::get<FuncResult>(valueOrFuncResult);
    processSharding(
        getFuncResultSharding(funcOp, resNum), valueOrFuncResult,
        transformShardings, callback,
        [funcOp = funcOp, resNum = resNum](TensorShardingAttr newSharding) {
          setFuncResultSharding(funcOp, resNum, newSharding);
        });
  }
}

// Applies `callback` on each sharding in `shardings` if present, and
// corresponding value in `values`, and calls `setShardingsFn` on the results if
// `transformShardings` is true.
void processShardings(
    ArrayRef<TensorShardingAttr> shardings, ValueRange values,
    bool transformShardings, TransformShardingForTensorFn callback,
    std::function<void(ArrayRef<TensorShardingAttr>)> setShardingsFn) {
  if (shardings.empty()) {
    return;
  }
  // To handle the case where an op has a maximal sharding but returns no,
  // values, we create a null value to pass to the callback. Note we need to
  // keep a stack variable of the value since `values` is a ValueRange which
  // only holds a reference.
  // TODO(b/391545244): do something smarter than sticking a null `Value` into
  // the callback. Not an issue now but can be in the future.
  Value emptyMaximalValue;
  if (values.empty()) {
    // This should be a single maximal sharding.
    assert(shardings.size() == 1);
    values = emptyMaximalValue;
  }

  if (!transformShardings) {
    for (auto [sharding, value] : llvm::zip_equal(shardings, values)) {
      callback(sharding, value);
    }
    return;
  }

  SmallVector<TensorShardingAttr> newShardings;
  for (auto [sharding, value] : llvm::zip_equal(shardings, values)) {
    newShardings.push_back(callback(sharding, value));
  }
  setShardingsFn(newShardings);
}

// Same as above but for `TensorShardingPerValueAttr`.
void processShardings(
    TensorShardingPerValueAttr shardings, ValueRange values,
    bool transformShardings, TransformShardingForTensorFn callback,
    std::function<void(TensorShardingPerValueAttr)> setShardingsFn) {
  return processShardings(shardings.getShardings(), values, transformShardings,
                          callback,
                          [&](ArrayRef<TensorShardingAttr> newShardings) {
                            setShardingsFn(TensorShardingPerValueAttr::get(
                                shardings.getContext(), newShardings));
                          });
}

void walkShardings(Operation* rootOp, TransformShardingForTensorFn callback,
                   ConsumeOpFn consumeOpFn, bool transformShardings) {
  rootOp->walk<WalkOrder::PreOrder>([&](Operation* op) {
    consumeOpFn(op);
    TypeSwitch<Operation*, void>(op)
        .Case<FuncOp>([&](FuncOp funcOp) {
          for (BlockArgument arg : funcOp.getArguments()) {
            processSharding(arg, transformShardings, callback);
          }
          for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
            processSharding(FuncResult(funcOp, resNum), transformShardings,
                            callback);
          }
        })
        .Case<ManualComputationOp>(
            [&](ManualComputationOp manualComputationOp) {
              processShardings(
                  manualComputationOp.getInShardings(),
                  manualComputationOp.getBody().getArguments(),
                  transformShardings, callback,
                  [&](TensorShardingPerValueAttr newShardings) {
                    manualComputationOp.setInShardingsAttr(newShardings);
                  });
              processShardings(
                  manualComputationOp.getOutShardings(),
                  manualComputationOp.getResults(), transformShardings,
                  callback, [&](TensorShardingPerValueAttr newShardings) {
                    manualComputationOp.setOutShardingsAttr(newShardings);
                  });
            })
        .Case<ShardableDataFlowOpInterface>(
            [&](ShardableDataFlowOpInterface shardableDataFlowOp) {
              processShardings(
                  shardableDataFlowOp.getBlockArgumentEdgeOwnerShardings(),
                  shardableDataFlowOp.getBlockArgumentEdgeOwners(),
                  transformShardings, callback,
                  [&](ArrayRef<TensorShardingAttr> newShardings) {
                    shardableDataFlowOp.setBlockArgumentEdgeOwnerShardings(
                        newShardings);
                  });
              processShardings(
                  shardableDataFlowOp.getOpResultEdgeOwnerShardings(),
                  shardableDataFlowOp.getOpResultEdgeOwners(),
                  transformShardings, callback,
                  [&](ArrayRef<TensorShardingAttr> newShardings) {
                    shardableDataFlowOp.setOpResultEdgeOwnerShardings(
                        newShardings);
                  });
            })
        .Default([&](Operation* op) {
          if (op->getNumResults() == 1) {
            // For ops with a single result, we use `get/setSharding` instead of
            // `get/setShardings`, since the latter only handle ops with an
            // unregistered sharding attribute, to also handle SDY ops like
            // `ShardingConstraintOp`.
            Value result = op->getResult(0);
            processSharding(result, transformShardings, callback);
          } else {
            processShardings(getShardings(op), op->getResults(),
                             transformShardings, callback,
                             [&](ArrayRef<TensorShardingAttr> newShardings) {
                               setShardings(op, newShardings);
                             });
          }
        });
  });
}

}  // namespace

void transformSharding(const ValueOrFuncResult& valueOrFuncResult,
                       TransformShardingFn transformFn) {
  processSharding(
      valueOrFuncResult, /*transformShardings=*/true,
      [transformFn](TensorShardingAttr newSharding, const ValueOrFuncResult&) {
        return transformFn(newSharding);
      });
}

void walkShardings(Operation* rootOp, ConsumeShardingAndTensorFn consumeFn,
                   ConsumeOpFn consumeOpFn) {
  walkShardings(
      rootOp,
      [consumeFn](TensorShardingAttr sharding,
                  const ValueOrFuncResult& valueOrFuncResult) {
        consumeFn(sharding, valueOrFuncResult);
        return sharding;
      },
      consumeOpFn,
      /*transformShardings=*/false);
}

void walkShardings(Operation* rootOp, ConsumeShardingFn consumeFn,
                   ConsumeOpFn consumeOpFn) {
  walkShardings(
      rootOp,
      [consumeFn](TensorShardingAttr sharding, const ValueOrFuncResult&) {
        consumeFn(sharding);
      },
      consumeOpFn);
}

void transformShardings(Operation* rootOp,
                        TransformShardingForTensorFn transformFn,
                        ConsumeOpFn consumeOpFn) {
  walkShardings(rootOp, transformFn, consumeOpFn, /*transformShardings=*/true);
}

void transformShardings(Operation* rootOp, TransformShardingFn transformFn,
                        ConsumeOpFn consumeOpFn) {
  transformShardings(
      rootOp,
      [transformFn](TensorShardingAttr newSharding, const ValueOrFuncResult&) {
        return transformFn(newSharding);
      },
      consumeOpFn);
}

}  // namespace sdy
}  // namespace mlir
