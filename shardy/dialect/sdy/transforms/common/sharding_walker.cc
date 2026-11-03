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
#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

namespace {

using func::FuncOp;

// Applies `callback` on `sharding` if present, and calls `setShardingFn` on the
// result if it changed.
void processSharding(
    TensorShardingAttr sharding, const ValueOrFuncResult& valueOrFuncResult,
    const TransformShardingForTensorFn& callback,
    llvm::function_ref<void(TensorShardingAttr)> setShardingFn) {
  if (!sharding) {
    return;
  }
  TensorShardingAttr newSharding = callback(sharding, valueOrFuncResult);
  if (newSharding != sharding) {
    setShardingFn(newSharding);
  }
}

// Applies `callback` on the sharding of `valueOrFuncResult` if present, and
// replaces the sharding with the result.
void processSharding(const ValueOrFuncResult& valueOrFuncResult,
                     const TransformShardingForTensorFn& callback) {
  if (auto* value = std::get_if<Value>(&valueOrFuncResult)) {
    processSharding(getSharding(*value), valueOrFuncResult, callback,
                    [&](TensorShardingAttr newSharding) {
                      setSharding(*value, newSharding);
                    });
  } else {
    auto [funcOp, resNum] = std::get<FuncResult>(valueOrFuncResult);
    processSharding(
        getFuncResultSharding(funcOp, resNum), valueOrFuncResult, callback,
        [funcOp = funcOp, resNum = resNum](TensorShardingAttr newSharding) {
          setFuncResultSharding(funcOp, resNum, newSharding);
        });
  }
}

// Applies `callback` on each sharding in `shardings` if present, and
// corresponding value in `values`, and calls `setShardingsFn` on the results if
// any sharding changed.
void processShardings(
    ArrayRef<TensorShardingAttr> shardings, ValueRange values,
    const TransformShardingForTensorFn& callback,
    llvm::function_ref<void(ArrayRef<TensorShardingAttr>)> setShardingsFn) {
  if (shardings.empty()) {
    return;
  }
  // To handle the case where an op has a single-device sharding but returns
  // no values, we create a null value to pass to the callback. Note we need to
  // keep a stack variable of the value since `values` is a ValueRange which
  // only holds a reference.
  // TODO(b/391545244): do something smarter than sticking a null `Value` into
  // the callback. Not an issue now but can be in the future.
  Value emptySingleDeviceValue;
  if (values.empty()) {
    // This should be a single-device sharding.
    assert(shardings.size() == 1);
    values = emptySingleDeviceValue;
  }

  SmallVector<TensorShardingAttr> newShardings;
  newShardings.reserve(shardings.size());
  for (auto [sharding, value] : llvm::zip_equal(shardings, values)) {
    newShardings.push_back(callback(sharding, value));
  }
  if (newShardings != shardings) {
    setShardingsFn(newShardings);
  }
}

// Same as above but for `TensorShardingPerValueAttr`.
void processShardings(
    TensorShardingPerValueAttr shardings, ValueRange values,
    const TransformShardingForTensorFn& callback,
    llvm::function_ref<void(TensorShardingPerValueAttr)> setShardingsFn) {
  return processShardings(shardings.getShardings(), values, callback,
                          [&](ArrayRef<TensorShardingAttr> newShardings) {
                            setShardingsFn(TensorShardingPerValueAttr::get(
                                shardings.getContext(), newShardings));
                          });
}

TensorShardingAttr getValueOrFuncResultSharding(
    const ValueOrFuncResult& valueOrFuncResult) {
  if (const auto* value = std::get_if<Value>(&valueOrFuncResult)) {
    return getSharding(*value);
  }
  const auto& funcResult = std::get<FuncResult>(valueOrFuncResult);
  return getFuncResultSharding(funcResult.funcOp, funcResult.resNum);
}

}  // namespace

void transformSharding(const ValueOrFuncResult& valueOrFuncResult,
                       const TransformShardingFn& transformFn) {
  processSharding(
      valueOrFuncResult,
      [&transformFn](TensorShardingAttr newSharding, const ValueOrFuncResult&) {
        return transformFn(newSharding);
      });
}

void transformShardings(Operation* rootOp,
                        const TransformShardingForTensorFn& transformFn,
                        const ConsumeOpFn& consumeOpFn) {
  rootOp->walk<WalkOrder::PreOrder>([&](Operation* op) {
    consumeOpFn(op);
    TypeSwitch<Operation*, void>(op)
        .Case([&](FuncOp funcOp) {
          // TODO(b/511994873): Unify `transform` with `processSharding` apis.
          auto transform = [&](mlir::NamedAttrList& attrs,
                               const ValueOrFuncResult& valueOrFuncResult) {
            if (auto sharding =
                    getValueOrFuncResultSharding(valueOrFuncResult)) {
              TensorShardingAttr newSharding =
                  transformFn(sharding, valueOrFuncResult);
              if (newSharding != sharding) {
                attrs.set(kShardingAttr, newSharding);
                return true;
              }
            }
            return false;
          };
          modifyAllArgAttrs(
              funcOp, [&](mlir::NamedAttrList& attrs, int64_t argNum) {
                return transform(attrs, funcOp.getArgument(argNum));
              });
          modifyAllResultAttrs(
              funcOp, [&](mlir::NamedAttrList& attrs, int64_t resNum) {
                return transform(attrs, FuncResult(funcOp, resNum));
              });
        })
        .Case([&](ManualComputationOp manualComputationOp) {
          processShardings(
              manualComputationOp.getInShardings(),
              manualComputationOp.getBody().getArguments(), transformFn,
              [&](TensorShardingPerValueAttr newShardings) {
                manualComputationOp.setInShardingsAttr(newShardings);
              });
          processShardings(
              manualComputationOp.getOutShardings(),
              manualComputationOp.getResults(), transformFn,
              [&](TensorShardingPerValueAttr newShardings) {
                manualComputationOp.setOutShardingsAttr(newShardings);
              });
        })
        .Case([&](ShardableDataFlowOpInterface shardableDataFlowOp) {
          processShardings(
              shardableDataFlowOp.getBlockArgumentEdgeOwnerShardings(),
              shardableDataFlowOp.getBlockArgumentEdgeOwners(), transformFn,
              [&](ArrayRef<TensorShardingAttr> newShardings) {
                shardableDataFlowOp.setBlockArgumentEdgeOwnerShardings(
                    newShardings);
              });
          processShardings(
              shardableDataFlowOp.getOpResultEdgeOwnerShardings(),
              shardableDataFlowOp.getOpResultEdgeOwners(), transformFn,
              [&](ArrayRef<TensorShardingAttr> newShardings) {
                shardableDataFlowOp.setOpResultEdgeOwnerShardings(newShardings);
              });
        })
        .Default([&](Operation* op) {
          if (op->getNumResults() == 1) {
            // For ops with a single result, we use `get/setSharding` instead of
            // `get/setShardings`, since the latter only handle ops with an
            // unregistered sharding attribute, to also handle SDY ops like
            // `ShardingConstraintOp`.
            Value result = op->getResult(0);
            processSharding(result, transformFn);
          } else {
            processShardings(getShardings(op), op->getResults(), transformFn,
                             [&](ArrayRef<TensorShardingAttr> newShardings) {
                               setShardings(op, newShardings);
                             });
          }
        });
  });
}

void walkShardings(Operation* rootOp, const ConsumeShardingFn& consumeFn,
                   const ConsumeOpFn& consumeOpFn) {
  transformShardings(
      rootOp,
      [&consumeFn](TensorShardingAttr sharding, const ValueOrFuncResult&) {
        consumeFn(sharding);
        return sharding;
      },
      consumeOpFn);
}

void transformShardings(Operation* rootOp,
                        const TransformShardingFn& transformFn,
                        const ConsumeOpFn& consumeOpFn) {
  transformShardings(
      rootOp,
      [&transformFn](TensorShardingAttr newSharding, const ValueOrFuncResult&) {
        return transformFn(newSharding);
      },
      consumeOpFn);
}

}  // namespace sdy
}  // namespace mlir
