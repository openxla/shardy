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

#include "shardy/dialect/sdy/ir/utils.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/data_flow_utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

namespace {

using func::FuncOp;

TensorShardingPerValueAttr getOrCreateShardingPerResult(Operation* op,
                                                        StringRef meshName) {
  if (auto shardingPerResult =
          op->getAttrOfType<TensorShardingPerValueAttr>(kShardingAttr)) {
    return shardingPerResult;
  }

  return TensorShardingPerValueAttr::getFullyOpen(
      op->getContext(), op->getResultTypes(), meshName);
}

template <typename T>
std::string mlirToString(T* mlirVal) {
  std::string out;
  {
    llvm::raw_string_ostream os(out);
    mlirVal->print(os, OpPrintingFlags().enableDebugInfo());
  }
  return out;
}

}  // namespace

void emitOpWarningOnce(llvm::once_flag& flag, Operation* op, StringRef msg) {
  llvm::call_once(flag, [=]() {
    InFlightDiagnostic diag = emitWarning(op->getLoc(), msg);
    if (op->getContext()->shouldPrintOpOnDiagnostic()) {
      diag.attachNote().appendOp(*op, OpPrintingFlags().assumeVerified());
    }
  });
}

std::string attributeToString(Attribute attr) {
  std::string out;
  llvm::raw_string_ostream os(out);
  attr.print(os);
  return out;
}

std::string operationToString(Operation* op) { return mlirToString(op); }

std::string valueToString(Value value) { return mlirToString(&value); }

ShapedType dynCastStaticShapedType(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type);
      shapedType && shapedType.hasStaticShape()) {
    return shapedType;
  }
  return nullptr;
}

bool isStaticShapedType(Type type) {
  return dynCastStaticShapedType(type) != nullptr;
}

ArrayRef<int64_t> getTensorShape(Value value) {
  if (auto tensorType = dyn_cast<ShapedType>(value.getType())) {
    return tensorType.getShape();
  }
  return {};
}

int64_t getTensorRank(Value value) {
  if (auto tensorType = dyn_cast<ShapedType>(value.getType())) {
    return tensorType.getRank();
  }
  return 0;
}

int64_t isScalar(Value value) {
  if (auto tensorType = dyn_cast<ShapedType>(value.getType());
      tensorType && tensorType.hasRank()) {
    return tensorType.getRank() == 0;
  }
  return false;
}

MeshAttr getMeshAttr(const SymbolTable& symbolTable, StringRef meshName) {
  if (auto meshOp = symbolTable.lookup<MeshOp>(meshName)) {
    return meshOp.getMesh();
  }

  return nullptr;
}

MeshAttr getMeshAttr(const SymbolTable& symbolTable,
                     SymbolRefAttr meshSymName) {
  return getMeshAttr(symbolTable, meshSymName.getLeafReference());
}

MeshAttr getMeshAttr(Operation* op, StringRef meshName) {
  return getMeshAttr(op, SymbolRefAttr::get(op->getContext(), meshName));
}

MeshAttr getMeshAttr(Operation* op, SymbolRefAttr meshSymName) {
  if (auto meshOp =
          SymbolTable::lookupNearestSymbolFrom<MeshOp>(op, meshSymName)) {
    return meshOp.getMesh();
  }

  return nullptr;
}

std::string factorSymbolString(int64_t factor) {
  if (factor <= kStartAtZ) {
    return std::string(1, 'i' + factor);
  }
  return "z_" + std::to_string(factor - kStartAtZ);
}

Operation* getOwningOp(Value value) {
  if (Operation* op = value.getDefiningOp()) {
    return op;
  }
  // If there is no defining op, then that means the value is a block arg.
  // Get the op that owns this block.
  return value.getParentBlock()->getParentOp();
}

Value getShardableValue(Value value) {
  if (DataFlowEdgeOp op = getDataFlowEdge(value)) {
    return op.getResult();
  }

  if (isa<OpResult>(value)) {
    return value;
  }

  auto arg = cast<BlockArgument>(value);

  return TypeSwitch<Operation*, Value>(arg.getOwner()->getParentOp())
      .Case<ManualComputationOp, FuncOp, ShardableDataFlowOpInterface>(
          [&](Operation*) { return value; })
      .Default([&](Operation* op) {
        // We only fail if the value isn't scalar. Scalar block arguments, such
        // as the arguments of a reduction function, don't have a shardable
        // value. This is ok since they are scalars (rank 0) and therefore can't
        // be sharded.
        if (!isScalar(value)) {
          unreachableFormatv("region op '{0}' not supported", op->getName());
        }
        return nullptr;
      });
}

TensorShardingAttr getSharding(Value value) {
  value = getShardableValue(value);
  if (!value) {
    // This means the value is a scalar block argument, in which case it can't
    // be sharded.
    return TensorShardingAttr();
  }
  return TypeSwitch<Operation*, TensorShardingAttr>(getOwningOp(value))
      .Case<FuncOp>([value](FuncOp funcOp) {
        return funcOp.getArgAttrOfType<TensorShardingAttr>(
            cast<BlockArgument>(value).getArgNumber(), kShardingAttr);
      })
      .Case<DataFlowEdgeOp>([](DataFlowEdgeOp dataFlowEdgeOp) {
        return dataFlowEdgeOp.getShardingAttr();
      })
      .Case<ShardingConstraintOp>([](ShardingConstraintOp shardingConstraint) {
        return shardingConstraint.getShardingAttr();
      })
      .Case<ReshardOp>(
          [](ReshardOp reshardOp) { return reshardOp.getShardingAttr(); })
      .Case<ManualComputationOp>([&](ManualComputationOp manualComputationOp) {
        if (auto blockArg = dyn_cast<BlockArgument>(value)) {
          // Block arguments of a `ManualComputationOp` can only be referred to
          // inside the body. Remove any of the manual axes that are prefixed to
          // it so the body of the MC op doesn't know about them.
          return manualComputationOp.getInShardingWithoutManualAxes(
              blockArg.getArgNumber());
        }
        // An op outside of a `ManualComputationOp`, that is a user of the
        // `OpResult,` would request this value. As such keep the manual
        // axes as we can try propagating them.
        return manualComputationOp.getOutSharding(
            cast<OpResult>(value).getResultNumber());
      })
      // TODO: b/360076171 - Add tests for ShardableDataFlowOpInterface,
      // potentially with a test dialect.
      // TODO: b/360076171 - Move the cases to the interface.
      .Case<ShardableDataFlowOpInterface>([value](ShardableDataFlowOpInterface
                                                      shardableRegionOp) {
        if (auto blockArg = dyn_cast<BlockArgument>(value)) {
          return shardableRegionOp.getBlockArgumentEdgeOwnerSharding(
              blockArg.getArgNumber());
        }
        // TODO: b/360076171 - Add a `getOpResultEdgeOwnerSharding`.
        if (auto shardingPerResult =
                getOwningOp(value)->getAttrOfType<TensorShardingPerValueAttr>(
                    kShardingAttr)) {
          return shardingPerResult
              .getShardings()[cast<OpResult>(value).getResultNumber()];
        }
        return TensorShardingAttr();
      })
      .Default([value](Operation* op) {
        if (auto shardingPerResult =
                op->getAttrOfType<TensorShardingPerValueAttr>(kShardingAttr)) {
          return shardingPerResult
              .getShardings()[cast<OpResult>(value).getResultNumber()];
        }
        return TensorShardingAttr();
      });
}

TensorShardingAttr getOrCreateSharding(Value value, StringRef meshName) {
  if (TensorShardingAttr sharding = getSharding(value)) {
    return sharding;
  }

  return TensorShardingAttr::getFullyOpen(value.getContext(),
                                          getTensorRank(value), meshName);
}

void setSharding(Value value, TensorShardingAttr sharding) {
  value = getShardableValue(value);
  assert(value && "value should exist if its sharding is updated");
  TypeSwitch<Operation*>(getOwningOp(value))
      .Case<FuncOp>([&](FuncOp funcOp) {
        funcOp.setArgAttr(cast<BlockArgument>(value).getArgNumber(),
                          kShardingAttr, sharding);
      })
      .Case<DataFlowEdgeOp>([&](DataFlowEdgeOp dataFlowEdgeOp) {
        dataFlowEdgeOp.setShardingAttr(sharding);
      })
      .Case<ShardingConstraintOp>([&](ShardingConstraintOp shardingConstraint) {
        shardingConstraint.setShardingAttr(sharding);
      })
      .Case<ReshardOp>(
          [&](ReshardOp reshardOp) { reshardOp.setShardingAttr(sharding); })
      .Case<ManualComputationOp>([&](ManualComputationOp manualComputationOp) {
        if (auto blockArg = dyn_cast<BlockArgument>(value)) {
          // We only set `in_shardings` when propagating from a use inside
          // the body of the `ManualComputationOp` to the `in_shardings`, and
          // since propagation within the body of the op doesn't see the manual
          // axes, we need to add them back.
          manualComputationOp.setInShardingAddingManualAxes(
              blockArg.getArgNumber(), sharding);
        } else {
          // This would happen when an op outside of a `ManualComputationOp`
          // is a user of a result of the `ManualComputationOp`. In this case,
          // we don't need to add the manual axes as they were already seen by
          // the user.
          manualComputationOp.setOutSharding(
              cast<OpResult>(value).getResultNumber(), sharding);
        }
      })
      .Case<ShardableDataFlowOpInterface>(
          [&](ShardableDataFlowOpInterface shardableRegionOp) {
            if (auto blockArg = dyn_cast<BlockArgument>(value)) {
              shardableRegionOp.setBlockArgumentEdgeOwnerSharding(
                  blockArg.getArgNumber(), sharding);
            }
            // TODO: b/360076171 - Add a `setOpResultEdgeOwnerSharding`.
            if (auto opResult = dyn_cast<OpResult>(value)) {
              Operation* op = opResult.getOwner();
              op->setAttr(
                  kShardingAttr,
                  getOrCreateShardingPerResult(op, sharding.getMeshName())
                      .replaceValueSharding(
                          cast<OpResult>(value).getResultNumber(), sharding));
            }
          })
      .Default([&](Operation* op) {
        op->setAttr(kShardingAttr,
                    getOrCreateShardingPerResult(op, sharding.getMeshName())
                        .replaceValueSharding(
                            cast<OpResult>(value).getResultNumber(), sharding));
      });
}

SmallVector<TensorShardingAttr> getShardings(ValueRange values) {
  return llvm::to_vector(
      llvm::map_range(values, [](Value value) { return getSharding(value); }));
}

void removeShardingRules(Operation* rootOp) {
  rootOp->walk([](Operation* op) {
    if (auto shardingRule =
            op->getAttrOfType<OpShardingRuleAttr>(kShardingRuleAttr)) {
      if (!shardingRule.isCustom()) {
        op->removeAttr(kShardingRuleAttr);
      }
    }
  });
}

std::optional<StringRef> getCommonMeshName(
    ArrayRef<TensorShardingAttr> operandShardings,
    ArrayRef<TensorShardingAttr> resultsShardings) {
  StringRef meshName;
  for (TensorShardingAttr sharding : llvm::concat<const TensorShardingAttr>(
           operandShardings, resultsShardings)) {
    if (sharding) {
      if (meshName.empty()) {
        meshName = sharding.getMeshName();
      } else if (meshName != sharding.getMeshName()) {
        // Found more than one mesh name.
        return std::nullopt;
      }
    }
  }

  return meshName.empty() ? std::nullopt : std::make_optional(meshName);
}

ManualAxisToOwner getParentManualComputationOps(Operation* op) {
  ManualAxisToOwner alreadyManualAxes;
  auto parent = op->getParentOfType<ManualComputationOp>();
  while (parent) {
    for (StringRef axisName : parent.getManualAxes()) {
      alreadyManualAxes[axisName] = parent;
    }
    parent = parent->getParentOfType<ManualComputationOp>();
  }
  return alreadyManualAxes;
}

llvm::SmallDenseSet<StringRef> getParentManualAxes(Operation* op) {
  llvm::SmallDenseSet<StringRef> manualAxes;
  auto parent = op->getParentOfType<ManualComputationOp>();
  while (parent) {
    manualAxes.insert(parent.getManualAxes().begin(),
                      parent.getManualAxes().end());
    parent = parent->getParentOfType<ManualComputationOp>();
  }
  return manualAxes;
}

}  // namespace sdy
}  // namespace mlir
