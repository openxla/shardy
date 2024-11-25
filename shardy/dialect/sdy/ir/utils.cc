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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
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

void replaceShardingAtIndex(Operation* op, unsigned index,
                            TensorShardingAttr sharding) {
  if (TensorShardingPerValueAttr shardingPerResult = getShardingPerValue(op)) {
    setShardings(op, shardingPerResult.replaceValueSharding(index, sharding));
  } else {
    setShardings(op,
                 TensorShardingPerValueAttr::getOpenWithShardingAtIndex(
                     op->getContext(), op->getResultTypes(), index, sharding));
  }
}

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

MeshAttr getMeshOrLookup(const SymbolTable& symbolTable, Attribute meshOrRef) {
  if (auto mesh = dyn_cast<MeshAttr>(meshOrRef)) {
    return mesh;
  }
  return getMeshAttr(symbolTable, cast<FlatSymbolRefAttr>(meshOrRef));
}

MeshAttr getMeshOrLookup(Operation* op, Attribute meshOrRef) {
  if (auto mesh = dyn_cast<MeshAttr>(meshOrRef)) {
    return mesh;
  }
  return getMeshAttr(op, cast<FlatSymbolRefAttr>(meshOrRef));
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

Attribute getCommonMeshOrRef(ArrayRef<TensorShardingAttr> operandShardings,
                             ArrayRef<TensorShardingAttr> resultsShardings,
                             const SymbolTable& symbolTable) {
  Attribute meshOrRef;
  MeshAttr mesh;
  for (TensorShardingAttr sharding : llvm::concat<const TensorShardingAttr>(
           operandShardings, resultsShardings)) {
    if (!sharding) {
      continue;
    }
    MeshAttr otherMesh = sharding.getMesh(symbolTable);
    if (!mesh || mesh.empty()) {
      mesh = otherMesh;
      meshOrRef = sharding.getMeshOrRef();
    } else if (otherMesh != mesh && !otherMesh.empty()) {
      // Found more than one mesh name.
      return nullptr;
    }
  }

  return meshOrRef;
}

MeshAttr getCommonMesh(ArrayRef<TensorShardingAttr> operandShardings,
                       ArrayRef<TensorShardingAttr> resultsShardings,
                       const SymbolTable& symbolTable) {
  if (Attribute meshOrRef =
          getCommonMeshOrRef(operandShardings, resultsShardings, symbolTable)) {
    return getMeshOrLookup(symbolTable, meshOrRef);
  }
  return nullptr;
}

MeshAttr getCommonMesh(ArrayRef<TensorShardingAttr> operandShardings,
                       ArrayRef<TensorShardingAttr> resultsShardings,
                       Operation* op) {
  return getCommonMesh(operandShardings, resultsShardings,
                       SymbolTable(op->getParentOfType<ModuleOp>()));
}

std::optional<StringRef> getCommonMeshName(
    ArrayRef<TensorShardingAttr> operandShardings,
    ArrayRef<TensorShardingAttr> resultsShardings,
    const SymbolTable& symbolTable) {
  Attribute meshOrRef =
      getCommonMeshOrRef(operandShardings, resultsShardings, symbolTable);
  // We assume that if there is a common mesh, then there can only be a unique
  // symbol name referencing that mesh.
  return meshOrRef
             ? std::make_optional(cast<FlatSymbolRefAttr>(meshOrRef).getValue())
             : std::nullopt;
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

TensorShardingAttr getSharding(Value value, bool removeManualAxes) {
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
          if (removeManualAxes) {
            // Block arguments of a `ManualComputationOp` can only be referred
            // to inside the body. Remove any of the manual axes that are
            // prefixed to it so the body of the MC op doesn't know about them.
            return manualComputationOp.getInShardingWithoutManualAxes(
                blockArg.getArgNumber());
          }
          return manualComputationOp.getInSharding(blockArg.getArgNumber());
        }
        // An op outside of a `ManualComputationOp`, that is a user of the
        // `OpResult,` would request this value. As such keep the manual
        // axes as we can try propagating them.
        return manualComputationOp.getOutSharding(
            cast<OpResult>(value).getResultNumber());
      })
      // TODO: b/360076171 - Add tests for ShardableDataFlowOpInterface,
      // potentially with a test dialect.
      .Case<ShardableDataFlowOpInterface>(
          [value](ShardableDataFlowOpInterface shardableRegionOp) {
            return shardableRegionOp.getEdgeOwnerSharding(value);
          })
      .Default([value](Operation* op) {
        if (TensorShardingPerValueAttr shardingPerResult =
                getShardingPerValue(op)) {
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

void setSharding(Value value, TensorShardingAttr sharding,
                 bool addManualAxes) {
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
          if (addManualAxes) {
            // We only set `in_shardings` when propagating from a use inside
            // the body of the `ManualComputationOp` to the `in_shardings`, and
            // since propagation within the body of the op doesn't see the
            // manual axes, we need to add them back.
            manualComputationOp.setInShardingAddingManualAxes(
                blockArg.getArgNumber(), sharding);
          } else {
            manualComputationOp.setInSharding(blockArg.getArgNumber(),
                                              sharding);
          }
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
            shardableRegionOp.setEdgeOwnerSharding(value, sharding);
          })
      .Default([&](Operation* op) {
        replaceShardingAtIndex(op, cast<OpResult>(value).getResultNumber(),
                               sharding);
      });
}

TensorShardingAttr getFuncResultSharding(FuncOp funcOp, int64_t resNum) {
  return funcOp.getResultAttrOfType<TensorShardingAttr>(resNum, kShardingAttr);
}

void setFuncResultSharding(FuncOp funcOp, int64_t resNum,
                           TensorShardingAttr sharding) {
  funcOp.setResultAttr(resNum, kShardingAttr, sharding);
}

SmallVector<AxisRefAttr> getGreatestCommonPrefix(ArrayRef<AxisRefAttr> first,
                                                 ArrayRef<AxisRefAttr> second) {
  SmallVector<AxisRefAttr> result;
  for (auto [firstAxisRef, secondAxisRef] : llvm::zip(first, second)) {
    if (firstAxisRef == secondAxisRef) {
      result.push_back(firstAxisRef);
      continue;
    }
    if (auto prefix = firstAxisRef.getGreatestCommonPrefix(secondAxisRef);
        prefix) {
      result.push_back(*prefix);
    }
    break;
  }
  return result;
}

SmallVector<TensorShardingAttr> getShardings(ValueRange values) {
  return llvm::to_vector(
      llvm::map_range(values, [](Value value) { return getSharding(value); }));
}

ArrayRef<TensorShardingAttr> getShardings(Operation* op) {
  if (auto shardingPerResult = getShardingPerValue(op)) {
    return shardingPerResult.getShardings();
  }
  return {};
}

TensorShardingPerValueAttr getShardingPerValue(Operation* op) {
  return op->getAttrOfType<TensorShardingPerValueAttr>(kShardingAttr);
}

RankedTensorType getLocalTypeFromValue(Value value) {
  RankedTensorType globalTensorType =
      dyn_cast<RankedTensorType>(value.getType());
  assert(globalTensorType && getShardableValue(value));
  TensorShardingAttr sharding =
      TypeSwitch<Operation*, TensorShardingAttr>(getOwningOp(value))
          .Case<FuncOp>([value](FuncOp funcOp) {
            return funcOp.getArgAttrOfType<TensorShardingAttr>(
                cast<BlockArgument>(value).getArgNumber(), kShardingAttr);
          })
          .Case<ReshardOp>(
              [](ReshardOp reshardOp) { return reshardOp.getShardingAttr(); })
          .Case<ManualComputationOp>(
              [&](ManualComputationOp manualComputationOp) {
                if (auto blockArg = dyn_cast<BlockArgument>(value)) {
                  return manualComputationOp.getInSharding(
                      blockArg.getArgNumber());
                }
                return manualComputationOp.getOutSharding(
                    cast<OpResult>(value).getResultNumber());
              })
          .Default([value](Operation* op) {
            if (TensorShardingPerValueAttr shardingPerResult =
                    getShardingPerValue(op)) {
              return shardingPerResult
                  .getShardings()[cast<OpResult>(value).getResultNumber()];
            }
            return TensorShardingAttr();
          });
  if (!sharding) {
    return globalTensorType;
  }
  RankedTensorType localType = sharding.getLocalTensorType(
      globalTensorType, sharding.getMesh(getOwningOp(value)));
  return localType;
}

void setShardings(Operation* op, ArrayRef<TensorShardingAttr> shardings) {
  if (shardings.empty()) {
    return;
  }
  setShardings(op,
               TensorShardingPerValueAttr::get(op->getContext(), shardings));
}

void setShardings(Operation* op, TensorShardingPerValueAttr shardingPerValue) {
  op->setAttr(kShardingAttr, shardingPerValue);
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

}  // namespace sdy
}  // namespace mlir
