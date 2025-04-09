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
#include <functional>
#include <iterator>
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
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
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

MeshOp getMeshOp(Operation* op, SymbolRefAttr meshSymName) {
  return SymbolTable::lookupNearestSymbolFrom<sdy::MeshOp>(
      op, meshSymName);
}

MeshOp getMeshOp(Operation* op, StringRef meshName) {
  return getMeshOp(op, SymbolRefAttr::get(op->getContext(), meshName));
}

MeshOp getMeshOp(const SymbolTable& symbolTable, StringRef meshName) {
  return symbolTable.lookup<MeshOp>(meshName);
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
  if (MeshOp meshOp = getMeshOp(symbolTable, meshName)) {
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
  if (MeshOp meshOp = getMeshOp(op, meshSymName)) {
    return meshOp.getMesh();
  }

  return nullptr;
}

Attribute getCommonMeshOrRef(ArrayRef<TensorShardingAttr> operandShardings,
                             ArrayRef<TensorShardingAttr> resultsShardings,
                             const SymbolTable& symbolTable,
                             const bool ignoreDeviceIds) {
  Attribute meshOrRef;
  MeshAttr mesh;
  for (TensorShardingAttr sharding : llvm::concat<const TensorShardingAttr>(
           operandShardings, resultsShardings)) {
    if (!sharding) {
      continue;
    }
    MeshAttr otherMesh = sharding.getMesh(symbolTable);
    // TODO(b/409516201): Why do we check mesh.empty(), and what if it ends up
    // propagating to the sharding with the empty mesh.
    if (!mesh || mesh.empty()) {
      mesh = otherMesh;
      meshOrRef = sharding.getMeshOrRef();
      continue;
    }
    if (otherMesh.empty()) {
      continue;
    }
    if (!otherMesh.equals(mesh, ignoreDeviceIds)) {
      // Found more than one mesh name.
      return nullptr;
    }
    // Prefer iota device id over non-iota.
    if (ignoreDeviceIds && otherMesh.getDeviceIds().empty()) {
      mesh = otherMesh;
      meshOrRef = sharding.getMeshOrRef();
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
    const SymbolTable& symbolTable, const bool ignoreDeviceIds) {
  Attribute meshOrRef = getCommonMeshOrRef(operandShardings, resultsShardings,
                                           symbolTable, ignoreDeviceIds);

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
  if (auto op = DataFlowEdgeOp::lookup(value)) {
    return op.getResult();
  }

  if (isa<OpResult>(value)) {
    return value;
  }

  return TypeSwitch<Operation*, Value>(getOwningOp(value))
      .Case<FuncOp>([&](FuncOp) { return value; })
      .Case<ShardableDataFlowOpInterface>(
          [&](ShardableDataFlowOpInterface shardableRegionOp) {
            return shardableRegionOp.getEdgeOwnerFromTarget(value);
          })
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
        return shardingConstraint.getSharding();
      })
      .Case<ReshardOp>(
          [](ReshardOp reshardOp) { return reshardOp.getSharding(); })
      .Case<CollectiveOpInterface>([](CollectiveOpInterface collectiveOp) {
        return collectiveOp.getOutSharding();
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

TensorShardingAttr getOrCreateSharding(Value value, Attribute meshOrRef,
                                       const bool closedIfMissing) {
  if (TensorShardingAttr sharding = getSharding(value)) {
    return sharding;
  }
  return closedIfMissing
             ? TensorShardingAttr::getFullyClosed(
                   value.getContext(), getTensorRank(value), meshOrRef)
             : TensorShardingAttr::getFullyOpen(
                   value.getContext(), getTensorRank(value), meshOrRef);
}

TensorShardingAttr getOrCreateSharding(Value value, StringRef meshName,
                                       const bool closedIfMissing) {
  return getOrCreateSharding(
      value, FlatSymbolRefAttr::get(value.getContext(), meshName),
      closedIfMissing);
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
      .Case<CollectiveOpInterface>([&](CollectiveOpInterface collectiveOp) {
        collectiveOp.setOutShardingAttr(sharding);
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
    if (auto prefix = firstAxisRef.getGreatestCommonPrefix(secondAxisRef)) {
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

SmallVector<TensorShardingAttr> getFullyOpenShardings(MLIRContext* context,
                                                      TypeRange types,
                                                      StringRef meshName) {
  SmallVector<TensorShardingAttr> shardings;
  shardings.reserve(types.size());
  for (Type type : types) {
    int64_t rank = 0;
    // TODO(tomnatan): remove mlir:: once Attribute::dyn_cast is removed.
    if (auto tensorType = mlir::dyn_cast<ShapedType>(type)) {
      assert(tensorType.hasStaticShape());
      rank = tensorType.getRank();
    }
    shardings.push_back(
        TensorShardingAttr::getFullyOpen(context, rank, meshName));
  }
  return shardings;
}

SmallVector<TensorShardingAttr> getOpenShardingsWithShardingAtIndex(
    MLIRContext* context, TypeRange types, int64_t index,
    TensorShardingAttr sharding) {
  assert(index >= 0 && index < types.size());
  SmallVector<TensorShardingAttr> shardings =
      getFullyOpenShardings(context, types, sharding.getMeshName());
  shardings[index] = sharding;
  return shardings;
}

namespace {

// Callback that removes free (non-manual) axes from a
// `dimSharding` in a `ManualComputationOp` at `firstFreeAxisIndex`.
//
// Some use cases are removing all axes up to `firstFreeAxisIndex` or removing
// all axes from `firstFreeAxisIndex`. This needs to happen on many different
// `DimShardingAttr`s in the `in_shardings` and `out_shardings` of a
// `ManualComputationOp`.
using ManualComputationShardingEraserFn = std::function<DimensionShardingAttr(
    DimensionShardingAttr dimSharding, int64_t firstFreeAxisIndex)>;

// Calls a dimension sharding erasing callback on the first free axis in
// a dimension. This uses the invariant that shardings are prefixed with any
// manual axes.
TensorShardingAttr eraseAxesFromManualComputationSharding(
    TensorShardingAttr outerManualSharding, ArrayRef<StringAttr> manualAxes,
    ManualComputationShardingEraserFn shardingEraser) {
  SmallVector<DimensionShardingAttr> newDimShardings;
  newDimShardings.reserve(outerManualSharding.getRank());
  for (DimensionShardingAttr dimSharding :
       outerManualSharding.getDimShardings()) {
    ArrayRef<AxisRefAttr> dimAxes = dimSharding.getAxes();
    // Axes in the range [0, firstFreeAxis) are manual axes, and
    // [firstFreeAxis, dimAxes.size()) are free axes.
    llvm::ArrayRef<AxisRefAttr>::const_iterator firstFreeAxisIt =
        llvm::partition_point(dimAxes, [&manualAxes](AxisRefAttr axis) {
          return llvm::is_contained(manualAxes, axis.getName());
        });
    newDimShardings.push_back(
        shardingEraser(dimSharding, firstFreeAxisIt - dimAxes.begin()));
  }
  // Grab any replicated axes that are not manual axes. Can't use
  // `partition_point` as there is no defined order for replicated axes.
  SmallVector<AxisRefAttr> newReplicatedAxes;
  llvm::copy_if(outerManualSharding.getReplicatedAxes(),
                std::back_inserter(newReplicatedAxes), [&](AxisRefAttr axis) {
                  return !llvm::is_contained(manualAxes, axis.getName());
                });
  return TensorShardingAttr::get(outerManualSharding.getContext(),
                                 outerManualSharding.getMeshOrRef(),
                                 newDimShardings, newReplicatedAxes);
}

}  // namespace

TensorShardingAttr eraseManualAxes(TensorShardingAttr outerManualSharding,
                                   ArrayRef<StringAttr> manualAxes) {
  if (manualAxes.empty()) {
    return outerManualSharding;
  }
  return eraseAxesFromManualComputationSharding(
      outerManualSharding, manualAxes,
      std::mem_fn(&DimensionShardingAttr::dropFrontShardingAxes));
}

TensorShardingAttr eraseFreeAxes(TensorShardingAttr outerManualSharding,
                                 ArrayRef<StringAttr> manualAxes) {
  return eraseAxesFromManualComputationSharding(
      outerManualSharding, manualAxes,
      std::mem_fn(&DimensionShardingAttr::takeFrontShardingAxes));
}

}  // namespace sdy
}  // namespace mlir
