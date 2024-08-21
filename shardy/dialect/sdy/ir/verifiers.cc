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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

namespace {

using func::FuncOp;
using ::llvm::SmallDenseMap;
using ::llvm::SmallDenseSet;

// Mapping between an axis name to the ManualComputationOp whose body is manual
// on.
using ManualAxisToOwner = SmallDenseMap<StringRef, ManualComputationOp>;

using EmitErrorFn = std::function<InFlightDiagnostic(StringRef)>;

EmitErrorFn getEmitErrorFn(Operation* op) {
  return [op](StringRef msg) { return op->emitOpError(msg); };
}

EmitErrorFn getEmitValueInRangeErrorFn(EmitErrorFn emitError, int64_t numValues,
                                       int64_t index) {
  return [=](StringRef msg) {
    if (numValues == 1) {
      return emitError("- ") << msg;
    }
    return emitError("") << index << " - " << msg;
  };
}

// Verifies the following for `axisRefs`:
//
// - All axis names are present in `axisNameToSize`.
// - There are no duplicate axis-refs, or an axis-ref that is already present in
//   `seenAxisRefs`.
// - No two adjacent axis-refs are consecutive sub-axes of that same full axis,
//   i.e., they can be merged into one sub-axis or the full axis.
LogicalResult verifyAxisRefList(
    ArrayRef<AxisRefAttr> axisRefs,
    const SmallDenseMap<StringRef, int64_t>& axisNameToSize,
    SmallDenseSet<AxisRefAttr>& seenAxisRefs,
    SmallDenseMap<StringRef, SmallVector<AxisRefAttr>>& axisNameToSubAxes,
    EmitErrorFn emitError) {
  AxisRefAttr prevSubAxisRef;
  for (AxisRefAttr axisRef : axisRefs) {
    StringRef axisName = axisRef.getName();
    if (!axisNameToSize.contains(axisName)) {
      return emitError("unknown axis name: \"") << axisName << "\"";
    }
    if (!seenAxisRefs.insert(axisRef).second) {
      return emitError("duplicate axis ref: ") << axisRef.toString();
    }
    if (!axisRef.getSubAxisInfo()) {
      prevSubAxisRef = axisRef;
      continue;
    }

    axisNameToSubAxes[axisName].push_back(axisRef);
    // Note that we ignore invalid cases where two sub-axes overlap, since
    // it will be checked below.
    if (prevSubAxisRef && prevSubAxisRef.canMerge(axisRef)) {
      return emitError("two consecutive sub-axes can be merged: ")
             << prevSubAxisRef.toString() << ", " << axisRef.toString();
    }
    prevSubAxisRef = axisRef;
  }

  return success();
}

// Verifies the following for each sub-axis in `subAxes`:
//
// - Its pre-size is at least 1.
// - Its size is greater than 1.
// - Its next pre-size must divide the size of the full axis, i.e., both its
//   pre-size and size divide the size of the full axis, and the sub-axis
//   doesn't go beyond the full axis.
// - The size of the sub-axis isn't equal to the size of the full axis, in which
//   case the full axis should be used instead.
// - It doesn't overlap with any other sub-axis or the full axis, if it's
//   present in `seenAxisRefs`.
//
// NOTE: this method assumes `subAxes` is sorted.
LogicalResult verifySubAxes(ArrayRef<AxisRefAttr> subAxes, StringRef axisName,
                            int64_t axisSize, MeshAttr mesh,
                            const SmallDenseSet<AxisRefAttr>& seenAxisRefs,
                            EmitErrorFn emitError) {
  if (!subAxes.empty() &&
      seenAxisRefs.contains(AxisRefAttr::get(mesh.getContext(), axisName))) {
    return emitError("both sub-axis and full-axis are used for axis name: \"")
           << axisName << "\"";
  }

  AxisRefAttr prevSubAxis;
  for (AxisRefAttr subAxis : subAxes) {
    SubAxisInfoAttr subAxisInfo = subAxis.getSubAxisInfo();
    if (subAxisInfo.getPreSize() < 1) {
      return emitError("sub-axis pre-size must be at least 1: ")
             << subAxis.toString();
    }
    if (subAxisInfo.getSize() <= 1) {
      return emitError("sub-axis sizes must be greater than 1: ")
             << subAxis.toString();
    }
    if (axisSize % subAxisInfo.getNextPreSize() != 0) {
      return emitError("sub-axis next pre-size ")
             << subAxisInfo.getNextPreSize()
             << " doesn't divide the size of the full "
                "axis "
             << axisSize << ": " << subAxis.toString();
    }
    if (subAxisInfo.getSize() == axisSize) {
      return emitError("sub-axis size is equal to the full axis size: ")
             << subAxis.toString();
    }

    int64_t prevSubAxisNextPreSize =
        prevSubAxis ? prevSubAxis.getSubAxisInfo().getNextPreSize() : 1;

    if (prevSubAxisNextPreSize > subAxisInfo.getPreSize()) {
      return emitError("overlapping sub-axes: ")
             << prevSubAxis.toString() << ", " << subAxis.toString();
    }
    prevSubAxis = subAxis;
  }

  return success();
}

// ManualComputations op are allowed to be nested within each other. However,
// they cannot operate on the same manual axes. This function creates a mapping
// from a manual mesh axis name to the corresponding ManualComputationOp that
// operates on it to help with verifying this is the case.
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

LogicalResult emitBoundAxisInManualComputationError(EmitErrorFn emitError,
                                                    StringRef boundAxis,
                                                    Location parentLoc) {
  return (emitError("operates on axis \"")
          << boundAxis
          << "\" which is already bound by a parent sdy.manual_computation "
             "op")
             .attachNote(parentLoc)
         << "parent bounding this axis as manual";
}

// Verifies the following for `shardingAttr`:
//
// If `type` isn't a `ShapedType`, the sharding must have rank 0 and no
// replicated axes.
//
// - The tensor should have a rank and static shape.
// - The number of dimension shardings is equal to the rank of the tensor.
// - Dimensions of size 0 aren't sharded.
// - Replicated axes are ordered w.r.t. `mesh` (see
//   AxisRefAttr::getMeshComparator).
// - All dimension shardings and the replicated axes are each a valid axis-ref
//   list (see `verifyAxisRefList`).
// - All sub-axes in `shardingAttr` (see `verifySubAxes`).
// - If `alreadyManualAxes` is not empty, then it verifies that when
//   shardingAttr is inside a ManualComputationOp (possibly
//   nested), then it only operates on axes not already marked as manual.
// - If a dimension sharding has a priority:
//     -- The priority is greater than or equal to 0.
//     -- The dimension has at least one axis if it is closed.
// - If `checkDivisibility` is true, verifies that each dimension size
//   is divisible by its sharded size.
LogicalResult verifyTensorShardingAttr(TensorShardingAttr shardingAttr,
                                       Type type, MeshAttr mesh,
                                       EmitErrorFn emitError,
                                       bool checkDivisibility,
                                       ManualAxisToOwner alreadyManualAxes) {
  auto tensorType = dyn_cast<ShapedType>(type);
  if (!tensorType) {
    if (shardingAttr.getRank() != 0 ||
        !shardingAttr.getReplicatedAxes().empty()) {
      return emitError(
                 "non-shaped tensors can only have a sharding with rank 0 ")
             << "and no replicated axes. type: " << type
             << ", sharding: " << shardingAttr;
    }
    return success();
  }
  if (!tensorType.hasStaticShape()) {
    return emitError(
               "only ranked tensors with a static shape can have a sharding. ")
           << "type: " << type;
  }
  int64_t rank = tensorType.getRank();
  if (shardingAttr.getRank() != rank) {
    return emitError("sharding doesn't match tensor rank: ")
           << shardingAttr.getRank() << " != " << rank;
  }

  SmallDenseMap<StringRef, int64_t> axisNameToSize;
  axisNameToSize.reserve(mesh.getAxes().size());
  for (MeshAxisAttr axis : mesh.getAxes()) {
    axisNameToSize[axis.getName()] = axis.getSize();
  }

  // Verify dimension shardings
  SmallDenseSet<AxisRefAttr> seenAxisRefs;
  SmallDenseMap<StringRef, SmallVector<AxisRefAttr>> axisNameToSubAxes;
  for (auto [dim, dimShardingAndSize] : llvm::enumerate(llvm::zip_equal(
           shardingAttr.getDimShardings(), tensorType.getShape()))) {
    auto [dimSharding, dimSize] = dimShardingAndSize;
    if (dimSharding.getPriority()) {
      if (dimSharding.getPriority().value() < 0) {
        return emitError("dim ") << dim << " has a negative priority";
      }
      if (dimSharding.getIsClosed() && dimSharding.emptyAxes()) {
        return emitError("dim ")
               << dim << " is empty and closed but has a priority";
      }
    }
    if (failed(verifyAxisRefList(dimSharding.getAxes(), axisNameToSize,
                                 seenAxisRefs, axisNameToSubAxes, emitError))) {
      return failure();
    }
    if (dimSize == 0 && !dimSharding.emptyAxes()) {
      return emitError("dim ") << dim << " of size 0 is sharded";
    }
    if (checkDivisibility) {
      int64_t shardedSize = dimSharding.getShardedSize(mesh);
      if (dimSize % shardedSize != 0) {
        return emitError("dim ")
               << dim << " with size " << dimSize
               << " is not divisible by its sharded size " << shardedSize;
      }
    }
  }

  // Verify replicated axes
  auto axisRefComparator = AxisRefAttr::getMeshComparator(mesh);
  ArrayRef<AxisRefAttr> replicatedAxes = shardingAttr.getReplicatedAxes();
  if (!llvm::is_sorted(replicatedAxes, axisRefComparator)) {
    return emitError("replicated axes are not ordered w.r.t. mesh");
  }
  if (failed(verifyAxisRefList(replicatedAxes, axisNameToSize, seenAxisRefs,
                               axisNameToSubAxes, emitError))) {
    return failure();
  }

  // Verify all sub-axes are valid.
  for (auto& [axisName, subAxes] : axisNameToSubAxes) {
    int64_t axisSize = axisNameToSize[axisName];
    // We need to sort the sub-axes since this is assumed by `verifySubAxes`.
    llvm::sort(subAxes, axisRefComparator);
    if (failed(verifySubAxes(subAxes, axisName, axisSize, mesh, seenAxisRefs,
                             emitError))) {
      return failure();
    }
  }

  // Verify all sharding and replicated axes don't already exist as a manual
  // axis due to a parent ManualComputationOp.
  if (!alreadyManualAxes.empty()) {
    if (shardingAttr.anyOfAxisRef([&](AxisRefAttr axis) {
          auto manualAxisIt = alreadyManualAxes.find(axis.getName());
          if (manualAxisIt != alreadyManualAxes.end()) {
            return emitBoundAxisInManualComputationError(
                       emitError, axis.getName(),
                       manualAxisIt->second->getLoc())
                .failed();
          }
          return false;
        })) {
      return failure();
    }
  }

  return success();
}

// Same as the overload above, but in addition verifies that the mesh name in
// `shardingAttr` is present in the module's symbol table.
LogicalResult verifyTensorShardingAttr(TensorShardingAttr shardingAttr,
                                       Type type, Operation* op,
                                       EmitErrorFn emitError) {
  SymbolRefAttr meshName = shardingAttr.getMeshSymName();
  MeshAttr mesh = getMeshAttr(op, meshName);
  if (!mesh) {
    return emitError("unknown mesh: ") << meshName;
  }

  return verifyTensorShardingAttr(shardingAttr, type, mesh, emitError,
                                  /*checkDivisibility=*/false,
                                  getParentManualComputationOps(op));
}

// Verifies the following for `shardingPerValueAttr`:
//
// - The number of tensor shardings is equal to the number of tensors (size of
//   `types`).
// - All shardings reference the same mesh name, and that name is present in the
//   module's symbol table.
// - All shardings are valid (see `verifyTensorShardingAttr`).
LogicalResult verifyTensorShardingPerValueAttr(
    TensorShardingPerValueAttr shardingPerValueAttr, TypeRange types,
    Operation* op, EmitErrorFn emitError, SymbolRefAttr meshName) {
  ArrayRef<TensorShardingAttr> shardingsPerValue =
      shardingPerValueAttr.getShardings();
  if (shardingsPerValue.size() != types.size()) {
    return emitError("shardings don't match number of values: ")
           << shardingsPerValue.size() << " shardings" << " vs " << types.size()
           << " values";
  }

  MeshAttr mesh;
  for (auto [index, entry] :
       llvm::enumerate(llvm::zip(shardingsPerValue, types))) {
    EmitErrorFn valueEmitError =
        getEmitValueInRangeErrorFn(emitError, types.size(), index);
    auto [shardingAttr, resultType] = entry;
    SymbolRefAttr otherMeshName = shardingAttr.getMeshSymName();
    if (!meshName) {
      meshName = otherMeshName;
    } else if (meshName != otherMeshName) {
      return emitError("shardings can only refer to one mesh: ")
             << otherMeshName << " vs " << meshName;
    }

    if (!mesh) {
      mesh = getMeshAttr(op, meshName);
      if (!mesh) {
        return valueEmitError("unknown mesh: ") << meshName;
      }
    }

    if (failed(verifyTensorShardingAttr(
            shardingAttr, resultType, mesh, valueEmitError,
            /*checkDivisibility=*/false, getParentManualComputationOps(op)))) {
      return failure();
    }
  }

  return success();
}

// Verifies an attribute of either a function argument or result.
LogicalResult verifyFuncAttribute(FuncOp funcOp, NamedAttribute attr, Type type,
                                  int64_t index, StringRef valueKindStr) {
  EmitErrorFn emitError = [=](StringRef msg) {
    return funcOp->emitOpError(valueKindStr) << " " << index << " - " << msg;
  };
  if (attr.getName() == kShardingAttr) {
    auto sharding = dyn_cast<TensorShardingAttr>(attr.getValue());
    if (!sharding) {
      return emitError(
          "should have a sharding attribute of type TensorShardingAttr");
    }
    return verifyTensorShardingAttr(sharding, type, funcOp, emitError);
  }
  return success();
}

// Verifies the following for `mappings`:
//
// - Number of mappings and types match.
// - There is at least one mapping (can't have a rule for an op with no
//   operands/results).
// - Rank of each `TensorMappingAttr` matches the rank of the corresponding
//   tensor type.
// - No duplicate factor indices in each `TensorMappingAttr`.
// - Per `DimMappingAttr`:
//   * There is at least one factor index.
//   * Factor indices must be in range [0, `factorSizes.size()`).
//   * If there are multiple factors, none of them can have size 1.
LogicalResult verifyShardingRuleMapping(Operation* op, TypeRange types,
                                        ArrayRef<TensorMappingAttr> mappings,
                                        ArrayRef<int64_t> factorSizes,
                                        BitVector& seenFactorIndices,
                                        StringRef valueKindStr) {
  if (types.size() != mappings.size()) {
    return op->emitOpError("number of ")
           << valueKindStr << "s and mappings must match: " << types.size()
           << " != " << mappings.size();
  }

  if (mappings.empty()) {
    return op->emitOpError("number of ")
           << valueKindStr
           << "s mappings cannot be 0. Op sharding rules can only be defined "
              "on operations with at least one operand and result.";
  }
  for (auto [index, typeAndMapping] :
       llvm::enumerate(llvm::zip_equal(types, mappings))) {
    // `seenFactorIndices` helps makes sure we use all factors, while
    // `valueSeenFactorIndices` helps make sure that a specific operand/result
    // doesn't reuse the same factor.
    BitVector valueSeenFactorIndices(factorSizes.size());
    auto [type, mapping] = typeAndMapping;

    EmitErrorFn valueEmitError = getEmitValueInRangeErrorFn(
        [op, valueKindStr](StringRef msg) {
          return op->emitOpError(valueKindStr) << " " << msg;
        },
        types.size(), index);

    auto tensorType = dynCastStaticShapedType(type);
    if (!tensorType) {
      return valueEmitError(
                 "expected a ranked tensor with a static shape. type: ")
             << type;
    }

    if (mapping.getRank() != tensorType.getRank()) {
      return valueEmitError("mapping rank must match: ")
             << mapping.getRank() << " != " << tensorType.getRank();
    }
    for (auto [dimSize, dimMapping] :
         llvm::zip_equal(tensorType.getShape(), mapping.getDimMappings())) {
      ArrayRef<int64_t> factorIndices = dimMapping.getFactorIndices();

      if (factorIndices.empty()) {
        return valueEmitError("dim mapping must have at least one factor");
      }

      int64_t totalFactorSize = 1;
      for (int64_t factorIndex : factorIndices) {
        if (factorIndex < 0 || factorIndex >= factorSizes.size()) {
          return valueEmitError(
                     "expecting factor indices to be within "
                     "0<=...<num_factors; received: ")
                 << factorIndex << ", num_factors: " << factorSizes.size();
        }
        if (valueSeenFactorIndices.test(factorIndex)) {
          return valueEmitError(
              "cannot reuse factors for the same tensor value");
        }
        int64_t factorSize = factorSizes[factorIndex];
        if (factorSize == 1 && factorIndices.size() > 1) {
          return valueEmitError(
              "dim mapping can't have a factor of size 1 if there are "
              "multiple factors");
        }
        totalFactorSize *= factorSize;
        valueSeenFactorIndices.set(factorIndex);
        seenFactorIndices.set(factorIndex);
      }
    }
  }

  return success();
}

// Verifies the following for an `OpShardingRuleAttr`:
//
// - If the rule is custom, the operation the rule is attached to is a
//   `CustomCallOp`.
// - All defined factor sizes are used by at least one operand/result mapping.
// - All operand/result mappings are valid (see `verifyShardingRuleMapping`).
LogicalResult verifyOpShardingRuleAttr(OpShardingRuleAttr shardingRule,
                                       Operation* op) {
  if (shardingRule.isCustom() && !isa<stablehlo::CustomCallOp>(op)) {
    return op->emitOpError(
        "can only define custom sharding rules on stablehlo.custom_call");
  }

  BitVector seenFactorIndices(shardingRule.getNumFactors());
  ArrayRef<int64_t> factorSizes = shardingRule.getFactorSizes();
  if (failed(verifyShardingRuleMapping(
          op, op->getOperandTypes(), shardingRule.getOperandMappings(),
          factorSizes, seenFactorIndices, "operand"))) {
    return failure();
  }
  if (failed(verifyShardingRuleMapping(
          op, op->getResultTypes(), shardingRule.getResultMappings(),
          factorSizes, seenFactorIndices, "result"))) {
    return failure();
  }

  if (!seenFactorIndices.all()) {
    int unsetIndex = seenFactorIndices.find_first_unset();
    return op->emitOpError("has factor ")
           << factorSymbolString(unsetIndex) << "=" << factorSizes[unsetIndex]
           << " that isn't used in operand and result mappings";
  }

  return success();
}

}  // namespace

LogicalResult MeshAxisAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, StringRef name,
    int64_t size) {
  if (size <= 0) {
    return emitError() << "axis size must be at least 1, got: " << size;
  }
  return success();
}

LogicalResult MeshAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<MeshAxisAttr> axes, ArrayRef<int64_t> deviceIds) {
  SmallDenseSet<StringRef> seenAxisNames;
  for (MeshAxisAttr axis : axes) {
    if (!seenAxisNames.insert(axis.getName()).second) {
      return emitError() << "duplicate axis name: \"" << axis.getName() << "\"";
    }
  }

  if (deviceIds.empty()) {
    // The default device ids are iota(product(axes)).
    return success();
  }

  for (int64_t deviceId : deviceIds) {
    if (deviceId < 0) {
      return emitError() << "device id must be non-negative, got: " << deviceId;
    }
  }

  // Verify that the product of axis sizes matches the number of explicit device
  // ids.
  int64_t totalDeviceIds = deviceIds.size();
  int64_t totalProductOfAxes = std::accumulate(
      axes.begin(), axes.end(), 1,
      [](int64_t cur, MeshAxisAttr axis) { return cur * axis.getSize(); });

  if (totalProductOfAxes != totalDeviceIds) {
    return emitError() << "total product of axis sizes must match total number "
                          "of device ids, got: "
                       << totalProductOfAxes << " != " << totalDeviceIds;
  }

  if (axes.empty()) {
    return success();
  }

  std::vector<int64_t> sortedDeviceIds(deviceIds.begin(), deviceIds.end());
  llvm::sort(sortedDeviceIds);
  if (!llvm::equal(sortedDeviceIds, llvm::seq<int64_t>(0, totalDeviceIds))) {
    return emitError() << "sorted device ids must be iota(product(axes)), got: "
                       << sortedDeviceIds;
  }

  if (llvm::is_sorted(deviceIds)) {
    return emitError() << "if the ordered device ids are the same as "
                          "iota(product(axes)), no need to specify them for "
                          "simplicity";
  }

  return success();
}

LogicalResult MeshOp::verify() {
  if (!SymbolTable::lookupNearestSymbolFrom(*this, getSymNameAttr())) {
    return emitError() << "Mesh not in symbol table: @" << getSymName();
  }
  // No need to check for duplicate mesh names as this will be verified by the
  // symbol table.
  return success();
}

LogicalResult ShardingConstraintOp::verify() {
  return verifyTensorShardingAttr(getSharding(), getType(), *this,
                                  getEmitErrorFn(*this));
}

LogicalResult ReshardOp::verify() {
  return verifyTensorShardingAttr(getSharding(), getType(), *this,
                                  getEmitErrorFn(*this));
}

LogicalResult DataFlowEdgeOp::verify() {
  if (!getType().hasStaticShape()) {
    return emitOpError(
               "expected sdy.data_flow_edge to have a static-shaped result. ")
           << "type: " << getType();
  }
  if (!getInput().hasOneUse()) {
    return emitOpError(
        "expected input of sdy.data_flow_edge to have a single user");
  }
  if (Operation* depOp = getInput().getDefiningOp();
      depOp && inDialect<SdyDialect>(depOp)) {
    return emitOpError(
               "expected input of sdy.data_flow_edge to not be defined by an "
               "SdyDialect op")
               .attachNote(depOp->getLoc())
           << "sdy op defining the input of the sdy.data_flow_edge";
  }
  if (std::optional<TensorShardingAttr> sharding = getSharding();
      sharding && failed(verifyTensorShardingAttr(*sharding, getType(), *this,
                                                  getEmitErrorFn(*this)))) {
    return failure();
  }
  return success();
}

namespace {

// Returns the accumulated axes size of a tensor sharding with respect to manual
// axes.
// If an axis in dimShardingAxes belongs to manualAxes, it's an axis
// the user is doing a manual computation on, thus the ManualComputationOp's
// body will have tensors smaller wrt this manual axis.
int64_t accumulatedManualAxesSize(Operation* op,
                                  ArrayRef<AxisRefAttr> dimShardingAxes,
                                  ArrayRef<StringAttr> manualAxes,
                                  MeshAttr mesh) {
  int64_t axesFactor = 1;
  for (AxisRefAttr axisRef : dimShardingAxes) {
    if (std::find(manualAxes.begin(), manualAxes.end(), axisRef.getName()) !=
        manualAxes.end()) {
      axesFactor *= axisRef.getSize(mesh);
    }
  }

  return axesFactor;
}

// Returns an iterator to a manual axis that comes after a free axis in `axes`.
// If no such axis exists, returns `axes.end()`.
ArrayRef<AxisRefAttr>::iterator findManualAxisAfterFreeAxis(
    ArrayRef<AxisRefAttr> axes, ArrayRef<StringAttr> manualAxes) {
  auto isManualAxis = [&](AxisRefAttr axisRef) {
    return llvm::is_contained(manualAxes, axisRef.getName());
  };
  return std::find_if(llvm::find_if_not(axes, isManualAxis), axes.end(),
                      isManualAxis);
}

// For each set of op values (operands/results) and corresponding sharding for
// each value, verifies:
// 1. the sharding list itself wrt the mesh and `globalTypes`,
// 2. each sharding is bound to the same 'meshName',
// 3. the in/out shardings are valid w.r.t the corresponding global
//    type,
// 4. the number of global and local tensor inputs/outputs of the op region
//    match,
// 5. the manual axes come before any free axes in each dim sharding,
// 6. the global shape and local shapes of the op regions arguments/results
//    match, and
// 7. all manual axes are used in the value's sharding.
//
// `valueKindStr` is a string included in any verification error message
// specifying whether the values we are verifying are the operands or results.
// Pass in "operand" or "result".
template <typename GlobalRangeT, typename LocalRangeT>
LogicalResult verifyManualComputationValue(
    ManualComputationOp op, GlobalRangeT globalTypes, LocalRangeT localTypes,
    TensorShardingPerValueAttr shardingPerValueAttr, SymbolRefAttr meshName,
    ArrayRef<StringAttr> manualAxes, StringRef valueKindStr) {
  if (globalTypes.empty() && localTypes.empty() &&
      shardingPerValueAttr.getShardings().empty()) {
    // Nothing to verify since there are no values.
    return success();
  }
  // 1. Verify the sharding list itself wrt the mesh and `globalTypes`.
  // 2. Verify each sharding is bound to the same 'meshName'.
  // 3. Verify the in/out shardings are valid w.r.t the corresponding global
  //    type.
  if (failed(verifyTensorShardingPerValueAttr(
          shardingPerValueAttr, globalTypes, op,
          [op, valueKindStr](StringRef msg) {
            return op->emitOpError(valueKindStr) << " " << msg;
          },
          meshName))) {
    return failure();
  }

  // 4. Verify the number of global and local tensor inputs/outputs of the op
  //    region match.
  if (globalTypes.size() != localTypes.size()) {
    return op->emitOpError("number of op ")
           << valueKindStr << "s and region " << valueKindStr
           << "s must match. Op has " << globalTypes.size() << " op "
           << valueKindStr << "s and " << localTypes.size() << " region "
           << valueKindStr << "s";
  }

  MeshAttr mesh = getMeshAttr(op, meshName);
  for (auto [valueIndex, valueEntry] : llvm::enumerate(llvm::zip_equal(
           globalTypes, localTypes, shardingPerValueAttr.getShardings()))) {
    auto [globalType, localType, sharding] = valueEntry;
    auto globalRankedType = cast<RankedTensorType>(globalType);
    auto localRankedType = cast<RankedTensorType>(localType);

    // 5. Verify the manual axes come before any free axes in each dim sharding.
    for (auto [dim, dimSharding] :
         llvm::enumerate(sharding.getDimShardings())) {
      ArrayRef<AxisRefAttr> axes = dimSharding.getAxes();
      if (ArrayRef<AxisRefAttr>::iterator manualAxisItr =
              findManualAxisAfterFreeAxis(axes, manualAxes);
          manualAxisItr != axes.end()) {
        return op->emitOpError(valueKindStr)
               << " sharding at index " << valueIndex
               << " must have all manual axes come before free axes in its "
               << "dimension sharding at index " << dim << ". Saw manual axis "
               << manualAxisItr->toString() << " after free axis "
               << (manualAxisItr - 1)->toString();
      }
    }

    // 6. Verify the global shape and local shapes of the op regions
    //    arguments/results match.
    SmallVector<int64_t> newDimSizes;
    for (auto [dimensionSize, dimSharding] : llvm::zip_equal(
             globalRankedType.getShape(), sharding.getDimShardings())) {
      newDimSizes.push_back(dimensionSize /
                            accumulatedManualAxesSize(op, dimSharding.getAxes(),
                                                      manualAxes, mesh));
    }
    auto expectedLocalRankedType =
        RankedTensorType::get(newDimSizes, globalRankedType.getElementType());
    if (expectedLocalRankedType != localRankedType) {
      return op->emitOpError(valueKindStr)
             << " shape, corresponding sharding, and region " << valueKindStr
             << " shape at index " << valueIndex
             << " must match. Expected local shape " << expectedLocalRankedType
             << ", actual local shape " << localRankedType;
    }

    auto emitManualAxesError = [&, valueIndex = valueIndex](
                                   AxisRefAttr axis, StringRef manualAxisName) {
      return op->emitOpError(valueKindStr)
             << " sharding at index " << valueIndex
             << " cannot refer to the sub/split axes " << axis.toString()
             << " as the axis \"" << manualAxisName << "\" is a manual axis";
    };

    // 7. Verify every manual axis is used in each sharding.
    for (StringRef manualAxisName : manualAxes) {
      bool foundManualAxis = false;
      // First check replicated axes
      for (AxisRefAttr axis : sharding.getReplicatedAxes()) {
        if (axis.getName() == manualAxisName) {
          if (axis.getSubAxisInfo()) {
            return emitManualAxesError(axis, manualAxisName);
          }
          foundManualAxis = true;
          break;
        }
      }
      if (foundManualAxis) {
        continue;
      }
      // Not in replicated axes. Now check dim shardings.
      for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
        for (AxisRefAttr axis : dimSharding.getAxes()) {
          if (axis.getName() == manualAxisName) {
            if (axis.getSubAxisInfo()) {
              return emitManualAxesError(axis, manualAxisName);
            }
            foundManualAxis = true;
            break;
          }
        }
        if (foundManualAxis) {
          break;
        }
      }
      if (!foundManualAxis) {
        return op->emitOpError(valueKindStr)
               << " sharding at index " << valueIndex
               << " must refer to all manual_axes: {" << manualAxes << "}";
      }
    }
  }

  return success();
}

}  // namespace

mlir::LogicalResult TensorShardingAttr::verifyForType(
    Type type, MeshAttr mesh,
    std::function<InFlightDiagnostic(StringRef)> emitError,
    bool checkDivisibility) {
  return verifyTensorShardingAttr(*this, type, mesh, emitError,
                                  checkDivisibility,
                                  /*alreadyManualAxes=*/ManualAxisToOwner());
}

LogicalResult ManualComputationOp::verify() {
  ManualAxisToOwner alreadyManualAxes =
      getParentManualComputationOps(this->getOperation());
  for (StringRef axisName : getManualAxes()) {
    if (alreadyManualAxes.contains(axisName)) {
      return emitBoundAxisInManualComputationError(
          [this](StringRef msg) { return emitOpError(msg); }, axisName,
          alreadyManualAxes[axisName]->getLoc());
    }
  }

  // Pick any mesh. verifyManualComputationValue will verify all in/out
  // shardings refer to the same mesh.
  SymbolRefAttr meshName;
  if (!getInShardings().empty()) {
    meshName = getInShardings().getShardings().front().getMeshSymName();
  } else if (!getOutShardings().empty()) {
    meshName = getOutShardings().getShardings().front().getMeshSymName();
  }

  if (!getManualAxes().empty()) {
    if (!meshName) {
      return emitOpError(
          "cannot have manual_axes when there are no input/output shardings.");
    }
    if (!llvm::is_sorted(
            getManualAxes(),
            getMeshAttr(*this, meshName).getAxisNameComparator())) {
      return emitOpError("manual axes are not ordered w.r.t. mesh");
    }
  }
  if (failed(verifyManualComputationValue(
          *this, getOperandTypes(), getBody().getArgumentTypes(),
          getInShardings(), meshName, getManualAxes(), "operand")) ||
      failed(verifyManualComputationValue(
          *this, getResultTypes(), getBodyTerminatorOpOperandTypes(*this),
          getOutShardings(), meshName, getManualAxes(), "result"))) {
    return failure();
  }

  return success();
}

LogicalResult PropagationBarrierOp::verify() {
  if (getAllowedDirection() == PropagationDirection::BOTH) {
    return emitOpError(
        "cannot specify `BOTH` as the direction. Not blocking propagation "
        "direction makes the op redundant.");
  }
  return success();
}

LogicalResult SdyDialect::verifyRegionArgAttribute(Operation* op,
                                                   unsigned regionIndex,
                                                   unsigned argIndex,
                                                   NamedAttribute attr) {
  if (auto funcOp = dyn_cast<FuncOp>(op)) {
    return verifyFuncAttribute(
        funcOp, attr, funcOp.getArgumentTypes()[argIndex], argIndex, "arg");
  }
  return success();
}

LogicalResult SdyDialect::verifyRegionResultAttribute(Operation* op,
                                                      unsigned regionIndex,
                                                      unsigned resultIndex,
                                                      NamedAttribute attr) {
  if (auto funcOp = dyn_cast<FuncOp>(op)) {
    return verifyFuncAttribute(funcOp, attr,
                               funcOp.getResultTypes()[resultIndex],
                               resultIndex, "result");
  }
  return success();
}

LogicalResult SdyDialect::verifyOperationAttribute(Operation* op,
                                                   NamedAttribute attr) {
  if (attr.getName() == kShardingAttr) {
    auto shardingPerValue =
        dyn_cast<TensorShardingPerValueAttr>(attr.getValue());
    if (!shardingPerValue) {
      return op->emitOpError("should have a sharding attribute of type ")
             << "TensorShardingPerValueAttr";
    }

    // TODO(tomnatan): verify all relevant tensors are bound to the same mesh

    return verifyTensorShardingPerValueAttr(
        shardingPerValue, op->getResultTypes(), op,
        [op](StringRef msg) { return op->emitOpError("result ") << msg; },
        SymbolRefAttr());
  }

  if (attr.getName() == kShardingRuleAttr) {
    auto shardingRule = dyn_cast<OpShardingRuleAttr>(attr.getValue());
    if (!shardingRule) {
      return op->emitOpError("should have a sharding rule attribute of type ")
             << "OpShardingRuleAttr for attr named '" << kShardingRuleAttr
             << "'";
    }
    return verifyOpShardingRuleAttr(shardingRule, op);
  }

  return success();
}

}  // namespace sdy
}  // namespace mlir
