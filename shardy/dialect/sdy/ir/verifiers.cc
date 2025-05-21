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

#include "shardy/dialect/sdy/ir/verifiers.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <tuple>
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
#include "mlir/IR/BuiltinOps.h"
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
// - The tensor should have a rank.
// - The number of dimension shardings is equal to the rank of the tensor.
// - Replicated axes are ordered w.r.t. `mesh` (see
//   AxisRefAttr::getMeshComparator).
// - Unreduced axes are ordered w.r.t. `mesh` (see
//   AxisRefAttr::getMeshComparator).
// - All dimension shardings, replicated axes, and unreduced axes are each a
//   valid axis-ref list (see `verifyAxisRefList`).
// - All sub-axes in `shardingAttr` (see `verifySubAxes`).
// - There are no duplicate axis-refs or sub-axes that overlap with one another
//   across all fields.
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
  if (mesh.isMaximal() || (!type && shardingAttr.isFullyReplicated())) {
    // A maximal sharding says that this op should be executed on a single
    // device. Skip checking against the type of the op. Just make sure there
    // are no dimension shardings and replicated axes.
    if (shardingAttr.getRank() != 0 ||
        !shardingAttr.getReplicatedAxes().empty() ||
        !shardingAttr.getUnreducedAxes().empty()) {
      return emitError(
          "a maximal sharding can only have a sharding with rank 0 and no "
          "replicated or unreduced axes.");
    }
    return success();
  }
  auto tensorType = dyn_cast<ShapedType>(type);
  if (auto tupleType = dyn_cast<TupleType>(type)) {
    if (tupleType.size() != 1) {
      return emitError("ops can only have a sharding for a tuple of size 1: ")
             << tupleType;
    }
    tensorType = dyn_cast<ShapedType>(tupleType.getType(0));
  }
  if (!tensorType) {
    if (shardingAttr.getRank() != 0 ||
        !shardingAttr.getReplicatedAxes().empty() ||
        !shardingAttr.getUnreducedAxes().empty()) {
      return emitError(
                 "non-shaped tensors can only have a sharding with rank 0 ")
             << "and no replicated or unreduced axes. type: " << type
             << ", sharding: " << shardingAttr;
    }
    return success();
  }

  if (!tensorType.hasRank()) {
    return emitError("only ranked tensors can have a sharding. ")
           << "type: " << type;
  }

  int64_t rank = tensorType.getRank();
  if (shardingAttr.getRank() != rank) {
    return emitError("sharding doesn't match tensor rank: ")
           << shardingAttr.getRank() << " != " << rank;
  }

  SmallDenseMap<StringRef, int64_t> axisNameToSize = mesh.getAxisNameToSize();

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

    if (dimSize == ShapedType::kDynamic) {
      continue;
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

  auto axisRefComparator = AxisRefAttr::getMeshComparator(mesh);
  auto verifySortedAxisRefList = [&](ArrayRef<AxisRefAttr> axisRefList,
                                     StringRef name) -> LogicalResult {
    if (!llvm::is_sorted(axisRefList, axisRefComparator)) {
      return emitError(name) << " axes are not ordered w.r.t. mesh";
    }
    return verifyAxisRefList(axisRefList, axisNameToSize, seenAxisRefs,
                             axisNameToSubAxes, emitError);
  };

  // Verify replicated axes
  if (failed(verifySortedAxisRefList(shardingAttr.getReplicatedAxes(),
                                     "replicated"))) {
    return failure();
  }

  // Verify unreduced axes
  if (failed(verifySortedAxisRefList(shardingAttr.getUnreducedAxes(),
                                     "unreduced"))) {
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

// Same as the overload above, but in addition verifies that if `shardingAttr`
// references a mesh by name, it's present in the module's symbol table.
LogicalResult verifyTensorShardingAttr(TensorShardingAttr shardingAttr,
                                       Type type, Operation* op,
                                       EmitErrorFn emitError) {
  MeshAttr mesh = shardingAttr.getMesh(op);
  if (!mesh) {
    // We can assume the sharding has a mesh symbol name.
    return emitError("unknown mesh: ") << shardingAttr.getMeshSymName();
  }
  return verifyTensorShardingAttr(shardingAttr, type, mesh, emitError,
                                  /*checkDivisibility=*/false,
                                  getParentManualComputationOps(op));
}

// Same as the overload above, but in addition verifies that if `shardingAttr`
// references a mesh by name, it's present in the given `symbolTable`.
LogicalResult verifyTensorShardingAttr(TensorShardingAttr shardingAttr,
                                       Type type, Operation* op,
                                       const SymbolTable& symbolTable,
                                       EmitErrorFn emitError) {
  MeshAttr mesh = shardingAttr.getMesh(symbolTable);
  if (!mesh) {
    // We can assume the sharding has a mesh symbol name.
    return emitError("unknown mesh: ") << shardingAttr.getMeshSymName();
  }
  return verifyTensorShardingAttr(shardingAttr, type, mesh, emitError,
                                  /*checkDivisibility=*/false,
                                  getParentManualComputationOps(op));
}

// Verifies the following for `shardingPerValueAttr`:
//
// - The number of tensor shardings is equal to the number of tensors (size of
//   `types`).
// - All shardings are valid (see `verifyTensorShardingAttr`).
// TODO(bartchr): relax this to allow different meshes when the op is a dataflow
// op
LogicalResult verifyTensorShardingPerValueAttr(
    TensorShardingPerValueAttr shardingPerValueAttr, TypeRange types,
    Operation* op, EmitErrorFn emitError, const SymbolTable& symbolTable) {
  ArrayRef<TensorShardingAttr> shardingsPerValue =
      shardingPerValueAttr.getShardings();
  if (types.empty() && shardingsPerValue.size() == 1) {
    TensorShardingAttr firstSharding = shardingsPerValue.front();
    if (firstSharding.getMesh(symbolTable).isMaximal() ||
        firstSharding.isFullyReplicated()) {
      return verifyTensorShardingAttr(
          firstSharding, Type(), op, symbolTable,
          getEmitValueInRangeErrorFn(emitError, types.size(), /*index=*/0));
    }
  }
  if (shardingsPerValue.size() != types.size()) {
    return emitError("shardings don't match number of values: ")
           << shardingsPerValue.size() << " shardings" << " vs " << types.size()
           << " values";
  }

  for (auto [index, entry] :
       llvm::enumerate(llvm::zip(shardingsPerValue, types))) {
    EmitErrorFn valueEmitError =
        getEmitValueInRangeErrorFn(emitError, types.size(), index);
    auto [shardingAttr, resultType] = entry;
    if (failed(verifyTensorShardingAttr(shardingAttr, resultType, op,
                                        symbolTable, valueEmitError))) {
      return failure();
    }
  }

  return success();
}

// Same as the overload above, but creates a `SymbolTable`.
LogicalResult verifyTensorShardingPerValueAttr(
    TensorShardingPerValueAttr shardingPerValueAttr, TypeRange types,
    Operation* op, EmitErrorFn emitError) {
  return verifyTensorShardingPerValueAttr(
      shardingPerValueAttr, types, op, emitError,
      SymbolTable(op->getParentOfType<ModuleOp>()));
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

// Verifies the following for `indices`.
// 1. It is sorted.
// 2. Its elements are unique.
// 3. Its elements are in range [0, `numFactors`).
LogicalResult indicesSortedUniqueInBound(Operation* op, int64_t numFactors,
                                         ArrayRef<int64_t> indices) {
  if (indices.empty()) {
    return success();
  }

  if (!llvm::is_sorted(indices)) {
    return op->emitOpError("indices of special factors must be sorted");
  }
  if (std::adjacent_find(indices.begin(), indices.end()) != indices.end()) {
    return op->emitOpError("indices of special factors must be unique");
  }

  if (indices.front() < 0) {
    return op->emitOpError("index must be non-negative");
  }
  if (indices.back() >= numFactors) {
    return op->emitOpError("index must be less than ")
           << numFactors << ", got: " << indices.back();
  }

  return success();
}

// Verifies the following for an `OpShardingRuleAttr`:
//
// - All defined factor sizes are used by at least one operand/result mapping.
// - All operand/result mappings are valid (see `verifyShardingRuleMapping`).
LogicalResult verifyOpShardingRuleAttr(OpShardingRuleAttr shardingRule,
                                       Operation* op) {
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

  ArrayRef<int64_t> reductionFactors = shardingRule.getReductionFactors();
  ArrayRef<int64_t> needReplicationFactors =
      shardingRule.getNeedReplicationFactors();
  ArrayRef<int64_t> permutationFactors = shardingRule.getPermutationFactors();

  if (failed(indicesSortedUniqueInBound(op, shardingRule.getNumFactors(),
                                        reductionFactors))) {
    return failure();
  }
  if (failed(indicesSortedUniqueInBound(op, shardingRule.getNumFactors(),
                                        needReplicationFactors))) {
    return failure();
  }
  if (failed(indicesSortedUniqueInBound(op, shardingRule.getNumFactors(),
                                        permutationFactors))) {
    return failure();
  }

  SmallDenseSet<int64_t> specialFactors;
  specialFactors.insert(reductionFactors.begin(), reductionFactors.end());
  specialFactors.insert(needReplicationFactors.begin(),
                        needReplicationFactors.end());
  specialFactors.insert(permutationFactors.begin(), permutationFactors.end());
  if (specialFactors.size() != reductionFactors.size() +
                                   needReplicationFactors.size() +
                                   permutationFactors.size()) {
    return op->emitOpError(
        "a factor can only be in one of the reduction, need replication, or "
        "permutation factor sets");
  }

  if (failed(indicesSortedUniqueInBound(
          op, shardingRule.getNumFactors(),
          shardingRule.getBlockedPropagationFactors()))) {
    return failure();
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
      depOp && inDialect<SdyDialect>(depOp) &&
      !mlir::isa<ShardableDataFlowOpInterface>(depOp)) {
    return emitOpError(
               "expected input of sdy.data_flow_edge to not be defined by an "
               "SdyDialect op (other than an sdy.named_computation).")
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
int64_t accumulatedManualAxesSize(
    Operation* op, ArrayRef<AxisRefAttr> dimShardingAxes,
    const llvm::SmallDenseSet<StringRef>& manualAxes, MeshAttr mesh) {
  int64_t axesFactor = 1;
  for (AxisRefAttr axisRef : dimShardingAxes) {
    if (manualAxes.contains(axisRef.getName())) {
      axesFactor *= axisRef.getSize(mesh);
    }
  }

  return axesFactor;
}

// Returns an iterator to a manual axis that comes after a free axis in `axes`.
// If no such axis exists, returns `axes.end()`.
ArrayRef<AxisRefAttr>::iterator findManualAxisAfterFreeAxis(
    ArrayRef<AxisRefAttr> axes,
    const llvm::SmallDenseSet<StringRef>& manualAxes) {
  auto isManualAxis = [&](AxisRefAttr axisRef) {
    return manualAxes.contains(axisRef.getName());
    ;
  };
  return std::find_if(llvm::find_if_not(axes, isManualAxis), axes.end(),
                      isManualAxis);
}

// For each set of op values (operands/results) and corresponding sharding for
// each value, verifies:
// 1. the sharding list itself wrt the mesh and `globalTypes`,
// 2. the in/out shardings are valid w.r.t the corresponding global type,
// 3. the number of global and local tensor inputs/outputs of the op region
//    match,
// 4. the manual axes come before any free axes in each dim sharding,
// 5. The manual axes cannot introduce padding. The dimension size must be
//    divisible by the corresponding manual axes size.
// 6. the global shape and local shapes of the op regions arguments/results
//    match, and
// 7. No manual axes are split.
//
// `valueKindStr` is a string included in any verification error message
// specifying whether the values we are verifying are the operands or results.
// Pass in "operand" or "result".
template <typename GlobalRangeT, typename LocalRangeT>
LogicalResult verifyManualComputationValue(
    ManualComputationOp op, GlobalRangeT globalTypes, LocalRangeT localTypes,
    TensorShardingPerValueAttr shardingPerValueAttr,
    const SymbolTable& symbolTable,
    const llvm::SmallDenseSet<StringRef>& manualAxesSet,
    StringRef valueKindStr) {
  if (globalTypes.empty() && localTypes.empty() &&
      shardingPerValueAttr.getShardings().empty()) {
    // Nothing to verify since there are no values.
    return success();
  }
  // 1. Verify the sharding list itself wrt the mesh and `globalTypes`.
  // 2. Verify the in/out shardings are valid w.r.t the corresponding global
  //    type.
  if (failed(verifyTensorShardingPerValueAttr(
          shardingPerValueAttr, globalTypes, op,
          [op, valueKindStr](StringRef msg) {
            return op->emitOpError(valueKindStr) << " " << msg;
          },
          symbolTable))) {
    return failure();
  }

  // 3. Verify the number of global and local tensor inputs/outputs of the op
  //    region match.
  if (globalTypes.size() != localTypes.size()) {
    return op->emitOpError("number of op ")
           << valueKindStr << "s and region " << valueKindStr
           << "s must match. Op has " << globalTypes.size() << " op "
           << valueKindStr << "s and " << localTypes.size() << " region "
           << valueKindStr << "s";
  }

  for (auto [valueIndex, valueEntry] : llvm::enumerate(llvm::zip_equal(
           globalTypes, localTypes, shardingPerValueAttr.getShardings()))) {
    auto [globalType, localType, sharding] = valueEntry;

    // 4. Verify the manual axes come before any free axes in each dim sharding.
    for (auto [dim, dimSharding] :
         llvm::enumerate(sharding.getDimShardings())) {
      ArrayRef<AxisRefAttr> axes = dimSharding.getAxes();
      if (ArrayRef<AxisRefAttr>::iterator manualAxisItr =
              findManualAxisAfterFreeAxis(axes, manualAxesSet);
          manualAxisItr != axes.end()) {
        return op->emitOpError(valueKindStr)
               << " sharding at index " << valueIndex
               << " must have all manual axes come before free axes in its "
               << "dimension sharding at index " << dim << ". Saw manual axis "
               << manualAxisItr->toString() << " after free axis "
               << (manualAxisItr - 1)->toString();
      }
    }

    SmallVector<int64_t> newDimSizes;
    auto globalShapedType = mlir::dyn_cast<ShapedType>(globalType);
    if (!globalShapedType) {
      // Skipping verification for non-shaped types. This could for example be
      // a token type.
      continue;
    }
    for (auto [dimensionSize, dimSharding] : llvm::zip_equal(
             globalShapedType.getShape(), sharding.getDimShardings())) {
      if (dimensionSize == ShapedType::kDynamic) {
        newDimSizes.push_back(ShapedType::kDynamic);
      } else {
        // 5. The manual axes cannot introduce padding. The dimension size must
        //    be divisible by the corresponding manual axes size.

        // Safe to call `getMesh` because the sharding was already verified.
        int64_t manualAxesSize =
            accumulatedManualAxesSize(op, dimSharding.getAxes(), manualAxesSet,
                                      sharding.getMesh(symbolTable));
        if (dimensionSize % manualAxesSize != 0) {
          return op->emitOpError(valueKindStr)
                 << " dimension size " << dimensionSize
                 << " is not divisible by the manual axes size "
                 << manualAxesSize;
        }
        newDimSizes.push_back(dimensionSize / manualAxesSize);
      }
    }
    // 6. Verify the global shape and local shapes of the op regions
    //    arguments/results match.
    auto expectedLocalRankedType =
        RankedTensorType::get(newDimSizes, globalShapedType.getElementType());
    auto localRankedType = mlir::cast<RankedTensorType>(localType);
    if (expectedLocalRankedType != localRankedType) {
      return op->emitOpError(valueKindStr)
             << " shape, corresponding sharding, and region " << valueKindStr
             << " shape at index " << valueIndex
             << " must match. Expected local shape " << expectedLocalRankedType
             << ", actual local shape " << localRankedType;
    }

    // 7. No manual axes are split.
    if (sharding.anyOfAxisRef([&](AxisRefAttr axis) {
          return axis.getSubAxisInfo() &&
                 manualAxesSet.contains(axis.getName());
        })) {
      return op->emitOpError(valueKindStr)
             << " sharding at index " << valueIndex
             << " cannot use a manual axis as a sub/split axis. Saw manual "
                "axes {"
             << manualAxesSet << "} and sharding " << sharding << ".";
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
      getParentManualComputationOps(getOperation());
  for (StringRef axisName : getManualAxes()) {
    if (alreadyManualAxes.contains(axisName)) {
      return emitBoundAxisInManualComputationError(
          [this](StringRef msg) { return emitOpError(msg); }, axisName,
          alreadyManualAxes[axisName]->getLoc());
    }
  }

  bool noInOutShardings = getInShardings().empty() && getOutShardings().empty();
  if (noInOutShardings && !isa<ReturnOp>(&getBody().front().front()) &&
      !getManualAxes().empty()) {
    return emitOpError(
        "cannot have manual_axes when there are no in/out shardings and the "
        "body is not empty.");
  }

  SymbolTable symbolTable(getOperation()->getParentOfType<ModuleOp>());
  llvm::SmallDenseSet<StringRef> manualAxesSet(getManualAxes().begin(),
                                               getManualAxes().end());
  if (failed(verifyManualComputationValue(
          *this, getOperandTypes(), getBody().getArgumentTypes(),
          getInShardings(), symbolTable, manualAxesSet, "operand")) ||
      failed(verifyManualComputationValue(
          *this, getResultTypes(), getBodyTerminatorOpOperandTypes(*this),
          getOutShardings(), symbolTable, manualAxesSet, "result"))) {
    return failure();
  }

  if (noInOutShardings) {
    return success();
  }

  // We verify a common mesh here so an invalid mesh name reference will be
  // caught before.
  MeshAttr mesh = getCommonMesh(getInShardings().getShardings(),
                                getOutShardings().getShardings(), symbolTable);
  if (!mesh && !getManualAxes().empty()) {
    return emitOpError(
        "all in and out shardings must be bound to the same mesh or an empty "
        "mesh.");
  }

  for (StringAttr axisName : getManualAxes()) {
    if (mesh && !mesh.hasAxis(axisName)) {
      return emitOpError("unknown manual axis: ") << axisName;
    }
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

namespace {

LogicalResult allInnerAndOuterTypesMatchInNamedComputation(
    NamedComputationOp op, TypeRange innerTypes, TypeRange outerTypes,
    StringRef innerName, StringRef outerName) {
  if (innerTypes.size() != outerTypes.size()) {
    return op.emitError("number of ")
           << innerName << "s must match the number of " << outerName
           << "s: " << innerTypes.size() << " != " << outerTypes.size();
  }

  for (auto [i, types] :
       llvm::enumerate(llvm::zip_equal(innerTypes, outerTypes))) {
    auto [innerType, outerType] = types;
    if (innerType != outerType) {
      return op.emitError("expected the type of the ")
             << i << "'th " << innerName
             << " to match the type of the corresponding " << outerName << ": "
             << innerType << " vs " << outerType;
    }
  }

  return success();
}

}  // namespace

LogicalResult NamedComputationOp::verify() {
  if (failed(allInnerAndOuterTypesMatchInNamedComputation(
          *this, getBody().getArgumentTypes(), getOperandTypes(),
          "block argument", "operand")) ||
      failed(allInnerAndOuterTypesMatchInNamedComputation(
          *this, getBodyTerminatorOpOperandTypes(*this), getResultTypes(),
          "returned value", "result"))) {
    return failure();
  }

  std::optional<TensorShardingPerValueAttr> inShardings = getInShardings();
  std::optional<TensorShardingPerValueAttr> outShardings = getOutShardings();
  if (!(inShardings || outShardings)) {
    return success();
  }

  // TODO(pxy): remove this once the `ShardableDataFlowOpInterface` is verified.
  // Verify the in/out shardings.
  if (inShardings &&
      failed(verifyTensorShardingPerValueAttr(
          *inShardings, getOperandTypes(), *this, [this](StringRef msg) {
            return emitOpError("in_shardings ") << msg;
          }))) {
    return failure();
  }
  if (outShardings &&
      failed(verifyTensorShardingPerValueAttr(
          *outShardings, getResultTypes(), *this, [this](StringRef msg) {
            return emitOpError("out_shardings ") << msg;
          }))) {
    return failure();
  }

  return success();
}

namespace {

// Verifies:
// 1. All collective axes per dimension are valid (see `verifyAxisRefList`).
// 2. Applying `collectiveAxesPerDim` to the operand sharding (via
//    `getExpectedResultDimSharding`) gets the output sharding.
template <typename OpTy>
LogicalResult verifyCollectiveWithAxesPerDim(
    OpTy op, ArrayRef<AxisRefListAttr> collectiveAxesPerDim,
    std::function<FailureOr<SmallVector<AxisRefAttr>>(
        DimensionShardingAttr operandDimSharding,
        ArrayRef<AxisRefAttr> dimCollectiveAxes, int64_t dim, MeshAttr mesh)>
        getExpectedResultDimSharding) {
  TensorShardingAttr resultSharding = op.getOutSharding();
  TensorShardingAttr operandSharding =
      getOrCreateSharding(op.getOperand(), resultSharding.getMeshOrRef());
  MeshAttr mesh = resultSharding.getMesh(op);
  MeshAttr operandMesh = operandSharding.getMesh(op);

  // 1. Verify all collective axes.
  SmallDenseSet<AxisRefAttr> seenAxisRefs;
  SmallDenseMap<StringRef, SmallVector<AxisRefAttr>> axisNameToSubAxes;
  SmallDenseMap<StringRef, int64_t> axisNameToSize = mesh.getAxisNameToSize();
  for (AxisRefListAttr axisRefList : collectiveAxesPerDim) {
    if (failed(verifyAxisRefList(axisRefList.getValue(), axisNameToSize,
                                 seenAxisRefs, axisNameToSubAxes,
                                 getEmitErrorFn(op)))) {
      return failure();
    }
  }

  // 2. Verify that applying `collectiveAxesPerDim` to the operand gets
  // outSharding.
  // For example:
  // operand sharding: (a, b, c, d)
  // gathering axes: (c, d)
  // -> (a, b)

  ArrayRef<DimensionShardingAttr> resultDimShardings =
      resultSharding.getDimShardings();
  ArrayRef<DimensionShardingAttr> operandDimShardings =
      operandSharding.getDimShardings();
  // 2.1. Verify same rank of result sharding and the collective axes.
  if (resultDimShardings.size() != collectiveAxesPerDim.size()) {
    return op.emitOpError("result sharding has rank ")
           << resultDimShardings.size() << " but collective axes has rank "
           << collectiveAxesPerDim.size();
  }

  // 2.2. Verify that applying `collectiveAxesPerDim` to the operand gets
  // `resultDimShardings`.
  for (auto [dim, dimEntry] : llvm::enumerate(
           llvm::zip_equal(operandDimShardings, collectiveAxesPerDim))) {
    auto [operandDimSharding, dimCollectiveAxes] = dimEntry;
    // TODO(tomnatan): use AxisListRef by avoiding circular dep.
    auto expectedDimShardingOrFailure = getExpectedResultDimSharding(
        operandDimSharding, dimCollectiveAxes.getValue(), dim, mesh);
    if (failed(expectedDimShardingOrFailure)) {
      return failure();
    }
    ArrayRef<AxisRefAttr> expectedDimSharding =
        expectedDimShardingOrFailure.value();
    if (expectedDimSharding != resultDimShardings[dim].getAxes()) {
      return op.emitOpError("result sharding doesn't match expected sharding ")
             << strippedAttrsString(ArrayRef(expectedDimSharding),
                                    /*stripMnemonic=*/true)
             << " on dimension " << dim;
    }
  }

  return success();
}

// Removes `gatheringAxes` from the suffix of axes in `dimSharding` and returns
// the result, or emits an error if `gatheringAxes` are not a suffix.
FailureOr<SmallVector<AxisRefAttr>> gatherAxesAlongDim(
    DimensionShardingAttr dimSharding, ArrayRef<AxisRefAttr> gatheringAxes,
    int64_t dim, MeshAttr mesh, StringRef axisType, EmitErrorFn emitError) {
  SmallVector<AxisRefAttr> expectedDimSharding =
      llvm::to_vector(dimSharding.getAxes());
  for (auto gatheringAxis : llvm::reverse(gatheringAxes)) {
    if (expectedDimSharding.empty() ||
        !gatheringAxis.suffixOf(expectedDimSharding.back(), mesh)) {
      return emitError("can't apply ")
             << axisType << " axis " << gatheringAxis.toString()
             << " to operand sharding on dimension " << dim;
    }
    AxisRefAttr shardingAxis = expectedDimSharding.back();
    expectedDimSharding.pop_back();
    if (std::optional<AxisRefAttr> prefixAxis =
            shardingAxis.getPrefixWithoutOverlap(gatheringAxis)) {
      expectedDimSharding.push_back(*prefixAxis);
    }
  }
  return expectedDimSharding;
}

// Appends `slicingAxes` to the axes in `dimSharding` and returns the result.
SmallVector<AxisRefAttr> sliceAxesAlongDim(DimensionShardingAttr dimSharding,
                                           ArrayRef<AxisRefAttr> slicingAxes,
                                           MeshAttr mesh) {
  SmallVector<AxisRefAttr> expectedDimSharding =
      llvm::to_vector(dimSharding.getAxes());
  for (auto slicingAxis : slicingAxes) {
    addAxisOrMerge(expectedDimSharding, slicingAxis, mesh);
  }
  return expectedDimSharding;
}

}  // namespace

LogicalResult AllGatherOp::verify() {
  return verifyCollectiveWithAxesPerDim(
      *this, getGatheringAxes(),
      [this](DimensionShardingAttr operandDimSharding,
             ArrayRef<AxisRefAttr> dimGatheringAxes, int64_t dim,
             MeshAttr mesh) -> FailureOr<SmallVector<AxisRefAttr>> {
        return gatherAxesAlongDim(operandDimSharding, dimGatheringAxes, dim,
                                  mesh, "gathering", getEmitErrorFn(*this));
      });
}

LogicalResult AllSliceOp::verify() {
  return verifyCollectiveWithAxesPerDim(
      *this, getSlicingAxes(),
      [](DimensionShardingAttr operandDimSharding,
         ArrayRef<AxisRefAttr> dimSlicingAxes, int64_t dim,
         MeshAttr mesh) -> FailureOr<SmallVector<AxisRefAttr>> {
        return sliceAxesAlongDim(operandDimSharding, dimSlicingAxes, mesh);
      });
}

LogicalResult AllToAllOp::verify() {
  TensorShardingAttr operandSharding = getSharding(getOperand());
  TensorShardingAttr resultSharding = getOutSharding();
  MeshAttr mesh = resultSharding.getMesh(*this);

  ArrayRef<AllToAllParamAttr> params = getParams();
  // 1. Verify that the parameter list is not empty.
  if (params.empty()) {
    return emitOpError("parameter list is empty");
  }
  auto decomposeParam = [](AllToAllParamAttr param)
      -> std::tuple<ArrayRef<AxisRefAttr>, int64_t, int64_t> {
    return std::make_tuple(param.getAxes(), param.getSrcDim(),
                           param.getTgtDim());
  };

  int64_t rank = getTensorRank(getResult());
  BitVector seenDims(rank);
  int64_t prevSrcDim = -1;
  SmallDenseSet<AxisRefAttr> seenAxisRefs;
  SmallDenseMap<StringRef, SmallVector<AxisRefAttr>> axisNameToSubAxes;
  SmallDenseMap<StringRef, int64_t> axisNameToSize = mesh.getAxisNameToSize();
  ArrayRef<DimensionShardingAttr> resultDimShardings =
      resultSharding.getDimShardings();
  ArrayRef<DimensionShardingAttr> operandDimShardings =
      operandSharding.getDimShardings();
  for (AllToAllParamAttr param : params) {
    auto [axes, srcDim, tgtDim] = decomposeParam(param);
    // 2. Verify `axes` is a valid list of axes.
    if (failed(verifyAxisRefList(axes, axisNameToSize, seenAxisRefs,
                                 axisNameToSubAxes, getEmitErrorFn(*this)))) {
      return failure();
    }
    // 3. Verify `src_dim` and `tgt_dim`.
    auto verifyDim = [this, rank](int64_t dim,
                                  StringRef dimName) -> LogicalResult {
      if (dim < 0 || dim >= rank) {
        return emitOpError(dimName) << " dimension " << dim
                                    << " is out of range [0, " << rank << ")";
      }
      return success();
    };

    if (failed(verifyDim(srcDim, "source"))) {
      return failure();
    }
    if (failed(verifyDim(tgtDim, "target"))) {
      return failure();
    }
    // Verify that `src_dim` and `tgt_dim` are not overlapping.
    for (int64_t dim : {srcDim, tgtDim}) {
      if (seenDims.test(dim)) {
        return emitOpError(
                   "overlapping source/target dimensions in all-to-all "
                   "params: ")
               << dim;
      }
      seenDims.set(dim);
    }
    // Verify that `src_dim` is sorted in ascending order.
    if (prevSrcDim >= srcDim) {
      return emitOpError(
                 "source dimensions are not sorted in ascending order: ")
             << srcDim << " appears after " << prevSrcDim;
    }
    prevSrcDim = srcDim;
  }
  // Move axes in another loop to avoid the overlapping dimensions error being
  // covered by the dim sharding verification.
  auto verifyDimSharding =
      [&](int64_t dim,
          ArrayRef<AxisRefAttr> expectedDimSharding) -> LogicalResult {
    if (expectedDimSharding != resultDimShardings[dim].getAxes()) {
      return emitOpError("result sharding doesn't match expected sharding ")
             << strippedAttrsString(ArrayRef(expectedDimSharding),
                                    /*stripMnemonic=*/true)
             << " on dimension " << dim;
    }
    return success();
  };
  for (AllToAllParamAttr param : params) {
    auto [axes, srcDim, tgtDim] = decomposeParam(param);
    // 4. Verify that moving `axes` from `src_dim` to `tgt_dim` in the
    // operand sharding gets `out_sharding`.
    auto expectedSrcDimShardingOrFailure =
        gatherAxesAlongDim(operandDimShardings[srcDim], axes, srcDim, mesh,
                           "all-to-all", getEmitErrorFn(*this));
    if (failed(expectedSrcDimShardingOrFailure)) {
      return failure();
    }
    if (failed(verifyDimSharding(srcDim,
                                 expectedSrcDimShardingOrFailure.value()))) {
      return failure();
    }
    if (failed(verifyDimSharding(
            tgtDim,
            sliceAxesAlongDim(operandDimShardings[tgtDim], axes, mesh)))) {
      return failure();
    }
  }
  // 5. Verify that non-src/tgt dims in the operand sharding stay the same in
  // `out_sharding`.
  for (auto [dim, dimShardings] : llvm::enumerate(
           llvm::zip_equal(operandDimShardings, resultDimShardings))) {
    auto [operandDimSharding, resultDimSharding] = dimShardings;
    if (!seenDims.test(dim) &&
        failed(verifyDimSharding(dim, operandDimSharding.getAxes()))) {
      return failure();
    }
  }
  return success();
}

LogicalResult CollectivePermuteOp::verify() {
  TensorShardingAttr operandSharding = getSharding(getOperand());
  TensorShardingAttr resultSharding = getOutSharding();
  MeshAttr mesh = resultSharding.getMesh(*this);
  MeshAttr operandMesh = operandSharding.getMesh(*this);
  if (mesh.getAxes() != operandMesh.getAxes()) {
    return emitOpError("result mesh has different axes than operand mesh")
               .attachNote(getTensor().getLoc())
           << "operand mesh: " << operandMesh;
  }
  if (operandSharding.getMeshOrRef() != resultSharding.getMeshOrRef() &&
      mesh.getDeviceIds() == operandMesh.getDeviceIds()) {
    return emitOpError(
               "result mesh name is different but same device ids as operand")
               .attachNote(getTensor().getLoc())
           << "operand mesh: " << operandMesh;
  }

  ArrayRef<DimensionShardingAttr> resultDimShardings =
      resultSharding.getDimShardings();
  ArrayRef<DimensionShardingAttr> operandDimShardings =
      operandSharding.getDimShardings();

  // For each dimension, verify that the sharded size of input and output
  // dimension shardings match.
  for (auto [dim, dimShardings] : llvm::enumerate(
           llvm::zip_equal(operandDimShardings, resultDimShardings))) {
    auto [inDimSharding, outDimSharding] = dimShardings;
    if (inDimSharding.getAxes() == outDimSharding.getAxes()) {
      continue;
    }
    int64_t inShardedSize = inDimSharding.getShardedSize(mesh);
    int64_t outShardedSize = outDimSharding.getShardedSize(mesh);
    if (inShardedSize != outShardedSize) {
      return emitOpError("sharded size of result doesn't match operand ")
             << "on dimension " << dim << ": " << outShardedSize
             << " != " << inShardedSize;
    }
  }

  return success();
}

LogicalResult verifyCollectiveOp(Operation* rawOp) {
  auto collectiveOp = dyn_cast<CollectiveOpInterface>(rawOp);
  if (!collectiveOp) {
    return failure();
  }
  // 1. Verify operand has a sharding.
  TensorShardingAttr optionalOperandSharding =
      getSharding(collectiveOp.getTensor());
  if (!collectiveOp.allowMissingInputSharding() && !optionalOperandSharding) {
    return collectiveOp.emitOpError("collective on operand without sharding");
  }

  // 2. Verify result sharding is valid w.r.t the corresponding type.
  TensorShardingAttr resultSharding = collectiveOp.getOutSharding();
  if (auto res =
          verifyTensorShardingAttr(resultSharding, collectiveOp.getType(),
                                   collectiveOp, getEmitErrorFn(collectiveOp));
      failed(res)) {
    return res;
  }

  // 3. Verify MeshAttr of result and operand is the same.
  if (!collectiveOp.allowDifferentMeshes()) {
    MeshAttr mesh = resultSharding.getMesh(collectiveOp);
    MeshAttr operandMesh = optionalOperandSharding
                               ? optionalOperandSharding.getMesh(collectiveOp)
                               : nullptr;
    if (operandMesh && mesh != operandMesh) {
      return collectiveOp.emitOpError("result mesh does not match operand mesh")
                 .attachNote(collectiveOp.getTensor().getLoc())
             << "operand mesh: " << operandMesh;
    }
  }

  // 4. Verify same rank of the result sharding and operand sharding.
  if (optionalOperandSharding &&
      resultSharding.getRank() != optionalOperandSharding.getRank()) {
    return collectiveOp.emitOpError("result sharding has rank ")
           << resultSharding.getRank() << " but operand sharding has rank "
           << optionalOperandSharding.getRank();
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

    return verifyTensorShardingPerValueAttr(
        shardingPerValue, op->getResultTypes(), op,
        [op](StringRef msg) { return op->emitOpError("result ") << msg; });
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

LogicalResult AllReduceOp::verify() {
  TensorShardingAttr resultSharding = getOutSharding();
  TensorShardingAttr operandSharding =
      getOrCreateSharding(getOperand(), resultSharding.getMeshOrRef());
  MeshAttr mesh = resultSharding.getMesh(*this);
  if (!operandSharding.areDimAxesEqual(resultSharding)) {
    return emitOpError("operand and result sharding have different axes");
  }

  // 1. Verify all reduction axes are valid.
  SmallDenseSet<AxisRefAttr> seenAxisRefs;
  SmallDenseMap<StringRef, SmallVector<AxisRefAttr>> axisNameToSubAxes;
  ArrayRef<AxisRefAttr> reductionAxes = getReductionAxes();
  SmallDenseMap<StringRef, int64_t> axisNameToSize = mesh.getAxisNameToSize();
  if (auto res = verifyAxisRefList(reductionAxes, axisNameToSize, seenAxisRefs,
                                   axisNameToSubAxes, getEmitErrorFn(*this));
      failed(res)) {
    return res;
  }

  // 2. Verify no axis from reduction_axes overlap with the operand sharding
  // axes.
  for (AxisRefAttr reductionAxisRef : reductionAxes) {
    if (operandSharding.anyOfAxisRef([reductionAxisRef](AxisRefAttr axisRef) {
          return axisRef.overlaps(reductionAxisRef);
        })) {
      return emitOpError("reduction axis ")
             << reductionAxisRef.toString()
             << " overlaps with operand sharding";
    }
  }

  return success();
}

LogicalResult ReduceScatterOp::verify() {
  return verifyCollectiveWithAxesPerDim(
      *this, getReduceScatterAxes(),
      [](DimensionShardingAttr operandDimSharding,
         ArrayRef<AxisRefAttr> dimSlicingAxes, int64_t dim,
         MeshAttr mesh) -> FailureOr<SmallVector<AxisRefAttr>> {
        return sliceAxesAlongDim(operandDimSharding, dimSlicingAxes, mesh);
      });
}

}  // namespace sdy
}  // namespace mlir
