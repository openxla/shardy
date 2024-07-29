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

#include "shardy/dialect/sdy/ir/dialect.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/enums.cc.inc"
#include "shardy/dialect/sdy/ir/parsers.h"   // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/printers.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"

namespace mlir {
namespace sdy {

void SdyDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "shardy/dialect/sdy/ir/attrs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "shardy/dialect/sdy/ir/ops.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// MeshAttr
//===----------------------------------------------------------------------===//

int64_t MeshAttr::getAxisSize(StringRef axisName) const {
  for (MeshAxisAttr meshAxis : getAxes()) {
    if (meshAxis.getName() == axisName) {
      return meshAxis.getSize();
    }
  }
  // Since verification will fail if an axis name doesn't appear in the bound
  // mesh, we can assume we would never get here.
  llvm_unreachable("unknown axis name");
}

int64_t MeshAttr::getTotalSize() const {
  ArrayRef<MeshAxisAttr> axes = getAxes();
  return std::accumulate(
      axes.begin(), axes.end(), 1,
      [](int64_t cur, MeshAxisAttr axis) { return cur * axis.getSize(); });
}

std::function<bool(StringRef lhs, StringRef rhs)>
MeshAttr::getAxisNameComparator() const {
  ArrayRef<MeshAxisAttr> axes = getAxes();
  return [axes](StringRef lhs, StringRef rhs) {
    if (lhs == rhs) {
      return false;
    }

    for (MeshAxisAttr axis : axes) {
      if (axis.getName() == lhs) {
        return true;
      }
      if (axis.getName() == rhs) {
        return false;
      }
    }

    llvm_unreachable("axis names not present in mesh");
  };
}

//===----------------------------------------------------------------------===//
// SubAxisInfoAttr
//===----------------------------------------------------------------------===//

bool SubAxisInfoAttr::operator<(const SubAxisInfoAttr& rhs) const {
  return std::make_pair(getPreSize(), getSize()) <
         std::make_pair(rhs.getPreSize(), rhs.getSize());
}

//===----------------------------------------------------------------------===//
// AxisRefAttr
//===----------------------------------------------------------------------===//

std::function<bool(AxisRefAttr lhs, AxisRefAttr rhs)>
AxisRefAttr::getMeshComparator(MeshAttr mesh) {
  return [mesh](AxisRefAttr lhs, AxisRefAttr rhs) {
    StringRef lhsName = lhs.getName();
    StringRef rhsName = rhs.getName();
    if (lhsName == rhsName) {
      // Both axis-refs have the same name, if one is a sub-axis and the other
      // is the full axis, then the sub-axis comes first.
      if (!lhs.getSubAxisInfo()) {
        return false;
      }
      if (!rhs.getSubAxisInfo()) {
        return true;
      }
      // Both axis-refs are sub-axes.
      return lhs.getSubAxisInfo() < rhs.getSubAxisInfo();
    }

    return mesh.getAxisNameComparator()(lhsName, rhsName);
  };
}

std::string AxisRefAttr::toString() const {
  return strippedAttrString(*this, /*stripMnemonic=*/true);
}

int64_t AxisRefAttr::getSize(MeshAttr mesh) const {
  if (getSubAxisInfo()) {
    return getSubAxisInfo().getSize();
  }
  return mesh.getAxisSize(getName());
}

int64_t AxisRefAttr::getSubAxisPreSize() const {
  return getSubAxisInfo() ? getSubAxisInfo().getPreSize() : 1;
}

bool AxisRefAttr::contains(AxisRefAttr other) const {
  if (other.getName() != getName()) {
    return false;
  }

  SubAxisInfoAttr thisSubAxisInfo = getSubAxisInfo();
  SubAxisInfoAttr otherSubAxisInfo = other.getSubAxisInfo();

  if (!thisSubAxisInfo) {
    // This is a full axis.
    return true;
  }

  if (!otherSubAxisInfo) {
    // The other is a full axis.
    return false;
  }

  return thisSubAxisInfo.getPreSize() <= otherSubAxisInfo.getPreSize() &&
         thisSubAxisInfo.getNextPreSize() >= otherSubAxisInfo.getNextPreSize();
}

bool AxisRefAttr::strictlyContains(AxisRefAttr other) const {
  return contains(other) && *this != other;
}

bool AxisRefAttr::prefixOf(AxisRefAttr other) const {
  return other.contains(*this) &&
         getSubAxisPreSize() == other.getSubAxisPreSize();
}

bool AxisRefAttr::strictPrefixOf(AxisRefAttr other) const {
  return prefixOf(other) && *this != other;
}

bool AxisRefAttr::overlaps(AxisRefAttr other) const {
  if (other.getName() != getName()) {
    return false;
  }

  SubAxisInfoAttr thisSubAxisInfo = getSubAxisInfo();
  SubAxisInfoAttr otherSubAxisInfo = other.getSubAxisInfo();

  if (!thisSubAxisInfo || !otherSubAxisInfo) {
    // One of the axes is full
    return true;
  }

  return thisSubAxisInfo.getPreSize() < otherSubAxisInfo.getNextPreSize() &&
         otherSubAxisInfo.getPreSize() < thisSubAxisInfo.getNextPreSize();
}

std::optional<AxisRefAttr> AxisRefAttr::getPrefixWithoutOverlap(
    AxisRefAttr other) const {
  if (!overlaps(other)) {
    return *this;
  }

  int64_t thisPreSize = getSubAxisPreSize();
  int64_t otherPreSize = other.getSubAxisPreSize();

  if (thisPreSize >= otherPreSize) {
    return std::nullopt;
  }
  return AxisRefAttr::get(getContext(), getName(),
                          SubAxisInfoAttr::get(getContext(), thisPreSize,
                                               otherPreSize / thisPreSize));
}

bool AxisRefAttr::canMerge(AxisRefAttr other) const {
  if (other.getName() != getName()) {
    return false;
  }
  if (!getSubAxisInfo() || !other.getSubAxisInfo()) {
    return false;
  }
  return getSubAxisInfo().getNextPreSize() ==
         other.getSubAxisInfo().getPreSize();
}

AxisRefAttr AxisRefAttr::merge(AxisRefAttr other, MeshAttr mesh) const {
  assert(canMerge(other));
  int64_t preSize = getSubAxisInfo().getPreSize();
  int64_t size = getSubAxisInfo().getSize() * other.getSubAxisInfo().getSize();
  if (preSize == 1 && mesh.getAxisSize(getName()) == size) {
    return AxisRefAttr::get(getContext(), getName());
  }
  return AxisRefAttr::get(getContext(), getName(),
                          SubAxisInfoAttr::get(getContext(), preSize, size));
}

//===----------------------------------------------------------------------===//
// DimensionShardingAttr
//===----------------------------------------------------------------------===//

DimensionShardingAttr DimensionShardingAttr::getSharded(
    StringRef axisName) const {
  assert(!getIsClosed() && "cannot shard a closed dimension");
  assert(llvm::all_of(getAxes(),
                      [axisName](AxisRefAttr axisRef) {
                        return axisName != axisRef.getName();
                      }) &&
         "cannot shard along an already bound axis");

  SmallVector<AxisRefAttr> newAxes(getAxes());
  newAxes.push_back(AxisRefAttr::get(getContext(), axisName));

  return DimensionShardingAttr::get(getContext(), newAxes, /*is_closed=*/false);
}

int64_t DimensionShardingAttr::getShardedSize(MeshAttr mesh) const {
  return std::accumulate(axis_begin(), axis_end(), 1,
                         [mesh](int64_t cur, AxisRefAttr axis) {
                           return cur * axis.getSize(mesh);
                         });
}

DimensionShardingAttr DimensionShardingAttr::sliceShardingAxes(
    // NOLINTNEXTLINE(readability-identifier-naming)
    size_t N, size_t M) const {
  return DimensionShardingAttr::get(getContext(), getAxes().slice(N, M),
                                    getIsClosed());
}

DimensionShardingAttr DimensionShardingAttr::dropFrontShardingAxes(
    // NOLINTNEXTLINE(readability-identifier-naming)
    size_t N) const {
  return sliceShardingAxes(N, getAxes().size() - N);
}

DimensionShardingAttr DimensionShardingAttr::takeFrontShardingAxes(
    // NOLINTNEXTLINE(readability-identifier-naming)
    size_t N) const {
  return sliceShardingAxes(0, N);
}

DimensionShardingAttr DimensionShardingAttr::dropPriority() const {
  return DimensionShardingAttr::get(getContext(), getAxes(), getIsClosed());
}

int64_t DimensionShardingAttr::getPriorityOrDefault() const {
  return getPriority().value_or(kDefaultPriority);
}

//===----------------------------------------------------------------------===//
// TensorShardingAttr
//===----------------------------------------------------------------------===//

namespace {

// Creates fully open or closed tensor sharding attr.
TensorShardingAttr getTensorShardingAttr(MLIRContext* context, int64_t rank,
                                         StringRef meshName, bool isClosed) {
  return TensorShardingAttr::get(
      context, meshName,
      /*dimShardings=*/
      SmallVector<DimensionShardingAttr>(
          rank, DimensionShardingAttr::get(context, {}, isClosed)),
      /*replicatedAxes=*/{});
}

}  // namespace

bool TensorShardingAttr::emptyAxes() const {
  return getReplicatedAxes().empty() &&
         llvm::all_of(getDimShardings(),
                      [](const DimensionShardingAttr& dimSharding) {
                        return dimSharding.emptyAxes();
                      });
}

bool TensorShardingAttr::anyOfAxisRef(
    std::function<bool(AxisRefAttr)> predicate) const {
  for (DimensionShardingAttr dimSharding : getDimShardings()) {
    if (llvm::any_of(dimSharding.getAxes(), predicate)) {
      return true;
    }
  }
  return llvm::any_of(getReplicatedAxes(), predicate);
}

bool TensorShardingAttr::isBound(StringRef axisName) const {
  return anyOfAxisRef([axisName](AxisRefAttr axisRef) {
    return axisName == axisRef.getName();
  });
}

bool TensorShardingAttr::canShard(int64_t dim, StringRef axisName) const {
  return !isBound(axisName) && !isClosed(dim);
}

bool TensorShardingAttr::canReplicate(StringRef axisName) const {
  return !isBound(axisName);
}

TensorShardingAttr TensorShardingAttr::closeShardingDims(
    ArrayRef<int64_t> dimIndices) const {
  SmallVector<DimensionShardingAttr> dimShardings(getDimShardings().begin(),
                                                  getDimShardings().end());
  for (int64_t dim : dimIndices) {
    dimShardings[dim] = DimensionShardingAttr::get(
        getContext(), dimShardings[dim].getAxes(), /*isClosed=*/true);
  }
  return TensorShardingAttr::get(getContext(), getMeshName(), dimShardings,
                                 getReplicatedAxes());
}

TensorShardingAttr TensorShardingAttr::openShardingDims(
    ArrayRef<int64_t> dimIndices) const {
  SmallVector<DimensionShardingAttr> dimShardings(getDimShardings().begin(),
                                                  getDimShardings().end());
  for (int64_t dim : dimIndices) {
    dimShardings[dim] = DimensionShardingAttr::get(
        getContext(), dimShardings[dim].getAxes(), /*isClosed=*/false);
  }
  return TensorShardingAttr::get(getContext(), getMeshName(), dimShardings,
                                 getReplicatedAxes());
}

TensorShardingAttr TensorShardingAttr::replaceDimSharding(
    int64_t dim, DimensionShardingAttr sharding) const {
  SmallVector<DimensionShardingAttr> shardings(getDimShardings());
  shardings[dim] = sharding;
  return TensorShardingAttr::get(getContext(), getMeshName(), shardings,
                                 getReplicatedAxes());
}

TensorShardingAttr TensorShardingAttr::getSharded(int64_t dim,
                                                  StringRef axisName) const {
  assert(canShard(dim, axisName));

  return replaceDimSharding(dim, getDimSharding(dim).getSharded(axisName));
}

TensorShardingAttr TensorShardingAttr::getReplicated(StringRef axisName,
                                                     MeshAttr mesh) const {
  assert(canReplicate(axisName));

  SmallVector<AxisRefAttr> newReplicatedAxes(getReplicatedAxes());
  AxisRefAttr newAxisRef = AxisRefAttr::get(getContext(), axisName);
  newReplicatedAxes.insert(
      llvm::upper_bound(newReplicatedAxes, newAxisRef,
                        AxisRefAttr::getMeshComparator(mesh)),
      newAxisRef);

  return TensorShardingAttr::get(getContext(), getMeshName(), getDimShardings(),
                                 newReplicatedAxes);
}

TensorShardingAttr TensorShardingAttr::getFullyClosed(MLIRContext* context,
                                                      int64_t rank,
                                                      StringRef meshName) {
  return getTensorShardingAttr(context, rank, meshName, /*isClosed=*/true);
}

TensorShardingAttr TensorShardingAttr::getFullyOpen(MLIRContext* context,
                                                    int64_t rank,
                                                    StringRef meshName) {
  return getTensorShardingAttr(context, rank, meshName, /*isClosed=*/false);
}

TensorShardingAttr TensorShardingAttr::getFullyOpenLike(
    TensorShardingAttr sharding) {
  return TensorShardingAttr::getFullyOpen(
      sharding.getContext(), sharding.getRank(), sharding.getMeshName());
}

//===----------------------------------------------------------------------===//
// TensorShardingPerValueAttr
//===----------------------------------------------------------------------===//

TensorShardingPerValueAttr TensorShardingPerValueAttr::getFullyOpen(
    MLIRContext* context, TypeRange types, StringRef meshName) {
  SmallVector<TensorShardingAttr> shardingPerResult;
  shardingPerResult.reserve(types.size());
  for (Type type : types) {
    int64_t rank = 0;
    // TODO(tomnatan): remove mlir:: once Attribute::dyn_cast is removed.
    if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
      rank = tensorType.getRank();
    }
    shardingPerResult.push_back(
        TensorShardingAttr::getFullyOpen(context, rank, meshName));
  }
  return TensorShardingPerValueAttr::get(context, shardingPerResult);
}

TensorShardingPerValueAttr TensorShardingPerValueAttr::replaceValueSharding(
    int64_t index, TensorShardingAttr sharding) const {
  SmallVector<TensorShardingAttr> shardings(getShardings());
  shardings[index] = sharding;
  return TensorShardingPerValueAttr::get(getContext(), shardings);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ConstantOpAdaptor adaptor(operands, attributes, properties);
  return hlo::inferConstantOp(location, adaptor.getValue(),
                              inferredReturnTypes);
}

bool ConstantOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  return stablehlo::ConstantOp::isCompatibleReturnTypes(l, r);
}

//===----------------------------------------------------------------------===//
// ManualComputationOp
//===----------------------------------------------------------------------===//

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
  TensorShardingAttr returnedSharding = outerManualSharding;
  for (auto [dimIndex, dimSharding] :
       llvm::enumerate(outerManualSharding.getDimShardings())) {
    ArrayRef<AxisRefAttr> dimAxes = dimSharding.getAxes();
    // Axes in the range [0, firstFreeAxis) are manual axes, and
    // [firstFreeAxis, dimAxes.size()) are free axes.
    llvm::ArrayRef<AxisRefAttr>::const_iterator firstFreeAxisIt =
        llvm::partition_point(dimAxes, [&manualAxes](AxisRefAttr axis) {
          return llvm::is_contained(manualAxes, axis.getName());
        });
    returnedSharding = returnedSharding.replaceDimSharding(
        dimIndex, shardingEraser(returnedSharding.getDimSharding(dimIndex),
                                 firstFreeAxisIt - dimAxes.begin()));
  }

  return returnedSharding;
}

// Removes free axes from the sharding.
//
// Guaranteed by verification that all in/out shardings in a
// `ManualComputationOp` are prefixed with the manual axes. So this removes the
// suffix of free axes (if any exist) from each dim sharding.
TensorShardingAttr eraseFreeAxes(TensorShardingAttr outerManualSharding,
                                 ArrayRef<StringAttr> manualAxes) {
  return eraseAxesFromManualComputationSharding(
      outerManualSharding, manualAxes,
      std::mem_fn(&DimensionShardingAttr::takeFrontShardingAxes));
}

// Removes manual axes from the sharding.
//
// Guaranteed by verification that all in/out shardings in a
// `ManualComputationOp` are prefixed with the manual axes. So this removes the
// prefix of manual axes (if any exist) from each dim sharding.
TensorShardingAttr eraseManualAxes(TensorShardingAttr outerManualSharding,
                                   ArrayRef<StringAttr> manualAxes) {
  if (manualAxes.empty()) {
    return outerManualSharding;
  }
  return eraseAxesFromManualComputationSharding(
      outerManualSharding, manualAxes,
      std::mem_fn(&DimensionShardingAttr::dropFrontShardingAxes));
  ;
}

// Re-adds any manual axes after the new sharding is determined across the
// `ManualComputationOp` barrier.
//
// ShardingProjection doesn't see the manual axes - it only deals with free
// axes, thus we cannot directly set in/out shardings of ManualComputation, as
// determined by ShardingProjection. We need to append them while accounting
// for the existing manual axes.
//
// Note that the dimension shardings of the result will be open/closed w.r.t.
// `newSharding`.
TensorShardingAttr addFreeAxesToManualComputationSharding(
    TensorShardingAttr outerManualSharding, TensorShardingAttr newSharding,
    ArrayRef<StringAttr> manualAxes) {
  // Remove all existing free axes first before adding possibly extra ones in
  // `newSharding`.
  TensorShardingAttr returnedSharding =
      eraseFreeAxes(outerManualSharding, manualAxes);

  SmallVector<DimensionShardingAttr> resultDimShardings(
      returnedSharding.getDimShardings());
  for (auto [resultDimSharding, newDimSharding] :
       llvm::zip(resultDimShardings, newSharding.getDimShardings())) {
    resultDimSharding = DimensionShardingAttr::get(
        resultDimSharding.getContext(),
        llvm::to_vector(llvm::concat<const AxisRefAttr>(
            resultDimSharding.getAxes(), newDimSharding.getAxes())),
        newDimSharding.getIsClosed());
  }
  return TensorShardingAttr::get(
      returnedSharding.getContext(), returnedSharding.getMeshName(),
      resultDimShardings, returnedSharding.getReplicatedAxes());
}

}  // namespace

TensorShardingAttr ManualComputationOp::getInShardingWithoutManualAxes(
    int64_t operandIndex) {
  return eraseManualAxes(getInSharding(operandIndex), getManualAxes());
}

TensorShardingAttr ManualComputationOp::getOutShardingWithoutManualAxes(
    int64_t resultIndex) {
  return eraseManualAxes(getOutSharding(resultIndex), getManualAxes());
}

void ManualComputationOp::setInShardingAddingManualAxes(
    int64_t operandIndex, TensorShardingAttr sharding) {
  setInSharding(operandIndex,
                addFreeAxesToManualComputationSharding(
                    getInSharding(operandIndex), sharding, getManualAxes()));
}

void ManualComputationOp::setOutShardingAddingManualAxes(
    int64_t resultIndex, TensorShardingAttr sharding) {
  setOutSharding(resultIndex,
                 addFreeAxesToManualComputationSharding(
                     getOutSharding(resultIndex), sharding, getManualAxes()));
}

void ManualComputationOp::setInSharding(int64_t operandIndex,
                                        TensorShardingAttr sharding) {
  setInShardingsAttr(
      getInShardings().replaceValueSharding(operandIndex, sharding));
}
void ManualComputationOp::setOutSharding(int64_t resultIndex,
                                         TensorShardingAttr sharding) {
  setOutShardingsAttr(
      getOutShardings().replaceValueSharding(resultIndex, sharding));
}

//===----------------------------------------------------------------------===//
// DataFlowEdgeOp
//===----------------------------------------------------------------------===//

DataFlowEdgeOp DataFlowEdgeOp::getDataFlowEdgeUser(Value root) {
  // We assume the input of a DataFlowEdgeOp has exactly one user.
  return dyn_cast_or_null<DataFlowEdgeOp>(
      root && root.hasOneUse() ? *root.user_begin() : nullptr);
}

}  // namespace sdy
}  // namespace mlir

#include "shardy/dialect/sdy/ir/dialect.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "shardy/dialect/sdy/ir/attrs.cc.inc"
#define GET_OP_CLASSES
#include "shardy/dialect/sdy/ir/ops.cc.inc"
