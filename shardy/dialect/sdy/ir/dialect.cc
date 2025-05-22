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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "shardy/dialect/sdy/ir/bytecode.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/enums.cc.inc"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/extensions/stablehlo_extensions.h"
#include "shardy/dialect/sdy/ir/parsers.h"   // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/printers.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

namespace {
using llvm::SmallDenseSet;

struct ShardyDialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // All non-region based ops are inlinable.
  bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final {
    return true;
  }

  // `ManualComputationOp` and `NamedComputationOp` are ops with a region, and
  // it should be allowed to be inlined into another op.
  bool isLegalToInline(Region*, Region*, bool, IRMapping&) const final {
    return true;
  }

  void handleTerminator(Operation* op, ValueRange valuesToReplace) const final {
    auto sdyReturnOp = dyn_cast<ReturnOp>(op);
    if (!sdyReturnOp) return;

    for (auto [valueToReplace, newValue] :
         llvm::zip_equal(valuesToReplace, sdyReturnOp.getOperands())) {
      valueToReplace.replaceAllUsesWith(newValue);
    }
  }
};

}  // namespace

void SdyDialect::initialize() {
  addInterface<ShardyDialectInlinerInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "shardy/dialect/sdy/ir/attrs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "shardy/dialect/sdy/ir/ops.cc.inc"
      >();
  registerStablehloExtensions(getContext());
  detail::addBytecodeInterface(this);
}

namespace details {

SmallVector<TensorShardingAttr> getOpResultEdgeOwnerShardingsImpl(
    Operation* op) {
  return llvm::to_vector(getShardings(op));
}

void setOpResultEdgeOwnerShardingsImpl(Operation* op,
                                       ArrayRef<TensorShardingAttr> shardings) {
  setShardings(op, shardings);
}

}  // namespace details

namespace {

// Gets the sources given the edge `owner`.
//
// If the owner is a `BlockArgument`, returns the corresponding operand.
// If the owner is an `OpResult`, returns the corresponding operand of the
// terminator.
// Else returns an empty vector.
template <typename RegionOpTy>
SmallVector<OpOperand*> getEdgeSourcesFromRegionBasedOp(Value owner,
                                                        RegionOpTy op) {
  static_assert(
      OpTrait::template hasSingleBlockImplicitTerminator<RegionOpTy>::value);
  assert(getOwningOp(owner) == op.getOperation());
  return TypeSwitch<Value, SmallVector<OpOperand*>>(owner)
      .Case<BlockArgument>(
          [op](BlockArgument blockArg) -> SmallVector<OpOperand*> {
            return {&op->getOpOperand(blockArg.getArgNumber())};
          })
      .template Case<OpResult>(
          [op](OpResult opResult) -> SmallVector<OpOperand*> {
            return {
                &getBodyTerminatorOpOperands(op)[opResult.getResultNumber()]};
          })
      .Default([](Value _) -> SmallVector<OpOperand*> { return {}; });
}

// Returns the edge owner given a `source`.
//
// If the `source` is an operand of a terminator, return the corresponding
// result.
// Otherwise, it should be an operand of the `op`, so return the `BlockArgument`
// with the same index.
template <typename RegionOpTy>
Value getEdgeOwnerFromSource(OpOperand& source, RegionOpTy op) {
  static_assert(
      OpTrait::template hasSingleBlockImplicitTerminator<RegionOpTy>::value);
  Operation* sourceOwner = source.getOwner();
  if (sourceOwner->hasTrait<OpTrait::IsTerminator>()) {
    return op->getResult(source.getOperandNumber());
  }
  assert(sourceOwner == op);
  return op->getOperand(source.getOperandNumber());
}

}  // namespace

//===----------------------------------------------------------------------===//
// ShardableDataFlowOpInterface
//===----------------------------------------------------------------------===//

TensorShardingAttr
ShardableDataFlowOpInterface::getBlockArgumentEdgeOwnerSharding(
    unsigned index) {
  if (SmallVector<TensorShardingAttr> argShardings =
          getBlockArgumentEdgeOwnerShardings();
      !argShardings.empty()) {
    return argShardings[index];
  }
  return nullptr;
}

TensorShardingAttr ShardableDataFlowOpInterface::getOpResultEdgeOwnerSharding(
    unsigned index) {
  if (SmallVector<TensorShardingAttr> resultShardings =
          getOpResultEdgeOwnerShardings();
      !resultShardings.empty()) {
    return resultShardings[index];
  }
  return nullptr;
}

TensorShardingAttr ShardableDataFlowOpInterface::getEdgeOwnerSharding(
    Value owner) {
  if (auto blockArg = dyn_cast<BlockArgument>(owner)) {
    return getBlockArgumentEdgeOwnerSharding(blockArg.getArgNumber());
  }
  return getOpResultEdgeOwnerSharding(cast<OpResult>(owner).getResultNumber());
}

void ShardableDataFlowOpInterface::setBlockArgumentEdgeOwnerSharding(
    unsigned index, TensorShardingAttr sharding) {
  SmallVector<TensorShardingAttr> shardings;
  if (SmallVector<TensorShardingAttr> ownerShardings =
          getBlockArgumentEdgeOwnerShardings();
      !ownerShardings.empty()) {
    shardings = llvm::to_vector(ownerShardings);
    shardings[index] = sharding;
  } else {
    shardings = getOpenShardingsWithShardingAtIndex(
        getContext(),
        ValueTypeRange<ArrayRef<BlockArgument>>(getBlockArgumentEdgeOwners()),
        index, sharding);
  }
  setBlockArgumentEdgeOwnerShardings(shardings);
}

void ShardableDataFlowOpInterface::setOpResultEdgeOwnerSharding(
    unsigned index, TensorShardingAttr sharding) {
  SmallVector<TensorShardingAttr> shardings;
  if (SmallVector<TensorShardingAttr> ownerShardings =
          getOpResultEdgeOwnerShardings();
      !ownerShardings.empty()) {
    shardings = llvm::to_vector(ownerShardings);
    shardings[index] = sharding;
  } else {
    shardings = getOpenShardingsWithShardingAtIndex(
        getContext(), getOpResultEdgeOwners().getTypes(), index, sharding);
  }
  setOpResultEdgeOwnerShardings(shardings);
}

void ShardableDataFlowOpInterface::setEdgeOwnerSharding(
    Value owner, TensorShardingAttr sharding) {
  if (auto blockArg = dyn_cast<BlockArgument>(owner)) {
    setBlockArgumentEdgeOwnerSharding(blockArg.getArgNumber(), sharding);
  } else {
    setOpResultEdgeOwnerSharding(cast<OpResult>(owner).getResultNumber(),
                                 sharding);
  }
}

//===----------------------------------------------------------------------===//
// MeshAttr
//===----------------------------------------------------------------------===//

bool MeshAttr::empty() const {
  return getAxes().empty() && getDeviceIds().empty();
}

bool MeshAttr::hasAxis(StringRef axisName) const {
  return llvm::any_of(getAxes(), [axisName](MeshAxisAttr axis) {
    return axis.getName() == axisName;
  });
}

int64_t MeshAttr::getAxisSize(StringRef axisName) const {
  for (MeshAxisAttr meshAxis : getAxes()) {
    if (meshAxis.getName() == axisName) {
      return meshAxis.getSize();
    }
  }
  // Since verification will fail if an axis name doesn't appear in the bound
  // mesh, we can assume we would never get here.
  llvm::report_fatal_error("unknown axis name");
}

int64_t MeshAttr::getTotalSize() const {
  return getTotalAxesSize(getAxes());
}

bool MeshAttr::isMaximal(int64_t deviceId) const {
  return isMaximal() && getMaximalDeviceId() == deviceId;
}

bool MeshAttr::isMaximal() const {
  return getAxes().empty() && getDeviceIds().size() == 1;
}

std::optional<int64_t> MeshAttr::getMaximalDeviceId() const {
  if (isMaximal()) {
    return getDeviceIds().front();
  }
  return std::nullopt;
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

llvm::SmallDenseMap<StringRef, int64_t> MeshAttr::getAxisNameToSize() const {
  llvm::SmallDenseMap<StringRef, int64_t> axisNameToSize;
  ArrayRef<MeshAxisAttr> axes = getAxes();
  axisNameToSize.reserve(axes.size());
  for (MeshAxisAttr axis : axes) {
    axisNameToSize[axis.getName()] = axis.getSize();
  }
  return axisNameToSize;
}

bool MeshAttr::equals(MeshAttr other, const bool ignoreDeviceIds) const {
  if (!ignoreDeviceIds) {
    return *this == other;
  }
  if (this->isMaximal() != other.isMaximal()) {
    return false;
  }
  return this->getAxes() == other.getAxes();
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
      // Both axis-refs have the same name, defer to AxisRefAttr::operator<
      return lhs < rhs;
    }

    return mesh.getAxisNameComparator()(lhsName, rhsName);
  };
}

bool AxisRefAttr::operator<(const AxisRefAttr& rhs) const {
  StringRef name = getName();
  StringRef rhsName = rhs.getName();
  if (name != rhsName) {
    return name < rhsName;
  }
  // Both axis-refs have the same name
  if (!getSubAxisInfo()) {
    // This is the full axis, it's smaller than `rhs` iff `rhs` is a sub-axis
    // with pre-size > 1.
    return rhs.getSubAxisPreSize() > 1;
  }
  if (!rhs.getSubAxisInfo()) {
    // This is a sub-axis and `rhs` is the full axis, this is smaller iff its
    // pre-size is 1.
    return getSubAxisPreSize() == 1;
  }
  // Both axis-refs are sub-axes.
  return getSubAxisInfo() < rhs.getSubAxisInfo();
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

int64_t AxisRefAttr::getNextPreSizeOrFullSize(MeshAttr mesh) const {
  return getSubAxisInfo() ? getSubAxisInfo().getNextPreSize() : getSize(mesh);
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
  return *this != other && contains(other);
}

bool AxisRefAttr::prefixOf(AxisRefAttr other) const {
  return other.contains(*this) &&
         getSubAxisPreSize() == other.getSubAxisPreSize();
}

bool AxisRefAttr::strictPrefixOf(AxisRefAttr other) const {
  return *this != other && prefixOf(other);
}

bool AxisRefAttr::suffixOf(AxisRefAttr other, MeshAttr mesh) const {
  return other.contains(*this) &&
         getNextPreSizeOrFullSize(mesh) == other.getNextPreSizeOrFullSize(mesh);
}

bool AxisRefAttr::strictSuffixOf(AxisRefAttr other, MeshAttr mesh) const {
  return *this != other && suffixOf(other, mesh);
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

namespace {

bool canSubAxesCoexist(int64_t minPreSize, int64_t maxPreSize,
                       int64_t minNextPreSize, int64_t maxNextPreSize) {
  if (minNextPreSize > maxPreSize) {
    // Sub-axes overlap, check if overlapping and non-overlapping parts are
    // valid.
    return minNextPreSize % maxPreSize == 0 && maxPreSize % minPreSize == 0 &&
           maxNextPreSize % minNextPreSize == 0;
  }
  // Sub-axes don't overlap, check if the gap is valid.
  return maxPreSize % minNextPreSize == 0;
}

}  // namespace

bool AxisRefAttr::canCoexist(AxisRefAttr other) const {
  if (getName() != other.getName()) {
    return true;
  }
  SubAxisInfoAttr thisSubAxisInfo = getSubAxisInfo();
  SubAxisInfoAttr otherSubAxisInfo = other.getSubAxisInfo();

  if (!thisSubAxisInfo || !otherSubAxisInfo) {
    // One of the axes is full
    return true;
  }

  int64_t thisPreSize = thisSubAxisInfo.getPreSize();
  int64_t otherPreSize = otherSubAxisInfo.getPreSize();
  int64_t thisNextPreSize = thisSubAxisInfo.getNextPreSize();
  int64_t otherNextPreSize = otherSubAxisInfo.getNextPreSize();

  auto [minPreSize, maxPreSize] = std::minmax(thisPreSize, otherPreSize);
  auto [minNextPreSize, maxNextPreSize] =
      std::minmax(thisNextPreSize, otherNextPreSize);

  return canSubAxesCoexist(minPreSize, maxPreSize, minNextPreSize,
                           maxNextPreSize);
}

std::optional<AxisRefAttr> AxisRefAttr::getOverlap(AxisRefAttr other) const {
  if (other.getName() != getName()) {
    return std::nullopt;
  }

  SubAxisInfoAttr thisSubAxisInfo = getSubAxisInfo();
  SubAxisInfoAttr otherSubAxisInfo = other.getSubAxisInfo();

  if (!thisSubAxisInfo) {
    // This is a full axis.
    return other;
  }

  if (!otherSubAxisInfo) {
    // Other is a full axis.
    return *this;
  }

  int64_t thisPreSize = thisSubAxisInfo.getPreSize();
  int64_t otherPreSize = otherSubAxisInfo.getPreSize();
  int64_t thisNextPreSize = thisSubAxisInfo.getNextPreSize();
  int64_t otherNextPreSize = otherSubAxisInfo.getNextPreSize();

  auto [minPreSize, maxPreSize] = std::minmax(thisPreSize, otherPreSize);
  auto [minNextPreSize, maxNextPreSize] =
      std::minmax(thisNextPreSize, otherNextPreSize);

  if (minNextPreSize <= maxPreSize ||
      !canSubAxesCoexist(minPreSize, maxPreSize, minNextPreSize,
                         maxNextPreSize)) {
    // No overlap or can't co-exist.
    return std::nullopt;
  }

  return AxisRefAttr::get(getContext(), getName(), maxPreSize,
                          minNextPreSize / maxPreSize);
}

std::optional<AxisRefAttr> AxisRefAttr::getPrefixWithoutOverlap(
    AxisRefAttr other) const {
  if (!canCoexist(other)) {
    return std::nullopt;
  }
  if (!overlaps(other)) {
    return *this;
  }

  int64_t thisPreSize = getSubAxisPreSize();
  int64_t otherPreSize = other.getSubAxisPreSize();

  if (thisPreSize >= otherPreSize) {
    return std::nullopt;
  }
  return AxisRefAttr::get(getContext(), getName(), thisPreSize,
                          otherPreSize / thisPreSize);
}

std::optional<AxisRefAttr> AxisRefAttr::getSuffixWithoutOverlap(
    AxisRefAttr other, MeshAttr mesh) const {
  if (!canCoexist(other)) {
    return std::nullopt;
  }
  if (!overlaps(other)) {
    return *this;
  }

  int64_t thisNextPreSize = getNextPreSizeOrFullSize(mesh);
  int64_t otherNextPreSize = other.getNextPreSizeOrFullSize(mesh);
  if (thisNextPreSize <= otherNextPreSize) {
    return std::nullopt;
  }
  return AxisRefAttr::get(getContext(), getName(), otherNextPreSize,
                          thisNextPreSize / otherNextPreSize);
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
  return AxisRefAttr::get(getContext(), getName(), preSize, size);
}

std::optional<AxisRefAttr> AxisRefAttr::getGreatestCommonPrefix(
    AxisRefAttr other) const {
  if (!canCoexist(other)) {
    return std::nullopt;
  }
  if (prefixOf(other)) {
    return *this;
  }
  if (other.prefixOf(*this)) {
    return other;
  }
  return std::nullopt;
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
                                    getIsClosed(), getPriority());
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

DimensionShardingAttr DimensionShardingAttr::getClosedLike(
    DimensionShardingAttr dimSharding) {
  return DimensionShardingAttr::get(dimSharding.getContext(),
                                    dimSharding.getAxes(), /*isClosed=*/true,
                                    /*priority=*/dimSharding.getPriority());
}

//===----------------------------------------------------------------------===//
// TensorShardingAttr
//===----------------------------------------------------------------------===//

// Creates fully open or closed tensor sharding attr.
TensorShardingAttr TensorShardingAttr::getFullyReplicated(MLIRContext* context,
                                                          int64_t rank,
                                                          Attribute meshOrRef,
                                                          bool isClosed) {
  return TensorShardingAttr::get(
      context, meshOrRef,
      /*dimShardings=*/
      SmallVector<DimensionShardingAttr>(
          rank, DimensionShardingAttr::get(context, {}, isClosed)),
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});
}

// Creates fully open or closed tensor sharding attr.
TensorShardingAttr TensorShardingAttr::getFullyReplicated(MLIRContext* context,
                                                          int64_t rank,
                                                          StringRef meshName,
                                                          bool isClosed) {
  return getFullyReplicated(
      context, rank, FlatSymbolRefAttr::get(context, meshName), isClosed);
}

MeshAttr TensorShardingAttr::getMesh(const SymbolTable& symbolTable) const {
  return getMeshOrLookup(symbolTable, getMeshOrRef());
}

MeshAttr TensorShardingAttr::getMesh(Operation* op) const {
  return getMeshOrLookup(op, getMeshOrRef());
}

bool TensorShardingAttr::emptyAxes() const {
  return getReplicatedAxes().empty() && getUnreducedAxes().empty() &&
         llvm::all_of(getDimShardings(),
                      [](const DimensionShardingAttr& dimSharding) {
                        return dimSharding.emptyAxes();
                      });
}

bool TensorShardingAttr::anyOfAxisRef(
    std::function<bool(AxisRefAttr)> predicate) const {
  return anyOfDimShardingOrReplicatedAxis(predicate) ||
         llvm::any_of(getUnreducedAxes(), predicate);
}

bool TensorShardingAttr::anyOfDimShardingOrReplicatedAxis(
    std::function<bool(AxisRefAttr)> predicate) const {
  for (DimensionShardingAttr dimSharding : getDimShardings()) {
    if (llvm::any_of(dimSharding.getAxes(), predicate)) {
      return true;
    }
  }
  return llvm::any_of(getReplicatedAxes(), predicate);
}

void TensorShardingAttr::forEachAxisRef(
    std::function<void(AxisRefAttr)> callback) const {
  for (DimensionShardingAttr dimSharding : getDimShardings()) {
    llvm::for_each(dimSharding.getAxes(), callback);
  }
  llvm::for_each(getReplicatedAxes(), callback);
  llvm::for_each(getUnreducedAxes(), callback);
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
  return TensorShardingAttr::get(getContext(), getMeshOrRef(), dimShardings,
                                 getReplicatedAxes(), getUnreducedAxes());
}

TensorShardingAttr TensorShardingAttr::openShardingDims(
    ArrayRef<int64_t> dimIndices) const {
  SmallVector<DimensionShardingAttr> dimShardings(getDimShardings().begin(),
                                                  getDimShardings().end());
  for (int64_t dim : dimIndices) {
    dimShardings[dim] = DimensionShardingAttr::get(
        getContext(), dimShardings[dim].getAxes(), /*isClosed=*/false);
  }
  return TensorShardingAttr::get(getContext(), getMeshOrRef(), dimShardings,
                                 getReplicatedAxes(), getUnreducedAxes());
}

TensorShardingAttr TensorShardingAttr::replaceDimSharding(
    int64_t dim, DimensionShardingAttr sharding) const {
  SmallVector<DimensionShardingAttr> shardings(getDimShardings());
  shardings[dim] = sharding;
  return TensorShardingAttr::get(getContext(), getMeshOrRef(), shardings,
                                 getReplicatedAxes(), getUnreducedAxes());
}

TensorShardingAttr TensorShardingAttr::replaceReplicatedAxes(
    ArrayRef<AxisRefAttr> replicatedAxes) const {
  return TensorShardingAttr::get(getContext(), getMeshOrRef(),
                                 getDimShardings(), replicatedAxes,
                                 getUnreducedAxes());
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

  return TensorShardingAttr::get(getContext(), getMeshOrRef(),
                                 getDimShardings(), newReplicatedAxes,
                                 getUnreducedAxes());
}

TensorShardingAttr TensorShardingAttr::getFullyClosed(MLIRContext* context,
                                                      int64_t rank,
                                                      Attribute meshOrRef) {
  return getFullyReplicated(context, rank, meshOrRef, /*isClosed=*/true);
}

TensorShardingAttr TensorShardingAttr::getFullyClosed(MLIRContext* context,
                                                      int64_t rank,
                                                      StringRef meshName) {
  return getFullyReplicated(context, rank, meshName, /*isClosed=*/true);
}

TensorShardingAttr TensorShardingAttr::getFullyClosedLike(
    TensorShardingAttr sharding) {
  return getFullyReplicated(sharding.getContext(), sharding.getRank(),
                            sharding.getMeshOrRef(), /*isClosed=*/true);
}

TensorShardingAttr TensorShardingAttr::getClosedLike(
    TensorShardingAttr sharding) {
  SmallVector<DimensionShardingAttr> closedDimShardings(sharding.getRank());
  for (int index = 0; index < sharding.getRank(); index++) {
    closedDimShardings[index] =
        DimensionShardingAttr::getClosedLike(sharding.getDimSharding(index));
  }
  return TensorShardingAttr::get(sharding.getContext(), sharding.getMeshOrRef(),
                                 /*dimShardings=*/closedDimShardings,
                                 /*replicatedAxes=*/{}, /*unreducedAxes=*/{});
}

TensorShardingAttr TensorShardingAttr::getClosed(
    MLIRContext* context, Attribute meshOrRef,
    ArrayRef<SmallVector<AxisRefAttr>> axesPerDim) {
  SmallVector<DimensionShardingAttr> dimShardings;
  dimShardings.reserve(axesPerDim.size());
  for (ArrayRef<AxisRefAttr> axes : axesPerDim) {
    dimShardings.push_back(
        DimensionShardingAttr::get(context, axes, /*is_closed=*/true));
  }
  return TensorShardingAttr::get(context, meshOrRef, dimShardings,
                                 /*replicatedAxes=*/{}, /*unreducedAxes=*/{});
}

TensorShardingAttr TensorShardingAttr::getFullyOpen(MLIRContext* context,
                                                    int64_t rank,
                                                    Attribute meshOrRef) {
  return getFullyReplicated(context, rank, meshOrRef, /*isClosed=*/false);
}

TensorShardingAttr TensorShardingAttr::getFullyOpen(MLIRContext* context,
                                                    int64_t rank,
                                                    StringRef meshName) {
  return getFullyReplicated(context, rank, meshName, /*isClosed=*/false);
}

TensorShardingAttr TensorShardingAttr::getFullyOpenLike(
    TensorShardingAttr sharding) {
  return getFullyReplicated(sharding.getContext(), sharding.getRank(),
                            sharding.getMeshOrRef(), /*isClosed=*/false);
}

RankedTensorType TensorShardingAttr::getLocalTensorType(
    RankedTensorType globalTensorType, MeshAttr mesh,
    bool allowNonDivisible) const {
  if (getDimShardings().empty()) {
    return globalTensorType;
  }
  SmallVector<int64_t> localShape;
  localShape.reserve(globalTensorType.getRank());

  for (auto [globalDimSize, dimSharding] :
       llvm::zip_equal(globalTensorType.getShape(), getDimShardings())) {
    if (ShapedType::isDynamic(globalDimSize)) {
      localShape.push_back(globalDimSize);
    } else {
      int64_t shardSize = dimSharding.getShardedSize(mesh);
      if (!allowNonDivisible && globalDimSize % shardSize != 0) {
        return nullptr;
      }
      // We allow non divisible sharding.
      int64_t localSize = (globalDimSize + shardSize - 1) / shardSize;
      localShape.push_back(localSize);
    }
  }
  return RankedTensorType::get(ArrayRef<int64_t>(localShape),
                               globalTensorType.getElementType());
}

RankedTensorType TensorShardingAttr::getGlobalTensorType(
    RankedTensorType localTensorType, MeshAttr mesh) const {
  if (getDimShardings().empty()) {
    return localTensorType;
  }
  SmallVector<int64_t> globalShape;
  globalShape.reserve(localTensorType.getRank());

  for (auto [localDimSize, dimSharding] :
       llvm::zip_equal(localTensorType.getShape(), getDimShardings())) {
    if (ShapedType::isDynamic(localDimSize)) {
      globalShape.push_back(localDimSize);
    } else {
      globalShape.push_back(dimSharding.getShardedSize(mesh) * localDimSize);
    }
  }
  return RankedTensorType::get(ArrayRef<int64_t>(globalShape),
                               localTensorType.getElementType());
}

bool TensorShardingAttr::areDimAxesEqual(
    TensorShardingAttr otherSharding) const {
  ArrayRef<DimensionShardingAttr> left = getDimShardings();
  ArrayRef<DimensionShardingAttr> right = otherSharding.getDimShardings();
  return left.size() == right.size() &&
         llvm::all_of(llvm::zip_equal(left, right), [](auto&& pair) {
           return std::get<0>(pair).getAxes() == std::get<1>(pair).getAxes();
         });
}

//===----------------------------------------------------------------------===//
// TensorShardingPerValueAttr
//===----------------------------------------------------------------------===//

TensorShardingPerValueAttr TensorShardingPerValueAttr::getFullyOpen(
    MLIRContext* context, TypeRange types, StringRef meshName) {
  return TensorShardingPerValueAttr::get(
      context, getFullyOpenShardings(context, types, meshName));
}

TensorShardingPerValueAttr TensorShardingPerValueAttr::getFullyClosed(
    MLIRContext* context, TypeRange types, StringRef meshName) {
  return TensorShardingPerValueAttr::get(
      context, getFullyClosedShardings(context, types, meshName));
}

TensorShardingPerValueAttr
TensorShardingPerValueAttr::getOpenWithShardingAtIndex(
    MLIRContext* context, TypeRange types, int64_t index,
    TensorShardingAttr sharding) {
  return TensorShardingPerValueAttr::get(
      context,
      getOpenShardingsWithShardingAtIndex(context, types, index, sharding));
}

TensorShardingPerValueAttr TensorShardingPerValueAttr::replaceValueSharding(
    int64_t index, TensorShardingAttr sharding) const {
  if (getSharding(index) == sharding) {
    return *this;
  }
  SmallVector<TensorShardingAttr> shardings(getShardings());
  shardings[index] = sharding;
  return TensorShardingPerValueAttr::get(getContext(), shardings);
}

//===----------------------------------------------------------------------===//
// TensorMappingAttr
//===----------------------------------------------------------------------===//

bool TensorMappingAttr::containsFactor(int64_t factorIndex) const {
  return llvm::any_of(
      getDimMappings(), [factorIndex](DimMappingAttr dimMapping) {
        return llvm::is_contained(dimMapping.getFactorIndices(), factorIndex);
      });
}

//===----------------------------------------------------------------------===//
// OpShardingRuleAttr
//===----------------------------------------------------------------------===//

// TODO(enver): Instead use ShapedType::getNumElements, as the factors might not
// be the exact size of the dim, e.g. concat.
SmallVector<int64_t> OpShardingRuleAttr::getTensorSizes() const {
  SmallVector<int64_t> tensorSizes;
  tensorSizes.reserve(getNumOperands() + getNumResults());
  for (const TensorMappingAttr& tensorMapping :
       llvm::concat<const TensorMappingAttr>(getOperandMappings(),
                                             getResultMappings())) {
    int64_t tensorSize = 1;
    for (DimMappingAttr dimMapping : tensorMapping.getDimMappings()) {
      for (int64_t factorIndex : dimMapping.getFactorIndices()) {
        tensorSize *= getFactorSize(factorIndex);
      }
    }
    tensorSizes.push_back(tensorSize);
  }
  return tensorSizes;
}

FactorType OpShardingRuleAttr::getFactorType(int64_t factorIndex) const {
  if (isReductionFactor(factorIndex)) {
    return FactorType::kReduction;
  }
  if (isNeedReplicationFactor(factorIndex)) {
    return FactorType::kNeedReplication;
  }
  if (isPermutationFactor(factorIndex)) {
    return FactorType::kPermutation;
  }
  return FactorType::kPassThrough;
}

bool OpShardingRuleAttr::isPassThroughFactor(int64_t factorIndex) const {
  return getFactorType(factorIndex) == FactorType::kPassThrough;
}

bool OpShardingRuleAttr::isReductionFactor(int64_t factorIndex) const {
  return llvm::is_contained(getReductionFactors(), factorIndex);
}

bool OpShardingRuleAttr::isNeedReplicationFactor(int64_t factorIndex) const {
  return llvm::is_contained(getNeedReplicationFactors(), factorIndex);
}

bool OpShardingRuleAttr::isPermutationFactor(int64_t factorIndex) const {
  return llvm::is_contained(getPermutationFactors(), factorIndex);
}

bool OpShardingRuleAttr::isBlockedPropagationFactor(int64_t factorIndex) const {
  return llvm::is_contained(getBlockedPropagationFactors(), factorIndex);
}

bool OpShardingRuleAttr::isFactorInAllNonScalarTensors(
    int64_t factorIndex) const {
  for (const TensorMappingAttr& tensorMapping :
       llvm::concat<const TensorMappingAttr>(getOperandMappings(),
                                             getResultMappings())) {
    if (tensorMapping.empty()) {
      // We do not consider scalar tensors.
      continue;
    }
    if (!tensorMapping.containsFactor(factorIndex)) {
      return false;
    }
  }
  return true;
}

bool OpShardingRuleAttr::isBatchingFactor(int64_t factorIndex) const {
  return isPassThroughFactor(factorIndex) &&
         isFactorInAllNonScalarTensors(factorIndex);
}

SmallVector<int64_t> OpShardingRuleAttr::getNonScalarTensorIndices() const {
  SmallVector<int64_t> nonScalarTensorIndices;

  for (auto [tensorIndex, tensorMapping] :
       llvm::enumerate(llvm::concat<const TensorMappingAttr>(
           getOperandMappings(), getResultMappings()))) {
    if (!tensorMapping.empty()) {
      nonScalarTensorIndices.push_back(tensorIndex);
    }
  }
  return nonScalarTensorIndices;
}

// TODO(enver): Consider returning a BitVector for batching factors.
SmallVector<int64_t> OpShardingRuleAttr::getBatchingFactors() const {
  SmallVector<int64_t> factorIndices;
  factorIndices.reserve(getNumFactors());
  for (int64_t index = 0; index < getNumFactors(); index++) {
    if (isBatchingFactor(index)) {
      factorIndices.push_back(index);
    }
  }
  return factorIndices;
}

bool OpShardingRuleAttr::hasDimensionsWithMultipleFactors() const {
  for (const TensorMappingAttr& tensorMapping :
       llvm::concat<const TensorMappingAttr>(getOperandMappings(),
                                             getResultMappings())) {
    for (DimMappingAttr dimMapping : tensorMapping.getDimMappings()) {
      if (dimMapping.getFactorIndices().size() > 1) {
        return true;
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// AllToAllParamList
//===----------------------------------------------------------------------===//

bool AllToAllParamListAttr::overlaps(AllToAllParamListAttr other) const {
  SmallDenseSet<int64_t> seenDims;
  seenDims.reserve(getValue().size() + other.getValue().size());
  for (AllToAllParamAttr param :
       llvm::concat<const AllToAllParamAttr>(getValue(), other.getValue())) {
    for (int64_t dim : {param.getSrcDim(), param.getTgtDim()}) {
      if (!seenDims.insert(dim).second) {
        return true;
      }
    }
  }
  return false;
}

AllToAllParamListAttr AllToAllParamListAttr::combineAndSort(
    AllToAllParamListAttr other) const {
  SmallVector<AllToAllParamAttr> combinedParams;
  combinedParams.reserve(size() + other.size());
  combinedParams.append(begin(), end());
  combinedParams.append(other.begin(), other.end());
  llvm::sort(combinedParams,
             [](const AllToAllParamAttr& a, const AllToAllParamAttr& b) {
               return a.getSrcDim() < b.getSrcDim();
             });
  return AllToAllParamListAttr::get(getContext(), combinedParams);
}

//===----------------------------------------------------------------------===//
// ManualComputationOp
//===----------------------------------------------------------------------===//

namespace {

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
        newDimSharding.getIsClosed(), newDimSharding.getPriority());
  }
  return TensorShardingAttr::get(newSharding.getContext(),
                                 newSharding.getMeshOrRef(), resultDimShardings,
                                 outerManualSharding.getReplicatedAxes(),
                                 outerManualSharding.getUnreducedAxes());
}

}  // namespace

void ManualComputationOp::setInShardings(
    ArrayRef<TensorShardingAttr> shardings) {
  setInShardingsAttr(TensorShardingPerValueAttr::get(getContext(), shardings));
}

void ManualComputationOp::setOutShardings(
    ArrayRef<TensorShardingAttr> shardings) {
  setOutShardingsAttr(TensorShardingPerValueAttr::get(getContext(), shardings));
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

SmallVector<TensorShardingAttr>
ManualComputationOp::getBlockArgumentEdgeOwnerShardings() {
  SmallVector<TensorShardingAttr> shardings;
  shardings.reserve(getInShardings().size());
  for (TensorShardingAttr sharding : getInShardings().getShardings()) {
    shardings.push_back(eraseManualAxes(sharding, getManualAxes()));
  }
  return shardings;
}

void ManualComputationOp::setBlockArgumentEdgeOwnerShardings(
    ArrayRef<TensorShardingAttr> shardings) {
  // TODO(bartchr): see if we can use a `to_vector`+`map_iterator` here.
  ArrayRef<StringAttr> manualAxes = getManualAxes();
  SmallVector<TensorShardingAttr> shardingsWithManualAxes;
  shardingsWithManualAxes.reserve(shardings.size());
  for (auto [i, sharding] : llvm::enumerate(shardings)) {
    shardingsWithManualAxes.push_back(addFreeAxesToManualComputationSharding(
        getInSharding(i), sharding, manualAxes));
  }
  setInShardings(shardingsWithManualAxes);
}

SmallVector<TensorShardingAttr>
ManualComputationOp::getOpResultEdgeOwnerShardings() {
  return llvm::to_vector(getOutShardings().getShardings());
}

void ManualComputationOp::setOpResultEdgeOwnerShardings(
    ArrayRef<TensorShardingAttr> shardings) {
  setOutShardings(shardings);
}

// Transforms the `sharding` of the target depending on `transformType`.
//
// 1) `transformType` == `kBeforeEdgePropagation`:
//   a) If the target is a block argument:
//       - add manual axes to the sharding.
//   b) If the target is a result:
//       - remove manual axes from the sharding.
// 2) `transformType` == `kAfterEdgePropagation`:
//   a) If the target is a block argument:
//       - remove manual axes from the sharding.
//   b) If the target is a result:
//       - add manual axes to the sharding.
TensorShardingAttr ManualComputationOp::transformTargetSharding(
    Value target, TensorShardingAttr sharding,
    DataFlowShardingTransformType transformType) {
  switch (transformType) {
    case DataFlowShardingTransformType::kBeforeEdgePropagation: {
      if (auto blockArg = dyn_cast<BlockArgument>(target)) {
        return addFreeAxesToManualComputationSharding(
            getInSharding(blockArg.getArgNumber()), sharding, getManualAxes());
      }
      return eraseManualAxes(sharding, getManualAxes());
    }
    case DataFlowShardingTransformType::kAfterEdgePropagation: {
      if (isa<BlockArgument>(target)) {
        return eraseManualAxes(sharding, getManualAxes());
      }
      return addFreeAxesToManualComputationSharding(
          getOutSharding(cast<OpResult>(target).getResultNumber()), sharding,
          getManualAxes());
    }
  }
  llvm_unreachable("received an unexpected target type.");
  return nullptr;
}

ArrayRef<BlockArgument> ManualComputationOp::getBlockArgumentEdgeOwners() {
  return getBody().getArguments();
}

ResultRange ManualComputationOp::getOpResultEdgeOwners() {
  return getResults();
}

// Gets the sources given the edge `owner`.
//
// Note that the return value is a vector, for `ManualComputationOp`s there can
// only be one value but sdy's interface expects a vector.
//
// For example, given the following:
// ```
// %r = sdy.manual_computation ...attributes... (%operand0) (%arg0)
//   %a = tanh(%arg0)
//   sdy.return %a
// }
// ```
// If the owner is a block argument (e.g., `%operand0`), return `%arg0`.
// If the owner is a result (e.g., `%r`), return `%a`.
SmallVector<OpOperand*> ManualComputationOp::getEdgeSources(Value owner) {
  return getEdgeSourcesFromRegionBasedOp(owner, *this);
}

// Returns the edge owner value given a `target`.
//
// For `NamedComputationOp`s, there is only one target per data flow edge which
// is also the edge owner.
Value ManualComputationOp::getEdgeOwnerFromTarget(Value target) {
  assert(getOwningOp(target) == getOperation());
  return target;
}

// Returns the edge owner given a `source`.
//
// If the `source` is an operand of a terminator, return the corresponding
// result. Otherwise it should be an operand of the `ManualComputationOp`,
// return the `BlockArgument` with the same index.
Value ManualComputationOp::getEdgeOwnerFromSource(OpOperand& source) {
  return sdy::getEdgeOwnerFromSource(source, *this);
}

bool ManualComputationOp::shouldKeepEdgeOwnerShardingsDivisible() {
  return true;
}

//===----------------------------------------------------------------------===//
// ShardingGroupOp
//===----------------------------------------------------------------------===//

LogicalResult ShardingGroupOp::inferReturnTypes(MLIRContext*,
                                                std::optional<Location>,
                                                ValueRange, DictionaryAttr,
                                                OpaqueProperties, RegionRange,
                                                SmallVectorImpl<Type>&) {
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

bool ConstantOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  return stablehlo::ConstantOp::isCompatibleReturnTypes(l, r);
}

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ConstantOpAdaptor adaptor(operands, attributes, properties);
  inferredReturnTypes.push_back(adaptor.getValue().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// DataFlowEdgeOp
//===----------------------------------------------------------------------===//

namespace {

// Returns the owning op if it is a `ShardableDataFlowOpInterface`, otherwise
// returns nullptr.
ShardableDataFlowOpInterface dynCastOwningShardableDataFlowOp(Value value) {
  return dyn_cast<ShardableDataFlowOpInterface>(getOwningOp(value));
}

// Returns the owning `ShardableDataFlowOpInterface` (assuming it exists).
ShardableDataFlowOpInterface castOwningShardableDataFlowOp(Value value) {
  ShardableDataFlowOpInterface shardableDataFlowOp =
      dynCastOwningShardableDataFlowOp(value);
  assert(shardableDataFlowOp);
  return shardableDataFlowOp;
}

// If `root` has a single use which is by a `DataFlowEdgeOp`, returns that
// `DataFlowEdgeOp`, otherwise returns `nullptr`.
DataFlowEdgeOp getDataFlowEdgeUser(Value owner) {
  // We assume the input of a DataFlowEdgeOp has exactly one user.
  return dyn_cast_or_null<DataFlowEdgeOp>(
      owner && owner.hasOneUse() ? *owner.user_begin() : nullptr);
}

}  // namespace

DataFlowEdgeOp DataFlowEdgeOp::lookup(Value target) {
  if (ShardableDataFlowOpInterface shardableDataFlowOp =
          dynCastOwningShardableDataFlowOp(target)) {
    return getDataFlowEdgeUser(
        shardableDataFlowOp.getEdgeOwnerFromTarget(target));
  }
  return nullptr;
}

DataFlowEdgeOp DataFlowEdgeOp::lookup(OpOperand& source) {
  Operation* op = source.getOwner();
  op = op->hasTrait<OpTrait::IsTerminator>() ? op->getParentOp() : op;
  if (auto shardableDataFlowOp = dyn_cast<ShardableDataFlowOpInterface>(op)) {
    return getDataFlowEdgeUser(
        shardableDataFlowOp.getEdgeOwnerFromSource(source));
  }
  return nullptr;
}

TensorShardingAttr DataFlowEdgeOp::transformTargetSharding(
    TensorShardingAttr sharding, DataFlowShardingTransformType transformType) {
  return castOwningShardableDataFlowOp(getInput())
      .transformTargetSharding(getInput(), sharding, transformType);
}

SmallVector<Value> DataFlowEdgeOp::getSources() {
  SmallVector<Value> sources;
  for (OpOperand* opOperand :
       castOwningShardableDataFlowOp(getInput()).getEdgeSources(getInput())) {
    sources.push_back(opOperand->get());
  }
  return sources;
}

SmallVector<Value> DataFlowEdgeOp::getNonOwnerTargets() {
  return castOwningShardableDataFlowOp(getInput())
      .getNonEdgeOwnerTargets(getInput());
}

//===----------------------------------------------------------------------===//
// NamedComputationOp
//===----------------------------------------------------------------------===//

void NamedComputationOp::setOpResultEdgeOwnerShardings(
    ArrayRef<TensorShardingAttr> shardings) {
  setOutShardingsAttr(TensorShardingPerValueAttr::get(getContext(), shardings));
}

SmallVector<TensorShardingAttr>
NamedComputationOp::getBlockArgumentEdgeOwnerShardings() {
  if (std::optional<TensorShardingPerValueAttr> inShardings =
          getInShardings()) {
    return llvm::to_vector(inShardings->getShardings());
  }
  return {};
}

SmallVector<TensorShardingAttr>
NamedComputationOp::getOpResultEdgeOwnerShardings() {
  if (std::optional<TensorShardingPerValueAttr> outShardings =
          getOutShardings()) {
    return llvm::to_vector(outShardings->getShardings());
  }
  return {};
}

void NamedComputationOp::setBlockArgumentEdgeOwnerShardings(
    ArrayRef<TensorShardingAttr> shardings) {
  setInShardingsAttr(TensorShardingPerValueAttr::get(getContext(), shardings));
}

ArrayRef<BlockArgument> NamedComputationOp::getBlockArgumentEdgeOwners() {
  return getBody().getArguments();
}

ResultRange NamedComputationOp::getOpResultEdgeOwners() { return getResults(); }

// Gets the sources given the edge `owner`.
//
// Note that the return value is a vector, for `NamedComputationOp`s there can
// only be one value but sdy's interface expects a vector.
//
// For example, given the following:
// ```
// %r = sdy.named_computation<"my_tan">(%operand0) (%arg0)
//   %a = tanh(%arg0)
//   sdy.return %a
// }
// ```
// If the owner is a block argument (e.g., `%operand0`), return `%arg0`.
// If the owner is a result (e.g., `%r`), return `%a`.
SmallVector<OpOperand*> NamedComputationOp::getEdgeSources(Value owner) {
  return getEdgeSourcesFromRegionBasedOp(owner, *this);
}

// Returns the edge owner value given a `target`.
//
// For `NamedComputationOp`s, there is only one target per data flow edge which
// is also the edge owner.
Value NamedComputationOp::getEdgeOwnerFromTarget(Value target) {
  assert(getOwningOp(target) == getOperation());
  return target;
}

// Returns the edge owner given a `source`.
//
// If the `source` is an operand of a terminator, return the corresponding
// result. Otherwise it should be an operand of the `NamedComputationOp`, return
// the `BlockArgument` with the same index.
Value NamedComputationOp::getEdgeOwnerFromSource(OpOperand& source) {
  return sdy::getEdgeOwnerFromSource(source, *this);
}

LogicalResult NamedComputationOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  NamedComputationOpAdaptor adaptor(operands, attributes, properties, regions);
  llvm::copy(getBodyTerminatorOperands(adaptor).getTypes(),
             std::back_inserter(inferredReturnTypes));
  return success();
}

//===----------------------------------------------------------------------===//
// AllSliceOp
//===----------------------------------------------------------------------===//

bool AllSliceOp::allowMissingInputSharding() { return true; }

Type AllSliceOp::getType() { return getResult().getType(); }

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

bool CollectivePermuteOp::allowDifferentMeshes() { return true; }

Type CollectivePermuteOp::getType() { return getResult().getType(); }

//===----------------------------------------------------------------------===//
// AllReduceOp
//===----------------------------------------------------------------------===//

bool AllReduceOp::allowMissingInputSharding() { return true; }

Type AllReduceOp::getType() { return getResult().getType(); }

//===----------------------------------------------------------------------===//
// ReduceScatterOp
//===----------------------------------------------------------------------===//

bool ReduceScatterOp::allowMissingInputSharding() { return true; }

Type ReduceScatterOp::getType() { return getResult().getType(); }

}  // namespace sdy
}  // namespace mlir

#include "shardy/dialect/sdy/ir/dialect.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "shardy/dialect/sdy/ir/attrs.cc.inc"
#define GET_OP_INTERFACE_CLASSES
#include "shardy/dialect/sdy/ir/op_interface.cc.inc"
#define GET_OP_CLASSES
#include "shardy/dialect/sdy/ir/ops.cc.inc"
