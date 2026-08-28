/* Copyright 2026 The Shardy Authors.

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

#include "shardy/dialect/sdy/transforms/export/utils.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

namespace {

// Returns a sorted vector containing all axes in `axesPerDim`.
SmallVector<AxisRefAttr> getOrderedAxes(ArrayRef<AxisList> axesPerDim) {
  SmallVector<AxisRefAttr> result;
  for (const AxisList& axes : axesPerDim) {
    result.append(axes.begin(), axes.end());
  }
  llvm::sort(result);
  return result;
}

}  // namespace

void alignSubAxesByDecomposition(AxisList& axes,
                                 ArrayRef<AxisRefAttr> orderedOtherAxes,
                                 MeshAttr mesh) {
  auto axisIt = axes.begin();
  while (axisIt != axes.end()) {
    AxisRefAttr axis = *axisIt;
    auto* overlapIt = axis.getFirstOverlapping(orderedOtherAxes);
    // There are two paths to complete the while loop below:
    // 1. the while condition is not met from the start, in which case we need
    //    to advance `axisIt`.
    // 2. we enter the while until the condition isn't met, in which case we
    //    only need to advance `axisIt` if it points to a created suffix.
    bool axisAdvancedInWhile = false;
    while (overlapIt != orderedOtherAxes.end() && overlapIt->canCoexist(axis) &&
           !overlapIt->contains(axis) && overlapIt->overlaps(axis)) {
      axisIt = axes.erase(axisIt);
      if (OptionalAxisRef prefix = axis.getPrefixWithoutOverlap(*overlapIt)) {
        axes.insert(axisIt, *prefix);
      }
      axes.insert(axisIt, *axis.getOverlap(*overlapIt));
      if (OptionalAxisRef suffix =
              axis.getSuffixWithoutOverlap(*overlapIt, mesh)) {
        // If there is a suffix, that should be the next axis to process.
        axisIt = axes.insert(axisIt, *suffix);
        axis = *suffix;
        ++overlapIt;
        axisAdvancedInWhile = false;
      } else {
        // Otherwise, we're done with the current axis.
        axisAdvancedInWhile = true;
        break;
      }
    }
    if (!axisAdvancedInWhile) {
      ++axisIt;
    }
  }
}

void alignSubAxesByDecomposition(SmallVector<AxisList>& axesPerDim,
                                 ArrayRef<AxisRefAttr> orderedOtherAxes,
                                 MeshAttr mesh) {
  if (orderedOtherAxes.empty()) {
    return;
  }
  for (AxisList& axes : axesPerDim) {
    alignSubAxesByDecomposition(axes, orderedOtherAxes, mesh);
  }
}

void alignSubAxesByDecomposition(SmallVector<AxisList>& inAxesPerDim,
                                 SmallVector<AxisList>& outAxesPerDim,
                                 MeshAttr mesh) {
  SmallVector<AxisRefAttr> orderedInAxes = getOrderedAxes(inAxesPerDim);
  SmallVector<AxisRefAttr> orderedOutAxes = getOrderedAxes(outAxesPerDim);
  alignSubAxesByDecomposition(inAxesPerDim, orderedOutAxes, mesh);
  alignSubAxesByDecomposition(outAxesPerDim, orderedInAxes, mesh);
}
TensorShardingAttr updateSharding(TensorShardingAttr sharding,
                                  ArrayRef<AxisList> axesPerDim) {
  MLIRContext* context = sharding.getContext();
  SmallVector<DimensionShardingAttr> dimShardings;
  dimShardings.reserve(sharding.getRank());
  for (auto [dimSharding, axes] :
       llvm::zip(sharding.getDimShardings(), axesPerDim)) {
    dimShardings.push_back(DimensionShardingAttr::get(
        context, llvm::to_vector(axes), dimSharding.getIsClosed(),
        dimSharding.getPriority()));
  }
  return TensorShardingAttr::get(context, sharding.getMeshOrRef(), dimShardings,
                                 sharding.getReplicatedAxes(),
                                 sharding.getUnreducedAxes(),
                                 sharding.getReductionOp());
}

bool isCommunicationFreeSliceDim(int64_t dimIdx, stablehlo::SliceOp sliceOp,
                                 TensorShardingAttr sharding, MeshAttr mesh) {
  int64_t shardCount = sharding.getDimShardings()[dimIdx].getShardedSize(mesh);

  if (shardCount <= 1) {
    return true;
  }

  if (sliceOp.getStartIndices()[dimIdx] != 0 ||
      sliceOp.getStrides()[dimIdx] != 1) {
    return false;
  }

  ArrayRef<int64_t> inShape = getTensorShape(sliceOp.getOperand());
  ArrayRef<int64_t> outShape = getTensorShape(sliceOp.getResult());
  int64_t inDimSize = inShape[dimIdx];
  int64_t outDimSize = outShape[dimIdx];

  // Conservatively return false for dynamic shapes if sharded across devices.
  if (inDimSize == ShapedType::kDynamic || outDimSize == ShapedType::kDynamic) {
    return false;
  }

  return llvm::divideCeil(inDimSize, shardCount) ==
         llvm::divideCeil(outDimSize, shardCount);
}

bool isCommunicationFreePadDim(int64_t dimIdx, stablehlo::PadOp padOp,
                               TensorShardingAttr sharding, MeshAttr mesh) {
  int64_t shardCount = sharding.getDimShardings()[dimIdx].getShardedSize(mesh);
  if (shardCount <= 1) {
    return true;
  }

  if (padOp.getInteriorPadding()[dimIdx] != 0) {
    return false;
  }

  ArrayRef<int64_t> inShape = getTensorShape(padOp.getOperand());
  int64_t inDimSize = inShape[dimIdx];
  if (inDimSize == ShapedType::kDynamic) {
    return false;
  }

  int64_t sIn = llvm::divideCeil(inDimSize, shardCount);
  int64_t sOut =
      llvm::divideCeil(padOp.getType().getDimSize(dimIdx), shardCount);
  int64_t low = padOp.getEdgePaddingLow()[dimIdx];

  for (int64_t t = 0; t < shardCount; ++t) {
    int64_t start = t * sOut - low;
    int64_t limit = start + sOut - 1;
    int64_t validStart = std::max<int64_t>(0, start);
    int64_t validLimit = std::min<int64_t>(inDimSize - 1, limit);
    if (validStart <= validLimit) {
      if (validStart / sIn != t || validLimit / sIn != t) {
        return false;
      }
    }
  }
  return true;
}

mlir::stablehlo::MeshAttr convertMeshAttr(MeshAttr sdyMesh) {
  SmallVector<mlir::stablehlo::MeshAxisAttr> shloAxes;
  for (MeshAxisAttr axisAttr : sdyMesh.getAxes()) {
    shloAxes.push_back(mlir::stablehlo::MeshAxisAttr::get(
        axisAttr.getContext(), axisAttr.getName(), axisAttr.getSize()));
  }
  DenseIntElementsAttr deviceIds;
  if (!sdyMesh.getDeviceIds().empty()) {
    auto type = RankedTensorType::get(
        {static_cast<int64_t>(sdyMesh.getDeviceIds().size())},
        Builder(sdyMesh.getContext()).getI64Type());
    deviceIds = DenseIntElementsAttr::get(type, sdyMesh.getDeviceIds());
  }
  return mlir::stablehlo::MeshAttr::get(sdyMesh.getContext(), shloAxes,
                                        deviceIds);
}

bool usePartitionId(int64_t replicaCount, int64_t partitionCount) {
  SDY_CHECK(replicaCount == 1 || partitionCount == 1)
      << "Shardy partitioner does not support "
         "replica_count > 1 && partition_count > 1 yet.";
  return replicaCount == 1;
}

Value getDeviceId(int64_t replicaCount, int64_t partitionCount, Location loc,
                  OpBuilder& rewriter) {
  Type i64Ty = rewriter.getI64Type();
  auto indexTy = RankedTensorType::get({}, i64Ty);
  Value idOp;
  if (usePartitionId(replicaCount, partitionCount)) {
    idOp = stablehlo::PartitionIdOp::create(rewriter, loc);
  } else {
    idOp = stablehlo::ReplicaIdOp::create(rewriter, loc);
  }
  auto idType = cast<RankedTensorType>(idOp.getType());
  if (idType.getElementType() != i64Ty) {
    auto destType = RankedTensorType::get(idType.getShape(), i64Ty);
    idOp = stablehlo::ConvertOp::create(rewriter, loc, destType, idOp);
  }
  if (idType.getRank() != 0) {
    idOp = stablehlo::ReshapeOp::create(rewriter, loc, indexTy, idOp);
  }
  return idOp;
}

stablehlo::ChannelHandleAttr getChannelHandle(MLIRContext* ctx,
                                              int64_t replicaCount,
                                              int64_t partitionCount,
                                              int64_t& nextChannelId) {
  if (usePartitionId(replicaCount, partitionCount)) {
    return stablehlo::ChannelHandleAttr::get(ctx, nextChannelId++,
                                             kCrossPartitionChannelHandleType);
  }
  return nullptr;
}

MeshOp getGlobalMeshOp(ModuleOp moduleOp) {
  for (MeshOp meshOp : moduleOp.getOps<MeshOp>()) {
    MeshAttr mesh = meshOp.getMesh();
    if (!mesh.isMaximal() && !mesh.getAxes().empty()) {
      return meshOp;
    }
  }
  return nullptr;
}

int64_t getShardIndex(int64_t deviceId, MeshAttr mesh,
                      ArrayRef<AxisRefAttr> axes) {
  int64_t logicalDeviceId = deviceId;
  ArrayRef<int64_t> deviceIds = mesh.getDeviceIds();
  if (!deviceIds.empty()) {
    const auto* it = llvm::find(deviceIds, deviceId);
    SDY_CHECK(it != deviceIds.end()) << "Device ID not found in mesh";
    logicalDeviceId = std::distance(deviceIds.begin(), it);
  } else {
    SDY_CHECK(deviceId >= 0 && deviceId < mesh.getTotalSize())
        << "Device ID out of range for mesh";
  }

  int64_t shardIndex = 0;
  for (AxisRefAttr axis : axes) {
    int64_t axisSize = axis.getSize(mesh);
    int64_t suffixSize = 1;
    bool foundAxis = false;
    for (MeshAxisAttr meshAxis : mesh.getAxes()) {
      if (foundAxis) {
        suffixSize *= meshAxis.getSize();
      }
      if (meshAxis.getName() == axis.getName()) {
        foundAxis = true;
      }
    }

    int64_t fullSize = mesh.getAxisSize(axis.getName());
    int64_t subAxisStride = fullSize / (axis.getSubAxisPreSize() * axisSize);
    int64_t axisCoord =
        (logicalDeviceId / (suffixSize * subAxisStride)) % axisSize;

    shardIndex = shardIndex * axisSize + axisCoord;
  }
  return shardIndex;
}

Type getDivisiblePaddedType(Type type, TensorShardingAttr sharding,
                            const SymbolTable& symbolTable,
                            const llvm::DenseSet<StringRef>* allowedAxes) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  if (!rankedType || !sharding || sharding.isFullyReplicated()) {
    return type;
  }
  MeshAttr mesh = sharding.getMesh(symbolTable);
  if (!mesh) {
    return type;
  }
  SmallVector<int64_t> newShape;
  bool changed = false;
  for (auto [dimSize, dimSharding] :
       llvm::zip_equal(rankedType.getShape(), sharding.getDimShardings())) {
    if (dimSize == ShapedType::kDynamic) {
      newShape.push_back(ShapedType::kDynamic);
      continue;
    }
    int64_t shardCount = 1;
    if (allowedAxes) {
      for (AxisRefAttr axisRef : dimSharding.getAxes()) {
        if (allowedAxes->contains(axisRef.getName())) {
          shardCount *= axisRef.getSize(mesh);
        }
      }
    } else {
      shardCount = dimSharding.getShardedSize(mesh);
    }
    int64_t paddedDim = llvm::alignTo(dimSize, shardCount);
    newShape.push_back(paddedDim);
    if (paddedDim != dimSize) {
      changed = true;
    }
  }
  if (!changed) {
    return type;
  }
  return RankedTensorType::get(newShape, rankedType.getElementType());
}

mlir::stablehlo::AxisRefAttr convertAxisRefAttr(AxisRefAttr sdyAxisRef) {
  MLIRContext* ctx = sdyAxisRef.getContext();
  mlir::stablehlo::SubAxisInfoAttr subAxisInfo = nullptr;
  if (auto sdySubAxisInfo = sdyAxisRef.getSubAxisInfo()) {
    subAxisInfo = mlir::stablehlo::SubAxisInfoAttr::get(
        ctx, sdySubAxisInfo.getPreSize(), sdySubAxisInfo.getSize());
  }
  return mlir::stablehlo::AxisRefAttr::get(ctx, sdyAxisRef.getName(),
                                           subAxisInfo);
}

FlatSymbolRefAttr getOrCreateMeshSymbol(Location loc, ModuleOp module,
                                        Attribute meshOrRef,
                                        SymbolTable& symbolTable) {
  if (auto sym = dyn_cast_or_null<FlatSymbolRefAttr>(meshOrRef)) {
    return sym;
  }
  auto meshAttr = dyn_cast_or_null<MeshAttr>(meshOrRef);
  if (!meshAttr) {
    return nullptr;
  }
  if (!meshAttr.isMaximal()) {
    if (MeshOp globalMeshOp = getGlobalMeshOp(module)) {
      if (globalMeshOp.getMesh() == meshAttr) {
        return FlatSymbolRefAttr::get(module.getContext(),
                                      globalMeshOp.getSymName());
      }
    }
  }
  for (MeshOp meshOp : module.getOps<MeshOp>()) {
    if (meshOp.getMesh() == meshAttr) {
      return FlatSymbolRefAttr::get(module.getContext(), meshOp.getSymName());
    }
  }
  OpBuilder moduleBuilder(module.getBodyRegion());
  MeshOp newMeshOp = MeshOp::create(moduleBuilder, loc, "mesh", meshAttr);
  symbolTable.insert(newMeshOp, module.getBody()->begin());
  return FlatSymbolRefAttr::get(module.getContext(), newMeshOp.getSymName());
}

FlatSymbolRefAttr getOrCreateMeshSymbol(Operation* op, Attribute meshOrRef,
                                        SymbolTable& symbolTable) {
  ModuleOp module =
      isa<ModuleOp>(op) ? cast<ModuleOp>(op) : op->getParentOfType<ModuleOp>();
  return getOrCreateMeshSymbol(op->getLoc(), module, meshOrRef, symbolTable);
}

bool ReshapeGroupInfo::hasIndivisibility(ArrayRef<int64_t> paddedInputShape,
                                         ArrayRef<int64_t> paddedOutputShape,
                                         RankedTensorType inType,
                                         RankedTensorType outType) const {
  auto checkDims = [&](int64_t startDim, int64_t lastDim,
                       ArrayRef<int64_t> paddedShape, RankedTensorType type) {
    for (int64_t d = startDim; d < lastDim; ++d) {
      if (paddedShape[d] != type.getDimSize(d)) {
        return true;
      }
    }
    return false;
  };
  return checkDims(inStartDim, inLastDim, paddedInputShape, inType) ||
         checkDims(outStartDim, outLastDim, paddedOutputShape, outType);
}

bool ReshapeGroupInfo::hasIndivisibility(TensorShardingAttr inSharding,
                                         TensorShardingAttr outSharding,
                                         MeshAttr mesh, RankedTensorType inType,
                                         RankedTensorType outType) const {
  auto checkDims = [&](int64_t startDim, int64_t lastDim,
                       TensorShardingAttr sharding, RankedTensorType type) {
    for (int64_t d = startDim; d < lastDim; ++d) {
      int64_t sc = sharding.getDimShardings()[d].getShardedSize(mesh);
      if (sc > 1 && type.getDimSize(d) % sc != 0) {
        return true;
      }
    }
    return false;
  };
  return checkDims(inStartDim, inLastDim, inSharding, inType) ||
         checkDims(outStartDim, outLastDim, outSharding, outType);
}

SmallVector<AxisRefAttr> ReshapeGroupInfo::getAllAxisRefs(
    TensorShardingAttr inSharding, TensorShardingAttr outSharding) const {
  SmallVector<AxisRefAttr> axisRefs;
  for (int64_t d = inStartDim; d < inLastDim; ++d) {
    for (AxisRefAttr axis : inSharding.getDimShardings()[d].getAxes()) {
      axisRefs.push_back(axis);
    }
  }
  for (int64_t d = outStartDim; d < outLastDim; ++d) {
    for (AxisRefAttr axis : outSharding.getDimShardings()[d].getAxes()) {
      axisRefs.push_back(axis);
    }
  }
  return axisRefs;
}

int64_t ReshapeGroupInfo::getInVolume(ArrayRef<int64_t> shape) const {
  return std::accumulate(shape.begin() + inStartDim, shape.begin() + inLastDim,
                         1LL, std::multiplies<int64_t>());
}

int64_t ReshapeGroupInfo::getOutVolume(ArrayRef<int64_t> shape) const {
  return std::accumulate(shape.begin() + outStartDim,
                         shape.begin() + outLastDim, 1LL,
                         std::multiplies<int64_t>());
}

SmallVector<ReshapeGroupInfo> buildReshapeGroupInfos(RankedTensorType inType,
                                                     RankedTensorType outType) {
  SmallVector<ReshapeGroupInfo> groups;
  int64_t inRank = inType.getRank();
  int64_t outRank = outType.getRank();

  int64_t inDim = 0;
  int64_t outDim = 0;

  auto countNonOneDims = [](RankedTensorType type, int64_t start,
                            int64_t last) {
    int64_t count = 0;
    for (int64_t d = start; d < last; ++d) {
      if (type.getDimSize(d) != 1) {
        count++;
      }
    }
    return count;
  };

  while (inDim < inRank || outDim < outRank) {
    int64_t inStart = inDim;
    int64_t outStart = outDim;

    int64_t inProd = inDim < inRank ? inType.getDimSize(inDim++) : 1;
    int64_t outProd = outDim < outRank ? outType.getDimSize(outDim++) : 1;

    while (inProd != outProd && (inDim < inRank || outDim < outRank)) {
      if (inProd < outProd && inDim < inRank) {
        inProd *= inType.getDimSize(inDim++);
      } else if (outProd < inProd && outDim < outRank) {
        outProd *= outType.getDimSize(outDim++);
      } else {
        break;
      }
    }

    ReshapeGroupInfo g;
    g.inStartDim = inStart;
    g.inLastDim = inDim;
    g.outStartDim = outStart;
    g.outLastDim = outDim;
    g.numInNontrivialDims = countNonOneDims(inType, inStart, inDim);
    g.numOutNontrivialDims = countNonOneDims(outType, outStart, outDim);

    if (inDim - inStart > 1) {
      SDY_CHECK(inType.getDimSize(inDim - 1) != 1);
    }
    if (outDim - outStart > 1) {
      SDY_CHECK(outType.getDimSize(outDim - 1) != 1);
    }

    groups.push_back(g);
  }

  return groups;
}

bool isCommunicationFreeReshape(stablehlo::ReshapeOp reshapeOp,
                                TensorShardingAttr inSharding,
                                TensorShardingAttr outSharding, MeshAttr mesh,
                                RankedTensorType inputType,
                                RankedTensorType outputType,
                                ArrayRef<ReshapeGroupInfo> reshapeGroups) {
  if (isShardingEquivalentAcrossReshapes(inSharding, inputType, outSharding,
                                         outputType, reshapeOp,
                                         /*allowNonDivisible=*/false)) {
    return true;
  }

  if (!isShardingEquivalentAcrossReshapes(inSharding, inputType, outSharding,
                                          outputType, reshapeOp,
                                          /*allowNonDivisible=*/true)) {
    return false;
  }

  for (const ReshapeGroupInfo& g : reshapeGroups) {
    if (!g.isPassthrough()) {
      return false;
    }
  }

  return true;
}

bool isCommunicationFreeReshape(stablehlo::ReshapeOp reshapeOp,
                                TensorShardingAttr inSharding,
                                TensorShardingAttr outSharding, MeshAttr mesh,
                                RankedTensorType inputType,
                                RankedTensorType outputType) {
  SmallVector<ReshapeGroupInfo> groups =
      buildReshapeGroupInfos(inputType, outputType);
  return isCommunicationFreeReshape(reshapeOp, inSharding, outSharding, mesh,
                                    inputType, outputType, groups);
}

Value sliceHighSideToType(OpBuilder& builder, Location loc, Value operand,
                          Type targetType, TensorShardingAttr sharding) {
  Type currentType = operand.getType();
  if (currentType == targetType) {
    return operand;
  }
  auto rankedTargetType = cast<RankedTensorType>(targetType);
  int64_t rank = rankedTargetType.getRank();
  SmallVector<int64_t> sliceStarts(rank, 0);
  SmallVector<int64_t> sliceStrides(rank, 1);
  Value sliced = stablehlo::SliceOp::create(
      builder, loc, rankedTargetType, operand,
      builder.getDenseI64ArrayAttr(sliceStarts),
      builder.getDenseI64ArrayAttr(rankedTargetType.getShape()),
      builder.getDenseI64ArrayAttr(sliceStrides));
  if (sharding) {
    setSharding(sliced, sharding);
  }
  return sliced;
}

Value padHighSideToType(OpBuilder& builder, Location loc, Value operand,
                        Type targetType, TensorShardingAttr sharding,
                        Value paddingValue, bool allowSlicePeephole) {
  Type currentType = operand.getType();
  if (currentType == targetType) {
    return operand;
  }
  if (allowSlicePeephole) {
    if (auto sliceOp = operand.getDefiningOp<stablehlo::SliceOp>()) {
      if (sliceOp.getOperand().getType() == targetType) {
        bool allZeros = llvm::all_of(sliceOp.getStartIndices(),
                                     [](int64_t idx) { return idx == 0; });
        bool allUnitStrides = llvm::all_of(
            sliceOp.getStrides(), [](int64_t stride) { return stride == 1; });
        if (allZeros && allUnitStrides) {
          return sliceOp.getOperand();
        }
      }
    }
  }
  auto currentRanked = cast<RankedTensorType>(currentType);
  auto targetRanked = cast<RankedTensorType>(targetType);
  SmallVector<int64_t> edgePaddingHigh(targetRanked.getRank(), 0);
  for (int d = 0; d < targetRanked.getRank(); ++d) {
    edgePaddingHigh[d] =
        targetRanked.getDimSize(d) - currentRanked.getDimSize(d);
  }
  Value padVal = paddingValue;
  if (!padVal) {
    auto zeroType = RankedTensorType::get({}, targetRanked.getElementType());
    padVal = stablehlo::ConstantOp::create(builder, loc,
                                           builder.getZeroAttr(zeroType));
  }
  Value padded = stablehlo::PadOp::create(
      builder, loc, targetRanked, operand, padVal,
      builder.getDenseI64ArrayAttr(
          SmallVector<int64_t>(targetRanked.getRank(), 0)),
      builder.getDenseI64ArrayAttr(edgePaddingHigh),
      builder.getDenseI64ArrayAttr(
          SmallVector<int64_t>(targetRanked.getRank(), 0)));
  if (sharding) {
    setSharding(padded, sharding);
  }
  return padded;
}
}  // namespace sdy
}  // namespace mlir
