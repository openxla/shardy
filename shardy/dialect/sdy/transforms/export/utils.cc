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
#include <iterator>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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
    if (shardCount > 1 && dimSize % shardCount != 0) {
      int64_t paddedDim = dimSize + shardCount - (dimSize % shardCount);
      newShape.push_back(paddedDim);
      changed = true;
    } else {
      newShape.push_back(dimSize);
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
}  // namespace sdy
}  // namespace mlir
