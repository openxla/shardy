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

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LLVM.h"
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

}  // namespace sdy
}  // namespace mlir
