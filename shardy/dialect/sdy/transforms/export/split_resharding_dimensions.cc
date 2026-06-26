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

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_SPLITRESHARDINGDIMENSIONSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

using AxisList = std::list<AxisRefAttr>;

// Returns a copy of `sharding` with its dimension axes updated to `axesPerDim`.
TensorShardingAttr updateSharding(TensorShardingAttr sharding,
                                  ArrayRef<AxisList> axesPerDim,
                                  MLIRContext* context) {
  SmallVector<DimensionShardingAttr> dimShardings;
  for (auto [dimSharding, axes] :
       llvm::zip(sharding.getDimShardings(), axesPerDim)) {
    dimShardings.push_back(DimensionShardingAttr::get(
        context, llvm::to_vector(axes), dimSharding.getIsClosed()));
  }
  return TensorShardingAttr::get(context, sharding.getMeshOrRef(), dimShardings,
                                 sharding.getReplicatedAxes(),
                                 sharding.getUnreducedAxes());
}

// Collects the indices of split groups that are replicated (unmapped) in
// targetSharding.
SmallVector<int64_t> getExtraGroupIndices(
    ArrayRef<AxisList> groups, TensorShardingAttr targetSharding) {
  auto isAxisSharded = [&](AxisRefAttr axis) {
    for (int64_t j = 0; j < targetSharding.getRank(); ++j) {
      if (llvm::is_contained(targetSharding.getDimSharding(j).getAxes(),
                             axis)) {
        return true;
      }
    }
    return false;
  };

  SmallVector<int64_t> extraGroupIndices;
  for (size_t i = 0; i < groups.size(); ++i) {
    if (llvm::none_of(groups[i], isAxisSharded)) {
      extraGroupIndices.push_back(i);
    }
  }
  return extraGroupIndices;
}

// Collects axes in the corresponding target dimension that are not in the
// split groups.
SmallVector<AxisRefAttr> getExtraTargetAxes(
    ArrayRef<AxisList> groups, TensorShardingAttr targetSharding, int64_t dim) {
  SmallVector<AxisRefAttr> extraTargetAxes;
  for (AxisRefAttr axis : targetSharding.getDimSharding(dim).getAxes()) {
    if (llvm::none_of(groups, [&](const AxisList& group) {
          return llvm::is_contained(group, axis);
        })) {
      extraTargetAxes.push_back(axis);
    }
  }
  return extraTargetAxes;
}

// Pairs unmapped target axes with replicated groups and checks size
// divisibility. Returns std::nullopt if the matching is invalid.
std::optional<SmallVector<SmallVector<AxisRefAttr>>> matchExtraAxes(
    ArrayRef<AxisList> groups, ArrayRef<int64_t> extraGroupIndices,
    ArrayRef<AxisRefAttr> extraTargetAxes, MeshAttr mesh) {
  SmallVector<SmallVector<AxisRefAttr>> matchedExtraAxes(groups.size());
  ArrayRef<AxisRefAttr> remainingTargetAxes(extraTargetAxes);
  for (int64_t groupIdx : extraGroupIndices) {
    int64_t groupSize = 1;
    for (AxisRefAttr axis : groups[groupIdx]) {
      groupSize *= axis.getSize(mesh);
    }
    int64_t accumulatedTargetSize = 1;
    size_t consumedCount = 0;
    while (accumulatedTargetSize < groupSize &&
           consumedCount < remainingTargetAxes.size()) {
      accumulatedTargetSize *= remainingTargetAxes[consumedCount].getSize(mesh);
      consumedCount++;
    }
    if (consumedCount > 0) {
      if (groupSize % accumulatedTargetSize != 0) {
        return std::nullopt;
      }
      matchedExtraAxes[groupIdx] = llvm::to_vector(
          remainingTargetAxes.take_front(consumedCount));
      remainingTargetAxes = remainingTargetAxes.drop_front(consumedCount);
    }
  }
  if (!remainingTargetAxes.empty()) {
    return std::nullopt;
  }
  return matchedExtraAxes;
}

// A dim in sharding is splittable if its axes can be split into consecutive
// subgroups, such that:
//
// 1. Each subgroup maps to a distinct target dimension in targetSharding
//    (with consecutive replicated/unmapped axes grouped into dummy target
//    dimensions).
// 2. Any target dimension targetDim != dim that a subgroup maps to does not
//    contain axes originating from other input dimensions j != dim. Such
//    cross-dimension axes are conservatively disallowed to avoid complex
//    multi-stage collective transitions (like dependent AllToAlls).
// 3. Any extra output axes in the corresponding target dimension dim can be
//    consecutively paired with replicated input groups of divisible sizes.
//    There is no valid way to place these extra axes into the split target
//    dimensions without dropping/losing them.
//
// Groups consecutive axes in `sharding.getDimSharding(dim)` that map to the
// same dimension in `targetSharding`. Returns the groups if the dimension is
// splittable or std::nullopt otherwise.
std::optional<SmallVector<AxisList>> getSplittableGroups(
    TensorShardingAttr sharding, TensorShardingAttr targetSharding,
    int64_t dim, MeshAttr mesh) {
  ArrayRef<AxisRefAttr> axes = sharding.getDimSharding(dim).getAxes();
  if (axes.empty()) {
    return std::nullopt;
  }

  SmallVector<AxisList> groups;
  SmallVector<int64_t> targetDims;
  int64_t nextReplicatedDim = -1;
  std::optional<int64_t> lastDim = std::nullopt;

  // Helper to find which dimension in `targetSharding` contains `axis`. It
  // groups consecutive replicated axes together and map them to the same
  // dummy dimension.
  auto getTargetDim = [&](AxisRefAttr axis) -> int64_t {
    for (int64_t j = 0; j < targetSharding.getRank(); ++j) {
      if (llvm::is_contained(
              targetSharding.getDimSharding(j).getAxes(), axis)) {
        return j;
      }
    }
    // If the previous axis was also replicated, map it to the same dummy
    // dimension to keep consecutive replicated axes grouped together.
    if (lastDim && *lastDim < 0) {
      return *lastDim;
    }
    return nextReplicatedDim--;
  };

  // Group consecutive axes that map to the same target dimension.
  for (AxisRefAttr axis : axes) {
    int64_t nextDim = getTargetDim(axis);
    lastDim = nextDim;

    if (!targetDims.empty() && nextDim == targetDims.back()) {
      groups.back().push_back(axis);
    } else {
      if (llvm::is_contained(targetDims, nextDim)) {
        return std::nullopt;
      }
      groups.push_back({axis});
      targetDims.push_back(nextDim);
    }
  }

  if (groups.size() < 2) {
    return std::nullopt;
  }

  // Verify that none of the target dimensions contains cross-dimension axes
  // (axes that exist in different input dimensions j != dim). Cross-dimension
  // axes in targetDim != dim will trigger a layout decomposition that
  // introduces consecutive sub-axes in unsplit input dimensions, failing
  // verification.
  auto isFreeOfCrossDimAxes = [&](int64_t targetDim) -> bool {
    if (targetDim < 0) return true;
    for (AxisRefAttr outAxis :
         targetSharding.getDimSharding(targetDim).getAxes()) {
      for (int64_t j = 0; j < sharding.getRank(); ++j) {
        if (j == dim) {
          continue;
        }
        for (AxisRefAttr inAxis : sharding.getDimSharding(j).getAxes()) {
          if (inAxis.getName() == outAxis.getName()) {
            return false;
          }
        }
      }
    }
    return true;
  };

  if (!isFreeOfCrossDimAxes(dim)) {
    return std::nullopt;
  }
  for (int64_t targetDim : targetDims) {
    if (!isFreeOfCrossDimAxes(targetDim)) {
      return std::nullopt;
    }
  }

  SmallVector<int64_t> extraGroupIndices =
      getExtraGroupIndices(groups, targetSharding);
  SmallVector<AxisRefAttr> extraTargetAxes =
      getExtraTargetAxes(groups, targetSharding, dim);

  if (!matchExtraAxes(groups, extraGroupIndices, extraTargetAxes, mesh)) {
    return std::nullopt;
  }

  return groups;
}

// Computes the physical shape after splitting `dimIn` by the axes sizes of
// `groupsToSplit`.
SmallVector<int64_t> computeSplitShape(
    ArrayRef<int64_t> shape, int64_t dimIn, ArrayRef<AxisList> groupsToSplit,
    MeshAttr mesh) {
  int64_t d0 = shape[dimIn];

  SmallVector<int64_t> splitShape(shape.begin(), shape.begin() + dimIn);

  int64_t accumSize = 1;
  for (const AxisList& group : llvm::drop_end(groupsToSplit)) {
    int64_t groupSize = 1;
    for (AxisRefAttr axis : group) {
      groupSize *= axis.getSize(mesh);
    }
    splitShape.push_back(groupSize);
    accumSize *= groupSize;
  }

  SDY_CHECK(d0 % accumSize == 0);
  splitShape.push_back(d0 / accumSize);
  splitShape.append(shape.begin() + dimIn + 1, shape.end());
  return splitShape;
}

// Replaces the dimension `dim` in `sharding` with new dimensions defined by
// `newDimAxes`. The new dimensions inherit the closed/open state of `dim`.
TensorShardingAttr expandShardingDim(
    TensorShardingAttr sharding, int64_t dim,
    ArrayRef<SmallVector<AxisRefAttr>> newDimAxes, MLIRContext* context) {
  auto dimShardings = llvm::to_vector(sharding.getDimShardings());
  bool isClosed = dimShardings[dim].getIsClosed();

  SmallVector<DimensionShardingAttr> splitDims;
  splitDims.reserve(newDimAxes.size());
  for (const auto& axes : newDimAxes) {
    splitDims.push_back(DimensionShardingAttr::get(context, axes, isClosed));
  }

  dimShardings.erase(dimShardings.begin() + dim);
  dimShardings.insert(dimShardings.begin() + dim, splitDims.begin(),
                      splitDims.end());

  return TensorShardingAttr::get(context, sharding.getMeshOrRef(), dimShardings,
                                 sharding.getReplicatedAxes(),
                                 sharding.getUnreducedAxes());
}

struct SplitInfo {
  int64_t dim;
  SmallVector<AxisList> groups;
  bool splitByInput;
};

// Handles resharding where input(output) dimensions are sharded by multiple
// axes, and these axes map to distinct dimensions in the output(input). Returns
// true if it splits any dimensions.
// E.g., [{A, B}] -> [{}, {A}, {B}] or [{A}, {B}] -> [{A, B}, {}].
bool tryReshardWithDimensionSplits(ReshardOp op, Value bypassedInput,
                                   TensorShardingAttr inSharding,
                                   TensorShardingAttr outSharding,
                                   RewriterBase& rewriter) {
  if (inSharding.getMeshName() != outSharding.getMeshName()) {
    return false;
  }

  // Decompose sub-axes in both shardings into their smallest common
  // components to ensure they can be compared and split consistently.
  SmallVector<AxisList> inAxesPerDim = getAxesPerDim<AxisList>(inSharding);
  SmallVector<AxisList> outAxesPerDim = getAxesPerDim<AxisList>(outSharding);
  MeshAttr mesh = inSharding.getMesh(op);
  if (!mesh) {
    return false;
  }

  alignSubAxesByDecomposition(inAxesPerDim, outAxesPerDim, mesh);
  inSharding = updateSharding(inSharding, inAxesPerDim, rewriter.getContext());
  outSharding =
      updateSharding(outSharding, outAxesPerDim, rewriter.getContext());

  auto tensorType = mlir::cast<RankedTensorType>(bypassedInput.getType());

  // Identify which dimensions can be split. We check both the input and
  // output shardings because a dimension split could be triggered by either
  // (e.g. [{A, B}] -> [{A}, {B}] splits the input, [{A}, {B}] -> [{A, B}]
  // splits the output).
  SmallVector<SplitInfo> splits;
  for (int64_t i = 0; i < inSharding.getRank(); ++i) {
    if (ShapedType::isDynamic(tensorType.getDimSize(i))) {
      continue;
    }
    if (auto groups =
            getSplittableGroups(inSharding, outSharding, i, mesh)) {
      splits.push_back({i, *groups, true});
    } else if (auto groups = getSplittableGroups(
                   outSharding, inSharding, i, mesh)) {
      splits.push_back({i, *groups, false});
    }
  }
  if (splits.empty()) {
    return false;
  }
  SmallVector<int64_t> splitShape(tensorType.getShape().begin(),
                                  tensorType.getShape().end());
  TensorShardingAttr inShardingSplit = inSharding;
  TensorShardingAttr outShardingSplit = outSharding;

  // Process splits in reverse order so that expanding a dimension doesn't
  // shift the indices of subsequent dimensions we need to split.
  for (const SplitInfo& split : llvm::reverse(splits)) {
    int64_t dimIn = split.dim;
    const SmallVector<AxisList>& groupsToSplit = split.groups;
    // Calculate the new tensor shape after splitting this dimension.
    splitShape = computeSplitShape(splitShape, dimIn, groupsToSplit, mesh);
    // Separate the axes that triggered the split (withAxes) and the matching
    // axes in the opposite sharding (withoutAxes) for each new split dimension.
    SmallVector<SmallVector<AxisRefAttr>> withAxes(groupsToSplit.size());
    SmallVector<SmallVector<AxisRefAttr>> withoutAxes(groupsToSplit.size());
    for (auto [i, group] : llvm::enumerate(groupsToSplit)) {
      withAxes[i].assign(group.begin(), group.end());
    }
    TensorShardingAttr shardingWithoutAxes =
        split.splitByInput ? outShardingSplit : inShardingSplit;

    SmallVector<int64_t> extraGroupIndices =
        getExtraGroupIndices(groupsToSplit, shardingWithoutAxes);
    SmallVector<AxisRefAttr> extraTargetAxes =
        getExtraTargetAxes(groupsToSplit, shardingWithoutAxes, dimIn);

    auto matchedExtraAxes = matchExtraAxes(
        groupsToSplit, extraGroupIndices, extraTargetAxes, mesh);
    SDY_CHECK(matchedExtraAxes.has_value());

    for (AxisRefAttr axis :
         shardingWithoutAxes.getDimSharding(dimIn).getAxes()) {
      for (size_t i = 0; i < groupsToSplit.size(); ++i) {
        if (llvm::is_contained(groupsToSplit[i], axis)) {
          withoutAxes[i].push_back(axis);
          break;
        }
      }
    }

    for (int64_t groupIdx : extraGroupIndices) {
      if (!(*matchedExtraAxes)[groupIdx].empty()) {
        withoutAxes[groupIdx].append((*matchedExtraAxes)[groupIdx].begin(),
                                     (*matchedExtraAxes)[groupIdx].end());
      }
    }

    // Expand the sharding attributes by inserting the new split dimensions.
    MLIRContext* ctx = rewriter.getContext();
    inShardingSplit =
        expandShardingDim(inShardingSplit, dimIn,
                          split.splitByInput ? withAxes : withoutAxes, ctx);
    outShardingSplit =
        expandShardingDim(outShardingSplit, dimIn,
                          split.splitByInput ? withoutAxes : withAxes, ctx);
  }

  // Replace the original reshard with a reshape -> reshard -> reshape
  // sequence. The inner reshard operates on the expanded, split dimensions.
  auto splitTensorType =
      RankedTensorType::get(splitShape, tensorType.getElementType());

  Value splitInputVal = stablehlo::ReshapeOp::create(
      rewriter, op->getLoc(), splitTensorType, bypassedInput);
  setSharding(splitInputVal, inShardingSplit);

  Value reshardedSplitVal = ReshardOp::create(
      rewriter, op->getLoc(), splitTensorType, splitInputVal, outShardingSplit);

  Value mergeOutputVal = stablehlo::ReshapeOp::create(
      rewriter, op->getLoc(), tensorType, reshardedSplitVal);
  setSharding(mergeOutputVal, op.getSharding());
  rewriter.replaceOp(op, mergeOutputVal);
  return true;
}

// This pass physically splits tensor dimensions (via stablehlo.reshape) to
// separate sharding axes that map to different dimensions.
//
// For example, if we reshard `[{"x", "y"}, {}] -> [{"x"}, {"y"}]`, the axes
// `x` and `y` are in the same dimension in the input but must go to different
// dimensions in the output. This pass splits the input dimension into two
// dimensions, allowing the subsequent collectives pass to lower them using
// more optimized collectives, by rewriting to:
//
//   %split_input = stablehlo.reshape %input : tensor<16x16xf32>
//       -> tensor<4x4x16xf32>
//   %reshard = sdy.reshard %split_input <..., [{"x"}, {"y"}, {}]>
//   %output = stablehlo.reshape %reshard : tensor<4x4x16xf32>
//       -> tensor<16x16xf32>
//
// We only split dimensions when the split subgroups can be aligned with the
// opposite sharding using logical reshapes alone (without transposes). This
// includes splitting replicated/unmapped axes by mapping them to virtual
// dimensions. We do not split if there are extra/other axes in either the
// splitting dimension, the mapped target dimensions, or the corresponding
// target dimension itself.
class SplitReshardingDimensionsPass
    : public impl::SplitReshardingDimensionsPassBase<
          SplitReshardingDimensionsPass> {
 protected:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](ReshardOp op) {
      TensorShardingAttr outSharding = op.getSharding();
      TensorShardingAttr inSharding =
          getOrCreateSharding(op.getInput(), outSharding.getMeshName());

      rewriter.setInsertionPoint(op);
      tryReshardWithDimensionSplits(op, op.getInput(), inSharding, outSharding,
                                    rewriter);
    });
  }
};

}  // namespace
}  // namespace sdy
}  // namespace mlir
