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
#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

namespace {

#include "shardy/dialect/sdy/ir/canonicalization.cc.inc"

// Pattern to remove unused block arguments and their corresponding operands
// from  a `ManualComputationOp`.
class ManualComputationUnusedInputsPattern
    : public OpRewritePattern<ManualComputationOp> {
 public:
  using OpRewritePattern<ManualComputationOp>::OpRewritePattern;

 private:
  LogicalResult matchAndRewrite(ManualComputationOp manualComputationOp,
                                PatternRewriter& rewriter) const override {
    BitVector unusedArgs(manualComputationOp.getNumOperands());
    for (BlockArgument arg : manualComputationOp.getRegion().getArguments()) {
      if (arg.use_empty()) {
        unusedArgs.set(arg.getArgNumber());
      }
    }
    if (unusedArgs.none()) {
      return failure();
    }

    manualComputationOp->eraseOperands(unusedArgs);
    manualComputationOp.getRegion().front().eraseArguments(unusedArgs);

    SmallVector<TensorShardingAttr> inShardings;
    inShardings.reserve(manualComputationOp.getNumOperands());
    for (int64_t index : unusedArgs.flip().set_bits()) {
      inShardings.push_back(manualComputationOp.getInSharding(index));
    }
    manualComputationOp.setInShardings(inShardings);

    return success();
  }
};

// Pattern to:
// 1. Inline a ManualComputationOp when the product of all manual axes is 1.
// 2. Erase a ManualComputationOp that has no inputs/outputs and an empty body.
class RedundantManualComputationPattern
    : public OpRewritePattern<ManualComputationOp> {
 public:
  using OpRewritePattern<ManualComputationOp>::OpRewritePattern;

 private:
  LogicalResult matchAndRewrite(ManualComputationOp manualComputationOp,
                                PatternRewriter& rewriter) const override {
    ArrayRef<TensorShardingAttr> inShardings =
        manualComputationOp.getInShardings().getShardings();
    ArrayRef<TensorShardingAttr> outShardings =
        manualComputationOp.getOutShardings().getShardings();

    if (inShardings.empty() && outShardings.empty() &&
        isa<ReturnOp>(manualComputationOp.getBody().front().front())) {
      rewriter.eraseOp(manualComputationOp);
      return success();
    }

    int64_t manualAxesProduct = 1;
    if (!inShardings.empty() || !outShardings.empty()) {
      MeshAttr mesh =
          getCommonMesh(inShardings, outShardings, manualComputationOp);
      assert(mesh && "expected inputs and outputs to have a common mesh");
      for (StringAttr manualAxis : manualComputationOp.getManualAxes()) {
        manualAxesProduct *= mesh.getAxisSize(manualAxis);
      }
    }

    if (manualAxesProduct != 1) {
      return rewriter.notifyMatchFailure(
          manualComputationOp, [](Diagnostic& diag) {
            diag << "product of manual axis sizes is not 1";
          });
    }

    mlir::InlinerInterface inliner(manualComputationOp.getContext());
    mlir::InlinerConfig config;
    if (inlineRegion(
            inliner, config.getCloneCallback(),
            &manualComputationOp.getRegion(), manualComputationOp->getBlock(),
            manualComputationOp->getIterator(),
            manualComputationOp.getOperands(), manualComputationOp.getResults())
            .failed()) {
      manualComputationOp.emitOpError(
          "failed to inline redundant ManualComputationOp.");
      return failure();
    }
    rewriter.eraseOp(manualComputationOp);
    return success();
  }
};

// Struct to hold the results of the reduce scatter fusion computation.
struct ReduceScatterFusionInfo {
  SmallVector<AxisRefListAttr> reduceScatterAxes;
  std::optional<SmallVector<AxisRefListAttr>> residualSlicingAxes;
  TensorShardingAttr reduceScatterOutSharding;
};

// Helper function to check if the reduce scatter fusion is possible and compute
// the axes. Returns std::nullopt on failure.
std::optional<ReduceScatterFusionInfo> matchAndComputeReduceScatterFusionAxes(
    AllReduceOp allReduceOp, AllSliceOp allSliceOp, PatternRewriter& rewriter) {
  MLIRContext* context = rewriter.getContext();
  MeshAttr mesh = allReduceOp.getOutSharding().getMesh(allReduceOp);
  ArrayRef<AxisRefAttr> reductionAxes = allReduceOp.getReductionAxes();
  ArrayRef<AxisRefListAttr> slicingAxes = allSliceOp.getSlicingAxes();
  llvm::SmallDenseSet<AxisRefAttr> remainingReductionAxes(reductionAxes.begin(),
                                                          reductionAxes.end());

  int64_t rank = slicingAxes.size();
  bool hasResidualSlicingAxes = false;
  SmallVector<AxisRefListAttr> reduceScatterAxes;
  SmallVector<AxisRefListAttr> residualSlicingAxes;
  reduceScatterAxes.reserve(rank);
  residualSlicingAxes.reserve(rank);

  for (AxisRefListAttr slicingAxesPerDim : slicingAxes) {
    SmallVector<AxisRefAttr> reduceScatterAxesPerDim;
    SmallVector<AxisRefAttr> residualSlicingAxesPerDim;

    auto axes = slicingAxesPerDim.getValue();
    const auto* it = axes.begin();
    const auto* end = axes.end();

    // Find the longest prefix of slicing axes for this dimension that are
    // also present in the remaining reduction axes.
    while (it != end && remainingReductionAxes.contains(*it)) {
      reduceScatterAxesPerDim.push_back(*it);
      remainingReductionAxes.erase(*it);
      ++it;
    }

    // Any remaining slicing axes for this dims become residual slicing axes.
    std::copy(it, end, std::back_inserter(residualSlicingAxesPerDim));

    reduceScatterAxes.push_back(
        AxisRefListAttr::get(context, reduceScatterAxesPerDim));
    residualSlicingAxes.push_back(
        AxisRefListAttr::get(context, residualSlicingAxesPerDim));
    if (!residualSlicingAxesPerDim.empty()) {
      hasResidualSlicingAxes = true;
    }
  }

  // Check if all reduction axes were consumed
  if (!remainingReductionAxes.empty()) {
    return std::nullopt;
  }

  // Calculate the output sharding of the ReduceScatterOp.
  TensorShardingAttr allReduceInSharding =
      getOrCreateSharding(allReduceOp.getTensor(), mesh,
                          /*closedIfMissing=*/true);
  SmallVector<DimensionShardingAttr> reduceScatterOutDimShardings;
  reduceScatterOutDimShardings.reserve(rank);

  for (auto [operandDimSharding, reduceScatterAxesPerDim] : llvm::zip_equal(
           allReduceInSharding.getDimShardings(), reduceScatterAxes)) {
    SmallVector<AxisRefAttr> expectedDimShardingAxes =
        llvm::to_vector(operandDimSharding.getAxes());
    expectedDimShardingAxes.reserve(expectedDimShardingAxes.size() +
                                    reduceScatterAxesPerDim.size());

    for (AxisRefAttr reduceScatterAxis : reduceScatterAxesPerDim) {
      addAxisOrMerge(expectedDimShardingAxes, reduceScatterAxis, mesh);
    }

    reduceScatterOutDimShardings.push_back(DimensionShardingAttr::get(
        context, expectedDimShardingAxes, /*is_closed=*/true));
  }

  ReduceScatterFusionInfo result;
  result.reduceScatterAxes = std::move(reduceScatterAxes);
  if (hasResidualSlicingAxes) {
    result.residualSlicingAxes = std::move(residualSlicingAxes);
  }
  result.reduceScatterOutSharding = TensorShardingAttr::get(
      context, allReduceOp.getOutShardingAttr().getMeshOrRef(),
      reduceScatterOutDimShardings,
      allSliceOp.getOutSharding().getReplicatedAxes(),
      allSliceOp.getOutSharding().getUnreducedAxes());

  return result;
}

// Pattern to fuse sdy.all_reduce + sdy.all_slice -> sdy.reduce_scatter +
// optional sdy.all_slice.
// For example,
//
// ```mlir
//    %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh, [{}, {}]>
//    %1 = sdy.all_slice [{"x", "z"}, {"y", "p"}] %0 out_sharding=<@mesh,
//    [{"x", "z"}, {"y", "p"}]>
// ```
//
// will be fused into
//
// ```mlir
//    %0 = sdy.reduce_scatter [{"x"}, {"y"}] %arg0 out_sharding=<@mesh, [{"x"},
//      {"y"}]>
//    %1 = sdy.all_slice [{"z"}, {"p"}] %0 out_sharding=<@mesh, [{"x", "z"},
//      {"y", "p"}]>
// ```
class ReduceScatterFusion : public OpRewritePattern<AllSliceOp> {
 public:
  using OpRewritePattern<AllSliceOp>::OpRewritePattern;

 private:
  LogicalResult matchAndRewrite(AllSliceOp allSliceOp,
                                PatternRewriter& rewriter) const override {
    auto allReduceOp = allSliceOp.getTensor().getDefiningOp<AllReduceOp>();
    if (!allReduceOp) {
      return rewriter.notifyMatchFailure(
          allSliceOp, "operand is not defined by sdy.all_reduce");
    }

    // Check if all uses of the allReduceOp are AllSliceOps identical to this
    // one.
    for (Operation* user : allReduceOp->getUsers()) {
      auto userSliceOp = dyn_cast<AllSliceOp>(user);
      if (!userSliceOp) {
        return rewriter.notifyMatchFailure(
            allSliceOp, "sdy.all_reduce has a user that is not sdy.all_slice");
      }
      // Check for identical slicing axes.
      if (userSliceOp.getSlicingAxes() != allSliceOp.getSlicingAxes()) {
        return rewriter.notifyMatchFailure(
            allSliceOp,
            "sdy.all_reduce has an sdy.all_slice user with different slicing "
            "axes");
      }
      if (userSliceOp->isBeforeInBlock(allSliceOp)) {
        return rewriter.notifyMatchFailure(
            allSliceOp,
            "This sdy.all_slice is not the first user of the sdy.all_reduce");
      }
    }

    MLIRContext* context = rewriter.getContext();

    // Call the helper function to perform checks and compute axes.
    std::optional<ReduceScatterFusionInfo> fusionInfo =
        matchAndComputeReduceScatterFusionAxes(allReduceOp, allSliceOp,
                                               rewriter);

    if (!fusionInfo) {
      return rewriter.notifyMatchFailure(
          allSliceOp,
          "Not all reduction axes could be matched as prefixes within the "
          "slicing axes.");
    }

    // Create the new ReduceScatterOp
    auto reduceScatterOp = ReduceScatterOp::create(
        rewriter, allReduceOp.getLoc(), allReduceOp.getTensor(),
        ListOfAxisRefListsAttr::get(context, fusionInfo->reduceScatterAxes),
        fusionInfo->reduceScatterOutSharding);

    Value finalResult = reduceScatterOp.getResult();

    // If there are residual axes, create a new AllSliceOp
    if (fusionInfo->residualSlicingAxes.has_value()) {
      finalResult =
          AllSliceOp::create(
              rewriter, allSliceOp.getLoc(), reduceScatterOp.getResult(),
              ListOfAxisRefListsAttr::get(
                  context, fusionInfo->residualSlicingAxes.value()),
              allSliceOp.getOutSharding())
              .getResult();
    }

    // Replace all identical AllSliceOp users with the final result.
    for (Operation* userSliceOp :
         llvm::make_early_inc_range(allReduceOp->getUsers())) {
      rewriter.replaceOp(userSliceOp, finalResult);
    }

    // Erase the original AllReduceOp. It's safe because all its users have
    // just been replaced.
    rewriter.eraseOp(allReduceOp);

    return success();
  }
};

class AllReduceOfShardedToUnreducedPattern
    : public OpRewritePattern<AllReduceOp> {
 public:
  using OpRewritePattern<AllReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllReduceOp allReduceOp,
                                PatternRewriter& rewriter) const override {
    auto s2uOp =
        allReduceOp.getTensor().getDefiningOp<ShardedToUnreducedOp>();
    if (!s2uOp) {
      return failure();
    }

    SmallVector<AxisRefAttr> s2uAxesVec;
    for (AxisRefListAttr axisRefList : s2uOp.getAxes()) {
      for (AxisRefAttr axisRef : axisRefList.getValue()) {
        s2uAxesVec.push_back(axisRef);
      }
    }
    sortAndMergeAxes(s2uAxesVec, s2uOp.getOutSharding().getMesh(s2uOp));

    if (!llvm::equal(s2uAxesVec, allReduceOp.getReductionAxes())) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<AllGatherOp>(
        allReduceOp, s2uOp.getTensor(), s2uOp.getAxesAttr(),
        allReduceOp.getOutSharding());
    return success();
  }
};

// Returns true if the operation is a metadata-only transformation that we can
// look through to find a redundant communication result.
bool isMetadataOp(Operation* op) {
  return isa<stablehlo::ReshapeOp, stablehlo::TransposeOp>(op);
}

// Helper to follow values upwards through metadata-only transformations to
// find the original data source.
Value unwrapMetadataOp(Value v) {
  while (Operation* op = v.getDefiningOp()) {
    if (!isMetadataOp(op)) {
      break;
    }
    v = op->getOperand(0);
  }
  return v;
}

// Walks up the reshape input chain to account for the effect of reshape and
// transpose operations on the sharding. This allows the re-use of a sharding
// result on a path with transpose operations, as routine
// isShardingEquivalentAcrossReshapes only accounts for the effect of reshape
// operations on the linear data order.
TensorShardingAttr getTargetSharding(TensorShardingAttr sharding, Value v) {
  while (Operation* op = v.getDefiningOp()) {
    if (auto transpose = dyn_cast<stablehlo::TransposeOp>(op)) {
      ArrayRef<int64_t> perm = transpose.getPermutation();
      SDY_CHECK(sharding.getDimShardings().size() == perm.size());

      SmallVector<DimensionShardingAttr> newDims(perm.size());
      for (auto [resDim, opDim] : llvm::enumerate(perm)) {
        newDims[opDim] = sharding.getDimShardings()[resDim];
      }
      sharding = TensorShardingAttr::get(
          sharding.getContext(), sharding.getMeshOrRef(), newDims,
          sharding.getReplicatedAxes(), sharding.getUnreducedAxes());
      v = transpose.getOperand();
      continue;
    }
    if (auto reshape = dyn_cast<stablehlo::ReshapeOp>(op)) {
      if (reshape.getOperand().getType().getRank() !=
          reshape.getType().getRank()) {
        break;
      }
      v = reshape.getOperand();
      continue;
    }
    SDY_CHECK(!isMetadataOp(op));
    break;
  }
  return sharding;
}

// Returns true if two shardings define the same physical linear distribution
// of data across devices, accounting for t1 is a reshape of t2. It works by
// verifying that every mesh axis in the sharding has the same linear stride,
// assuming both tensors share a consistent row-major memory order.
bool isShardingEquivalentAcrossReshapes(TensorShardingAttr s1, Type t1,
                                        TensorShardingAttr s2, Type t2,
                                        Operation* op) {
  if (s1 == s2 && t1 == t2) {
    return true;
  }
  if (!s1 || !s2 || s1.getMeshName() != s2.getMeshName()) {
    return false;
  }
  if (s1.getReplicatedAxes() != s2.getReplicatedAxes()) {
    return false;
  }

  auto rt1 = dyn_cast<RankedTensorType>(t1);
  auto rt2 = dyn_cast<RankedTensorType>(t2);
  if (!rt1 || !rt2) {
    return false;
  }

  // Skip optimization if there is an explicit layout in the tensors.
  // The linear stride calculation below assumes a default row-major layout.
  if (rt1.getEncoding() || rt2.getEncoding()) {
    return false;
  }

  MeshAttr mesh = s1.getMesh(op);
  if (!mesh) {
    return false;
  }

  // Calculates linear stride for every axis in the sharding.
  auto getAxisStrides = [&](TensorShardingAttr sharding,
                            RankedTensorType type) {
    SmallVector<std::pair<AxisRefAttr, int64_t>> axisStrides;
    int64_t cumulativeDimSize = 1;
    for (int64_t i = type.getRank() - 1; i >= 0; --i) {
      ArrayRef<AxisRefAttr> axes = sharding.getDimShardings()[i].getAxes();
      int64_t totalAxesSize = 1;
      for (AxisRefAttr axis : axes) {
        totalAxesSize *= axis.getSize(mesh);
      }
      // We use integer division here. If a dimension is not perfectly
      // divisible by the sharding axes, this truncated stride calculation
      // ensures we only identify shardings as equivalent if the mesh axes map
      // to the exact same linear offsets in the global buffer. Since we also
      // account for the sizes of each dimension and axis, if non-divisibility
      // causes the data to be partitioned differently across a reshape, the
      // strides will correctly mismatch.
      int64_t currentAxisStride =
          cumulativeDimSize *
          llvm::divideCeil(type.getDimSize(i), totalAxesSize);
      for (int64_t j = static_cast<int64_t>(axes.size()) - 1; j >= 0; --j) {
        axisStrides.push_back({axes[j], currentAxisStride});
        currentAxisStride *= axes[j].getSize(mesh);
      }
      cumulativeDimSize *= type.getDimSize(i);
    }
    llvm::stable_sort(axisStrides, [](const auto& a, const auto& b) {
      return a.second < b.second;
    });
    return axisStrides;
  };

  return getAxisStrides(s1, rt1) == getAxisStrides(s2, rt2);
}

// Searches for a compatible reshard result reachable from the root and
// excluding the current reshard. The current implementation only accepts a
// reshard as a cache hit if the path from the root to that reshard contains no
// transposes.
Value findCompatibleReshard(Value root, TensorShardingAttr targetSharding,
                            Type targetType, Block* block, ReshardOp current) {
  // Use a worklist to explore only "safe" paths (reshapes only).
  SmallVector<Value> worklist = {root};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) {
      continue;
    }
    for (Operation* user : v.getUsers()) {
      if (user == current) {
        continue;
      }
      if (auto otherReshard = dyn_cast<ReshardOp>(user)) {
        if (isShardingEquivalentAcrossReshapes(
                otherReshard.getSharding(), otherReshard.getType(),
                targetSharding, targetType, current) &&
            otherReshard->getBlock() == block &&
            otherReshard->isBeforeInBlock(current)) {
          return otherReshard.getResult();
        }
      } else if (!isa<stablehlo::TransposeOp>(user) && isMetadataOp(user)) {
        // Found a metadata-only op that is not a transpose. Follow it to find
        // a compatible reshard.
        worklist.push_back(user->getResult(0));
      }
    }
  }
  return nullptr;
}

// Recursively clones the transformations from the root to the target.
Value cloneTransformationPath(Value root, Value target, Value cachedResult,
                              PatternRewriter& rewriter) {
  if (target == root) {
    return cachedResult;
  }
  Operation* op = target.getDefiningOp();
  assert(isMetadataOp(op) && "Expected a supported metadata op.");

  Value input = op->getOperand(0);
  Value transformedInput =
      cloneTransformationPath(root, input, cachedResult, rewriter);

  rewriter.setInsertionPointAfterValue(transformedInput);
  Operation* cloned = rewriter.clone(*op);
  cloned->setOperand(0, transformedInput);
  return cloned->getResult(0);
}

class ReshardCommonSubexpressionEliminationPattern
    : public OpRewritePattern<ReshardOp> {
 public:
  using OpRewritePattern<ReshardOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshardOp reshardOp,
                                PatternRewriter& rewriter) const override {
    Value input = reshardOp.getInput();
    Value root = unwrapMetadataOp(input);

    TensorShardingAttr targetSharding =
        getTargetSharding(reshardOp.getSharding(), input);

    Value cachedResult =
        findCompatibleReshard(root, targetSharding, reshardOp.getType(),
                              reshardOp->getBlock(), reshardOp);

    if (cachedResult) {
      if (cachedResult.getType() != root.getType()) {
        rewriter.setInsertionPointAfterValue(cachedResult);
        cachedResult = stablehlo::ReshapeOp::create(
            rewriter, reshardOp.getLoc(), root.getType(), cachedResult);
      }
      rewriter.replaceOp(reshardOp, cloneTransformationPath(
                                        root, input, cachedResult, rewriter));
      return success();
    }
    return failure();
  }
};

}  // namespace

void ManualComputationOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
  results.add<ManualComputationUnusedInputsPattern,
              RedundantManualComputationPattern>(context);
}

void ReshardOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<ReshardCommonSubexpressionEliminationPattern>(context);
  results.addWithLabel<ReshardOfReshardPattern>(StringRef(kReshardLabel),
                                                context);
}

void AllGatherOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.addWithLabel<AllGatherNoopPattern>(StringRef(kCollectiveLabel),
                                             context);
}

void AllSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.addWithLabel<AllSliceOfAllGatherPattern, AllSliceNoopPattern>(
      StringRef(kCollectiveLabel), context);
}

void AllReduceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.addWithLabel<AllReduceNoopPattern,
                       AllReduceOfReplicatedToUnreducedPattern,
                       AllReduceOfShardedToUnreducedPattern>(
      StringRef(kCollectiveLabel), context);
}

void AllToAllOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.addWithLabel<AllToAllFusionPattern, AllToAllNoUsePattern>(
      StringRef(kCollectiveLabel), context);
}

void CollectivePermuteOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
  results.addWithLabel<CollectivePermuteNoopPattern>(
      StringRef(kCollectiveLabel), context);
}

void ReduceScatterOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                  MLIRContext* context) {
  results.addWithLabel<ReduceScatterFusion, ReduceScatterNoopPattern>(
      StringRef(kCollectiveLabel), context);
  ;
}

}  // namespace sdy
}  // namespace mlir
