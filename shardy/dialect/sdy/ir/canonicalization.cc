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
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

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

// Pattern to inline a ManualComputationOp when the product of all manual axes
// is 1.
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
      reduceScatterOutDimShardings, allReduceInSharding.getReplicatedAxes());

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
    auto reduceScatterOp = rewriter.create<ReduceScatterOp>(
        allReduceOp.getLoc(), allReduceOp.getTensor(),
        ListOfAxisRefListsAttr::get(context, fusionInfo->reduceScatterAxes),
        fusionInfo->reduceScatterOutSharding);

    Value finalResult = reduceScatterOp.getResult();

    // If there are residual axes, create a new AllSliceOp
    if (fusionInfo->residualSlicingAxes.has_value()) {
      finalResult =
          rewriter
              .create<AllSliceOp>(
                  allSliceOp.getLoc(), reduceScatterOp.getResult(),
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

class AlltoAllFusion : public OpRewritePattern<AllToAllOp> {
 public:
  using OpRewritePattern<AllToAllOp>::OpRewritePattern;

 private:
  LogicalResult matchAndRewrite(AllToAllOp allToAllOp,
                                PatternRewriter& rewriter) const override {
    if (range_size(allToAllOp->getUsers()) != 1) {
      return rewriter.notifyMatchFailure(
          allToAllOp, "op has multiple users");
    }
    auto userAllToAllOp = dyn_cast<AllToAllOp>(*allToAllOp->user_begin());
    if (!userAllToAllOp) {
      return rewriter.notifyMatchFailure(
          allToAllOp, "user is not all-to-all");
    }
    // Combine the params of the two all-to-all ops into one.
    SmallVector<AllToAllParamAttr> combinedParams;
    combinedParams.reserve(allToAllOp.getParams().size() +
                           userAllToAllOp.getParams().size());
    combinedParams.append(allToAllOp.getParams().begin(),
                          allToAllOp.getParams().end());
    combinedParams.append(userAllToAllOp.getParams().begin(),
                          userAllToAllOp.getParams().end());
    // Check for overlap in the source and target dimensions.
    BitVector seenDims(getTensorRank(allToAllOp.getResult()));
    for (AllToAllParamAttr param : combinedParams) {
      for (int64_t dim : {param.getSrcDim(), param.getTgtDim()}) {
        if (seenDims.test(dim)) {
          return rewriter.notifyMatchFailure(
              allToAllOp, "overlapping dimensions in the combined parameters");
        }
        seenDims.set(dim);
      }
    }
    llvm::sort(combinedParams,
               [](const AllToAllParamAttr& a, const AllToAllParamAttr& b) {
                 return a.getSrcDim() < b.getSrcDim();
               });
    rewriter.replaceOpWithNewOp<AllToAllOp>(
        userAllToAllOp, userAllToAllOp.getResult().getType(),
        allToAllOp.getTensor(), combinedParams,
        userAllToAllOp.getOutSharding());
    rewriter.eraseOp(allToAllOp);
    return success();
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
  results.add<ReshardOfReshardPattern>(context);
}

void AllGatherOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<AllGatherNoopPattern>(context);
}

void AllSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<AllSliceOfAllGatherPattern>(context);
  results.add<AllSliceNoopPattern>(context);
}

void AllReduceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.add<AllReduceNoopPattern>(context);
}

void AllToAllOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.add<AlltoAllFusion>(context);
}

void CollectivePermuteOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
  results.add<CollectivePermuteNoopPattern>(context);
}

void ReduceScatterOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                  MLIRContext* context) {
  results.add<ReduceScatterFusion>(context);
}

}  // namespace sdy
}  // namespace mlir
