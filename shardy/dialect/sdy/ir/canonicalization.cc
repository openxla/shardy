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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
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

// Returns true if folding tensor -> innerReshard -> outerSharding into
// tensor -> outerSharding is forbidden. Specifically:
// 1. Single-device sharding transitions must strictly pivot through
//    an intermediate fully-replicated sharding on the global mesh.
// 2. Cross-mesh transitions between different meshes must strictly pivot
//    through intermediate replicated shardings. Folding across different meshes
//    would produce forbidden direct cross-mesh transitions.
bool isForbiddenIntermediateReshard(Operation* op, Value tensor,
                                    TensorShardingAttr outerSharding) {
  auto innerReshard = dyn_cast_or_null<ReshardOp>(op);
  if (!innerReshard) {
    return false;
  }
  TensorShardingAttr inSharding = getSharding(tensor);
  TensorShardingAttr innerSharding = innerReshard.getSharding();
  if (!inSharding || !innerSharding || !outerSharding) {
    return false;
  }

  bool inIsSingleDevice = isSingleDeviceSharding(inSharding, op);
  bool outIsSingleDevice = isSingleDeviceSharding(outerSharding, op);

  // Single-device transitions: folding is forbidden if it would directly
  // reshard between a single-device sharding and a tiled sharding.
  if (inIsSingleDevice != outIsSingleDevice) {
    TensorShardingAttr otherSharding =
        inIsSingleDevice ? outerSharding : inSharding;
    return !isFullyReplicatedWithoutUnreducedAxes(otherSharding);
  }
  if (inIsSingleDevice && outIsSingleDevice) {
    // Prevent folding a multi-device bridge between two different
    // single-device shardings into a direct cross-device reshard.
    return inSharding.getMeshOrRef() != outerSharding.getMeshOrRef();
  }

  // Cross-mesh: prevent folding across different meshes unless all three
  // shardings are fully replicated without unreduced axes.
  bool differentMeshes =
      inSharding.getMeshOrRef() != outerSharding.getMeshOrRef() ||
      inSharding.getMeshOrRef() != innerSharding.getMeshOrRef() ||
      innerSharding.getMeshOrRef() != outerSharding.getMeshOrRef();
  if (differentMeshes) {
    return !isFullyReplicatedWithoutUnreducedAxes(inSharding) ||
           !isFullyReplicatedWithoutUnreducedAxes(innerSharding) ||
           !isFullyReplicatedWithoutUnreducedAxes(outerSharding);
  }
  return false;
}

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
      SDY_CHECK(mesh) << "expected inputs and outputs to have a common mesh";
      for (AxisRefAttr manualAxis : manualComputationOp.getManualAxes()) {
        manualAxesProduct *= manualAxis.getSize(mesh);
      }
    }
    bool isTrivial = manualAxesProduct == 1;
    if (!isTrivial) {
      // Check whether operands are trivially-manual by shapes.
      isTrivial = llvm::all_of(
          llvm::zip(manualComputationOp.getOperands(),
                    manualComputationOp.getRegion().getArgumentTypes()),
          [](auto pair) {
            return std::get<0>(pair).getType() == std::get<1>(pair);
          });
      if (isTrivial) {
        Operation* terminator =
            manualComputationOp.getBody().front().getTerminator();
        // Check whether results are trivially-manual by shapes.
        isTrivial = llvm::all_of(
            llvm::zip(terminator->getOperandTypes(),
                      manualComputationOp.getResultTypes()),
            [](auto pair) { return std::get<0>(pair) == std::get<1>(pair); });
      }
    }

    if (!isTrivial) {
      return rewriter.notifyMatchFailure(
          manualComputationOp, [](Diagnostic& diag) {
            diag << "manual computation is not trivial and product of manual "
                    "axis sizes is not 1";
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
  ReductionOp reductionOp;
};

// Permutes sharding's dimension shardings forward according to a
// stablehlo.transpose permutation (outDim reads from
// inDim = permutation[outDim]), preserving mesh and tensor-level axes.
TensorShardingAttr permuteShardingForward(TensorShardingAttr sharding,
                                          ArrayRef<int64_t> permutation) {
  SDY_CHECK_EQ(sharding.getDimShardings().size(), permutation.size());
  SmallVector<DimensionShardingAttr> newDimShardings;
  newDimShardings.reserve(permutation.size());
  for (int64_t inDim : permutation) {
    newDimShardings.push_back(sharding.getDimSharding(inDim));
  }
  return sharding.replaceDimShardings(newDimShardings);
}

// Permutes per-dimension attributes backward from a stablehlo.transpose
// result space to its operand space (inDim = permutation[outDim]).
template <typename T>
SmallVector<T> permuteBackward(ArrayRef<T> values,
                               ArrayRef<int64_t> permutation) {
  SDY_CHECK_EQ(values.size(), permutation.size());
  SmallVector<T> result(permutation.size());
  for (auto [outDim, inDim] : llvm::enumerate(permutation)) {
    result[inDim] = values[outDim];
  }
  return result;
}

// Permutes sharding's dimension shardings backward across a
// stablehlo.transpose (inDim = permutation[outDim]), preserving mesh and
// tensor-level axes.
TensorShardingAttr permuteShardingBackward(TensorShardingAttr sharding,
                                           ArrayRef<int64_t> permutation) {
  return sharding.replaceDimShardings(
      permuteBackward(sharding.getDimShardings(), permutation));
}

// Returns true if every stablehlo.transpose in transposes (ordered from
// slice-adjacent to allReduceOp-adjacent) preserves allReduceOutSharding
// (up to closedness and replicated axes) and allReduceMesh under its
// permutation.
bool isTransposeChainPreservingSharding(
    TensorShardingAttr allReduceOutSharding, MeshAttr allReduceMesh,
    ArrayRef<stablehlo::TransposeOp> transposes) {
  TensorShardingAttr expectedSharding = allReduceOutSharding;
  for (stablehlo::TransposeOp transposeOp : llvm::reverse(transposes)) {
    expectedSharding =
        permuteShardingForward(expectedSharding, transposeOp.getPermutation());
    TensorShardingAttr actualSharding = getSharding(transposeOp.getResult());
    if (!actualSharding) {
      if (!isFullyReplicatedWithoutUnreducedAxes(expectedSharding)) {
        return false;
      }
      continue;
    }
    if (actualSharding.getMesh(transposeOp) != allReduceMesh ||
        actualSharding.getUnreducedAxes() !=
            expectedSharding.getUnreducedAxes() ||
        actualSharding.getReductionOp() != expectedSharding.getReductionOp() ||
        actualSharding.getRank() != expectedSharding.getRank() ||
        !llvm::equal(actualSharding.getDimShardings(),
                     expectedSharding.getDimShardings(),
                     [](DimensionShardingAttr a, DimensionShardingAttr b) {
                       return a.getAxes() == b.getAxes();
                     })) {
      return false;
    }
  }
  return true;
}

// Helper function to check if the reduce scatter fusion is possible and compute
// the axes. Returns std::nullopt on failure.
std::optional<ReduceScatterFusionInfo> matchAndComputeReduceScatterFusionAxes(
    AllReduceOp allReduceOp, ArrayRef<AxisRefListAttr> slicingAxes,
    TensorShardingAttr sliceOutSharding, PatternRewriter& rewriter) {
  MLIRContext* context = rewriter.getContext();
  MeshAttr mesh = allReduceOp.getOutSharding().getMesh(allReduceOp);
  ArrayRef<AxisRefAttr> reductionAxes = allReduceOp.getReductionAxes();
  llvm::SmallSetVector<AxisRefAttr, 4> remainingReductionAxes(
      reductionAxes.begin(), reductionAxes.end());

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

    // Find the longest prefix of slicing axes for this dimension that can be
    // consumed from the remaining reduction axes (including sub-axis prefixes).
    while (it != end) {
      AxisRefAttr slicingAxis = *it;
      if (remainingReductionAxes.remove(slicingAxis)) {
        reduceScatterAxesPerDim.push_back(slicingAxis);
        ++it;
        continue;
      }
      AxisRefAttr matchedRedAxis;
      AxisRefAttr overlapAxis;
      for (AxisRefAttr redAxis : remainingReductionAxes) {
        if (std::optional<AxisRefAttr> overlap =
                redAxis.getOverlap(slicingAxis);
            overlap && overlap->prefixOf(slicingAxis)) {
          matchedRedAxis = redAxis;
          overlapAxis = *overlap;
          break;
        }
      }
      if (!matchedRedAxis) {
        break;
      }
      reduceScatterAxesPerDim.push_back(overlapAxis);
      remainingReductionAxes.remove(matchedRedAxis);
      if (std::optional<AxisRefAttr> redPrefix =
              matchedRedAxis.getPrefixWithoutOverlap(overlapAxis)) {
        remainingReductionAxes.insert(*redPrefix);
      }
      if (std::optional<AxisRefAttr> redSuffix =
              matchedRedAxis.getSuffixWithoutOverlap(overlapAxis, mesh)) {
        remainingReductionAxes.insert(*redSuffix);
      }
      ++it;
      if (std::optional<AxisRefAttr> slicingSuffix =
              slicingAxis.getSuffixWithoutOverlap(overlapAxis, mesh)) {
        residualSlicingAxesPerDim.push_back(*slicingSuffix);
        break;
      }
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

  ArrayRef<int64_t> tensorShape = getTensorShape(allReduceOp.getResult());
  for (int64_t d = 0; d < rank; ++d) {
    if (residualSlicingAxes[d].empty() || tensorShape[d] <= 0) {
      continue;
    }
    int64_t rsOutShardedSize =
        reduceScatterOutDimShardings[d].getShardedSize(mesh);
    int64_t finalOutShardedSize =
        sliceOutSharding.getDimSharding(d).getShardedSize(mesh);
    if (rsOutShardedSize > 1 &&
        llvm::alignTo(tensorShape[d], rsOutShardedSize) !=
            llvm::alignTo(tensorShape[d], finalOutShardedSize)) {
      return std::nullopt;
    }
  }

  ReduceScatterFusionInfo result;
  result.reduceScatterAxes = std::move(reduceScatterAxes);
  if (hasResidualSlicingAxes) {
    result.residualSlicingAxes = std::move(residualSlicingAxes);
  }
  result.reductionOp = allReduceOp.getReductionOp();
  result.reduceScatterOutSharding = TensorShardingAttr::get(
      context, allReduceOp.getOutShardingAttr().getMeshOrRef(),
      reduceScatterOutDimShardings, sliceOutSharding.getReplicatedAxes(),
      sliceOutSharding.getUnreducedAxes(), sliceOutSharding.getReductionOp());

  return result;
}

// Pattern to fuse sdy.all_reduce + (optional stablehlo.transpose chain) +
// sdy.all_slice -> sdy.reduce_scatter + optional sdy.all_slice +
// (optional stablehlo.transpose chain).
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

  LogicalResult matchAndRewrite(AllSliceOp allSliceOp,
                                PatternRewriter& rewriter) const override {
    // Walk backward through any chain of stablehlo.transpose ops from the
    // slice operand back to a defining sdy.all_reduce. transposes is ordered
    // from slice-adjacent (front()) to allReduce-adjacent (back()).
    SmallVector<stablehlo::TransposeOp> transposes;
    llvm::SmallPtrSet<Operation*, 4> visited;
    Value curOperand = allSliceOp.getTensor();
    while (auto transposeOp =
               curOperand.getDefiningOp<stablehlo::TransposeOp>()) {
      if (!visited.insert(transposeOp).second) {
        return rewriter.notifyMatchFailure(allSliceOp,
                                           "cyclic stablehlo.transpose chain");
      }
      transposes.push_back(transposeOp);
      curOperand = transposeOp.getOperand();
    }

    auto allReduceOp = curOperand.getDefiningOp<AllReduceOp>();
    if (!allReduceOp) {
      return rewriter.notifyMatchFailure(
          allSliceOp, "operand is not defined by sdy.all_reduce");
    }

    if (!transposes.empty()) {
      if (!allReduceOp->hasOneUse()) {
        return rewriter.notifyMatchFailure(
            allSliceOp,
            "sdy.all_reduce with intervening transpose must have a single use");
      }
      for (stablehlo::TransposeOp intermediateTranspose :
           llvm::drop_begin(transposes)) {
        if (!intermediateTranspose->hasOneUse()) {
          return rewriter.notifyMatchFailure(
              allSliceOp,
              "intermediate stablehlo.transpose must have a single use");
        }
      }
    }

    // Check if all uses of the root op directly feeding allSliceOp are
    // identical AllSliceOps.
    Operation* rootBeforeSlice = transposes.empty()
                                     ? allReduceOp.getOperation()
                                     : transposes.front().getOperation();
    MeshAttr allReduceMesh = allReduceOp.getOutSharding().getMesh(allReduceOp);
    for (Operation* user : rootBeforeSlice->getUsers()) {
      auto userSliceOp = dyn_cast<AllSliceOp>(user);
      if (!userSliceOp) {
        return rewriter.notifyMatchFailure(
            allSliceOp, "root op has a user that is not sdy.all_slice");
      }
      // An sdy.all_slice of a fully replicated tensor may switch meshes, which
      // an sdy.reduce_scatter on the sdy.all_reduce mesh cannot express.
      if (userSliceOp.getOutSharding().getMesh(userSliceOp) != allReduceMesh) {
        return rewriter.notifyMatchFailure(
            allSliceOp,
            "root op has an sdy.all_slice user on a different mesh");
      }
      // Check for identical slicing axes.
      if (userSliceOp.getSlicingAxes() != allSliceOp.getSlicingAxes()) {
        return rewriter.notifyMatchFailure(
            allSliceOp,
            "root op has an sdy.all_slice user with different slicing axes");
      }
      if (userSliceOp->getBlock() != allSliceOp->getBlock() ||
          userSliceOp->isBeforeInBlock(allSliceOp)) {
        return rewriter.notifyMatchFailure(
            allSliceOp,
            "this sdy.all_slice is not the first user in its block");
      }
    }

    if (!isTransposeChainPreservingSharding(allReduceOp.getOutSharding(),
                                            allReduceMesh, transposes)) {
      return rewriter.notifyMatchFailure(
          allSliceOp,
          "intervening stablehlo.transpose alters sharding or mesh");
    }

    MLIRContext* context = rewriter.getContext();

    // Pull allSliceOp's slicing axes and output sharding backward across the
    // transpose chain to the pre-transpose tensor space of allReduceOp.
    SmallVector<AxisRefListAttr> preTransposeSlicingAxes =
        llvm::to_vector(allSliceOp.getSlicingAxes());
    TensorShardingAttr preTransposeSliceOutSharding =
        allSliceOp.getOutSharding();
    for (stablehlo::TransposeOp transposeOp : transposes) {
      ArrayRef<int64_t> permutation = transposeOp.getPermutation();
      preTransposeSlicingAxes = permuteBackward<AxisRefListAttr>(
          preTransposeSlicingAxes, permutation);
      preTransposeSliceOutSharding =
          permuteShardingBackward(preTransposeSliceOutSharding, permutation);
    }

    std::optional<ReduceScatterFusionInfo> fusionInfo =
        matchAndComputeReduceScatterFusionAxes(
            allReduceOp, preTransposeSlicingAxes, preTransposeSliceOutSharding,
            rewriter);

    if (!fusionInfo) {
      return rewriter.notifyMatchFailure(
          allSliceOp,
          "Not all reduction axes could be matched as prefixes within the "
          "slicing axes.");
    }

    // Create the new ReduceScatterOp (and optional residual AllSliceOp) before
    // the original AllReduceOp so they dominate any intervening transposes.
    rewriter.setInsertionPoint(allReduceOp);
    auto reduceScatterOp = ReduceScatterOp::create(
        rewriter, allReduceOp.getLoc(), allReduceOp.getTensor(),
        ListOfAxisRefListsAttr::get(context, fusionInfo->reduceScatterAxes),
        fusionInfo->reductionOp, fusionInfo->reduceScatterOutSharding);

    Value preTransposeResult = reduceScatterOp.getResult();
    TensorShardingAttr preTransposeSharding =
        fusionInfo->reduceScatterOutSharding;

    if (fusionInfo->residualSlicingAxes.has_value()) {
      preTransposeSharding = preTransposeSliceOutSharding;
      preTransposeResult =
          AllSliceOp::create(
              rewriter, allSliceOp.getLoc(), reduceScatterOp.getResult(),
              ListOfAxisRefListsAttr::get(
                  context, fusionInfo->residualSlicingAxes.value()),
              preTransposeSharding)
              .getResult();
    }

    // Rewire and update sharding forward along the transpose chain so the
    // transposes now execute after reduce-scatter (and any residual all-slice).
    Value curValue = preTransposeResult;
    TensorShardingAttr curSharding = preTransposeSharding;
    for (stablehlo::TransposeOp transposeOp : llvm::reverse(transposes)) {
      curSharding =
          permuteShardingForward(curSharding, transposeOp.getPermutation());
      rewriter.modifyOpInPlace(transposeOp, [&] {
        if (transposeOp.getOperand() != curValue) {
          transposeOp->setOperand(0, curValue);
        }
        setSharding(transposeOp.getResult(), curSharding);
      });
      curValue = transposeOp.getResult();
    }

    // Replace all identical AllSliceOp users with the final result.
    for (Operation* userSliceOp :
         llvm::make_early_inc_range(rootBeforeSlice->getUsers())) {
      rewriter.replaceOp(userSliceOp, curValue);
    }

    // Erase the original AllReduceOp. All its users have been replaced or
    // rewired to preTransposeResult.
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
    auto s2uOp = allReduceOp.getTensor().getDefiningOp<ShardedToUnreducedOp>();
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

    rewriter.replaceOpWithNewOp<AllGatherOp>(allReduceOp, s2uOp.getTensor(),
                                             s2uOp.getAxesAttr(),
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
      sharding = permuteShardingBackward(sharding, transpose.getPermutation());
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

// Folds an AllGatherOp into a single-use BroadcastInDimOp when all gathering
// dimensions are newly broadcasted dimensions (not in broadcast_dimensions).
class AllGatherOfBroadcastInDimPattern : public OpRewritePattern<AllGatherOp> {
 public:
  using OpRewritePattern<AllGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllGatherOp allGatherOp,
                                PatternRewriter& rewriter) const override {
    auto bcastOp =
        allGatherOp.getTensor().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!bcastOp || !bcastOp->hasOneUse()) {
      return failure();
    }
    ArrayRef<int64_t> bcastDims = bcastOp.getBroadcastDimensions();
    for (auto [d, axes] : llvm::enumerate(allGatherOp.getGatheringAxes())) {
      if (!axes.empty() && llvm::is_contained(bcastDims, d)) {
        return failure();
      }
    }
    rewriter.modifyOpInPlace(bcastOp, [&]() {
      setSharding(bcastOp.getResult(), allGatherOp.getOutSharding());
    });
    rewriter.replaceOp(allGatherOp, bcastOp.getResult());
    return success();
  }
};

// Folds a collective op into a single-use IotaOp by updating the IotaOp
// sharding.
template <typename CollectiveOpTy>
class CollectiveOfIotaPattern : public OpRewritePattern<CollectiveOpTy> {
 public:
  using OpRewritePattern<CollectiveOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(CollectiveOpTy collectiveOp,
                                PatternRewriter& rewriter) const override {
    auto iotaOp =
        collectiveOp.getTensor().template getDefiningOp<stablehlo::IotaOp>();
    if (!iotaOp || !iotaOp->hasOneUse()) {
      return failure();
    }
    rewriter.modifyOpInPlace(iotaOp, [&]() {
      setSharding(iotaOp.getResult(), collectiveOp.getOutSharding());
    });
    rewriter.replaceOp(collectiveOp, iotaOp.getResult());
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
  results.add<ReshardCommonSubexpressionEliminationPattern>(context);
  results.addWithLabel<ReshardOfReshardPattern>(StringRef(kReshardLabel),
                                                context);
}

void AllGatherOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                              MLIRContext* context) {
  results.addWithLabel<AllGatherNoopPattern, AllGatherOfBroadcastInDimPattern>(
      StringRef(kCollectiveLabel), context);
  results.addWithLabel<CollectiveOfIotaPattern<AllGatherOp>>(
      StringRef(kIotaCollectiveLabel), context);
}

void AllSliceOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                             MLIRContext* context) {
  results.addWithLabel<AllSliceOfAllGatherPattern, AllSliceNoopPattern>(
      StringRef(kCollectiveLabel), context);
  results.addWithLabel<CollectiveOfIotaPattern<AllSliceOp>>(
      StringRef(kIotaCollectiveLabel), context);
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
  results.addWithLabel<CollectiveOfIotaPattern<AllToAllOp>>(
      StringRef(kIotaCollectiveLabel), context);
}

void CollectivePermuteOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
  results.addWithLabel<CollectivePermuteNoopPattern>(
      StringRef(kCollectiveLabel), context);
  results.addWithLabel<CollectiveOfIotaPattern<CollectivePermuteOp>>(
      StringRef(kIotaCollectiveLabel), context);
}

void ReduceScatterOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                  MLIRContext* context) {
  results.addWithLabel<ReduceScatterFusion, ReduceScatterNoopPattern>(
      StringRef(kCollectiveLabel), context);
  ;
}

}  // namespace sdy
}  // namespace mlir
