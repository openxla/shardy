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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_OPTIMIZECOLLECTIVESPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// Represents a matched chain of AllToAll operations preceded by a
// CollectivePermute.
struct A2AChain {
  SmallVector<AllToAllOp> a2as;  // In execution order (first to last).
  CollectivePermuteOp cp;
  Value input;
};

// Configuration for splitting tensor dimensions across a matched A2A chain.
struct SplitConfig {
  int64_t splitDim;
  SmallVector<AxisRefAttr> axesToSplit;
  SmallVector<int64_t> splitShape;
  TensorShardingAttr inShardingSplit;
  TensorShardingAttr outShardingSplit;
  SmallVector<int64_t> outPermutation;
  SmallVector<int64_t> transposedSplitShape;
  TensorShardingAttr transposedOutShardingSplit;
};

// Computes the physical shape after splitting `dimIn` by the axes sizes of
// `axesToSplit`. Returns std::nullopt if the dimension is dynamic or not evenly
// divisible by the total axes size.
std::optional<SmallVector<int64_t>> computeSplitShape(
    ArrayRef<int64_t> shape, int64_t dimIn, ArrayRef<AxisRefAttr> axesToSplit,
    MeshAttr mesh) {
  int64_t d0 = shape[dimIn];
  if (ShapedType::isDynamic(d0)) {
    return std::nullopt;
  }

  SmallVector<int64_t> splitShape(shape.begin(), shape.begin() + dimIn);

  int64_t accumSize = 1;
  for (AxisRefAttr axis : llvm::drop_end(axesToSplit)) {
    int64_t groupSize = axis.getSize(mesh);
    splitShape.push_back(groupSize);
    accumSize *= groupSize;
  }

  int64_t totalAxesSize = accumSize * axesToSplit.back().getSize(mesh);
  if (d0 % totalAxesSize != 0) {
    return std::nullopt;
  }

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

  auto splitDims = llvm::map_to_vector(newDimAxes, [&](const auto& axes) {
    return DimensionShardingAttr::get(context, axes, isClosed);
  });

  dimShardings.erase(dimShardings.begin() + dim);
  dimShardings.insert(dimShardings.begin() + dim, splitDims.begin(),
                      splitDims.end());

  return TensorShardingAttr::get(context, sharding.getMeshOrRef(), dimShardings,
                                 sharding.getReplicatedAxes(),
                                 sharding.getUnreducedAxes());
}

// Applies a dimension split to `splitShape` and updates `inSharding` and
// `outSharding` by expanding the split dimension `dim` into multiple dimensions
// corresponding to `axesToSplit`. Returns true on success.
bool applyDimensionSplit(int64_t dim, ArrayRef<AxisRefAttr> axesToSplit,
                         MeshAttr mesh, SmallVectorImpl<int64_t>& splitShape,
                         TensorShardingAttr& inSharding,
                         TensorShardingAttr& outSharding, MLIRContext* ctx) {
  auto newShape = computeSplitShape(splitShape, dim, axesToSplit, mesh);
  if (!newShape) {
    return false;
  }
  splitShape = *newShape;

  auto inNewDimAxes = llvm::map_to_vector(axesToSplit, [](AxisRefAttr axis) {
    return SmallVector<AxisRefAttr>{axis};
  });

  SmallVector<SmallVector<AxisRefAttr>> outNewDimAxes(axesToSplit.size());
  for (AxisRefAttr axis : outSharding.getDimSharding(dim).getAxes()) {
    const auto* it = llvm::find(axesToSplit, axis);
    if (it == axesToSplit.end()) {
      return false;
    }
    outNewDimAxes[std::distance(axesToSplit.begin(), it)].push_back(axis);
  }

  inSharding = expandShardingDim(inSharding, dim, inNewDimAxes, ctx);
  outSharding = expandShardingDim(outSharding, dim, outNewDimAxes, ctx);
  return true;
}

// Maps a dimension index from the unsplit tensor to the split tensor.
std::optional<int64_t> mapDim(int64_t dim, AxisRefAttr axis, int64_t splitDim,
                              ArrayRef<AxisRefAttr> axesToSplit) {
  if (dim < splitDim) {
    return dim;
  }
  if (dim > splitDim) {
    return dim + axesToSplit.size() - 1;
  }
  const auto* it = llvm::find(axesToSplit, axis);
  if (it == axesToSplit.end()) {
    return std::nullopt;
  }
  return splitDim + std::distance(axesToSplit.begin(), it);
}

bool isIdentityPermutation(ArrayRef<int64_t> perm) {
  for (int64_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != i) return false;
  }
  return true;
}

// Moves `axis` from `srcDim` to `tgtDim` in `sharding`.
// Assumes the dimensions are split and have at most one axis.
TensorShardingAttr moveAxis(TensorShardingAttr sharding, AxisRefAttr axis,
                            int64_t srcDim, int64_t tgtDim, MLIRContext* ctx) {
  auto dimShardings = llvm::to_vector(sharding.getDimShardings());

  // Remove axis from srcDim.
  auto srcAxes = llvm::to_vector(dimShardings[srcDim].getAxes());
  auto* it = llvm::find(srcAxes, axis);
  if (it != srcAxes.end()) {
    srcAxes.erase(it);
  }
  dimShardings[srcDim] = DimensionShardingAttr::get(
      ctx, srcAxes, dimShardings[srcDim].getIsClosed());

  // Add axis to tgtDim.
  auto tgtAxes = llvm::to_vector(dimShardings[tgtDim].getAxes());
  tgtAxes.push_back(axis);
  dimShardings[tgtDim] = DimensionShardingAttr::get(
      ctx, tgtAxes, dimShardings[tgtDim].getIsClosed());

  return TensorShardingAttr::get(ctx, sharding.getMeshOrRef(), dimShardings,
                                 sharding.getReplicatedAxes(),
                                 sharding.getUnreducedAxes());
}

// Validates and extracts a chain of AllToAll operations ending at
// `terminalOp`.
std::optional<A2AChain> extractA2AChain(AllToAllOp terminalOp) {
  for (OpOperand& use : terminalOp->getUses()) {
    if (isa<AllToAllOp>(use.getOwner())) {
      return std::nullopt;
    }
  }

  SmallVector<AllToAllOp> a2as;
  AllToAllOp currA2a = terminalOp;
  while (currA2a) {
    a2as.push_back(currA2a);
    currA2a = currA2a.getTensor().getDefiningOp<AllToAllOp>();
  }

  // All intermediate A2As must have a single use (by the next A2A in chain).
  for (size_t i = 1; i < a2as.size(); ++i) {
    if (!a2as[i]->hasOneUse()) {
      return std::nullopt;
    }
  }

  AllToAllOp firstA2a = a2as.back();
  auto cp = firstA2a.getTensor().getDefiningOp<CollectivePermuteOp>();
  if (!cp || !cp->hasOneUse()) {
    return std::nullopt;
  }

  // Guard: Each A2A must have a single parameter with a single axis.
  for (AllToAllOp a2a : a2as) {
    if (a2a.getParams().size() != 1 ||
        a2a.getParams()[0].getAxes().size() != 1) {
      return std::nullopt;
    }
  }

  std::reverse(a2as.begin(), a2as.end());
  return A2AChain{std::move(a2as), cp, cp.getTensor()};
}

// Analyzes the splittable dimension and shardings for a matched chain.
std::optional<SplitConfig> analyzeSplit(const A2AChain& chain,
                                        TensorShardingAttr outSharding,
                                        MLIRContext* ctx) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(chain.input.getType());
  if (!tensorType) {
    return std::nullopt;
  }

  TensorShardingAttr inSharding =
      getOrCreateSharding(chain.input, outSharding.getMeshName());
  MeshAttr mesh = inSharding.getMesh(chain.cp);
  if (!mesh) {
    return std::nullopt;
  }

  SmallVector<AxisList> inAxesPerDim = getAxesPerDim<AxisList>(inSharding);
  SmallVector<AxisList> outAxesPerDim = getAxesPerDim<AxisList>(outSharding);
  alignSubAxesByDecomposition(inAxesPerDim, outAxesPerDim, mesh);
  inSharding = updateSharding(inSharding, inAxesPerDim);
  outSharding = updateSharding(outSharding, outAxesPerDim);

  std::optional<int64_t> splitDim;
  for (int64_t i = 0; i < inSharding.getRank(); ++i) {
    if (!ShapedType::isDynamic(tensorType.getDimSize(i)) &&
        inSharding.getDimSharding(i).getAxes().size() >= 2) {
      splitDim = i;
      break;
    }
  }

  if (!splitDim) {
    return std::nullopt;
  }

  ArrayRef<AxisRefAttr> axesToSplit =
      inSharding.getDimSharding(*splitDim).getAxes();

  SmallVector<int64_t> splitShape = llvm::to_vector(tensorType.getShape());
  TensorShardingAttr inShardingSplit = inSharding;
  TensorShardingAttr outShardingSplit = outSharding;

  if (!applyDimensionSplit(*splitDim, axesToSplit, mesh, splitShape,
                           inShardingSplit, outShardingSplit, ctx)) {
    return std::nullopt;
  }

  // Pre-validate that all A2A parameters can be mapped to split dimensions.
  for (AllToAllOp a2a : chain.a2as) {
    AllToAllParamAttr param = a2a.getParams()[0];
    AxisRefAttr axis = param.getAxes()[0];
    if (!mapDim(param.getSrcDim(), axis, *splitDim, axesToSplit) ||
        !mapDim(param.getTgtDim(), axis, *splitDim, axesToSplit)) {
      return std::nullopt;
    }
  }

  // Compute the permutation needed for the output transpose.
  // Merging split dimensions back requires them to be in stride-major order to
  // be communication-free. If the A2A operations left replicated dimensions
  // major, we compute a local permutation to swap them. This avoids
  // cross-device network communication.
  int64_t k = axesToSplit.size();
  SmallVector<int64_t> permutedGroup;
  permutedGroup.reserve(k);
  for (int64_t i = 0; i < k; ++i) {
    permutedGroup.push_back(*splitDim + i);
  }

  // Target axes of the merged dimension in outSharding
  ArrayRef<AxisRefAttr> targetAxes =
      outSharding.getDimSharding(*splitDim).getAxes();

  // Sort the split dimensions: sharded dimensions first (in target axes order),
  // followed by replicated dimensions.
  auto getSortKey = [&](int64_t dimIdx) {
    ArrayRef<AxisRefAttr> axes =
        outShardingSplit.getDimSharding(dimIdx).getAxes();
    if (axes.empty()) {
      return targetAxes.size();
    }
    const auto* it = llvm::find(targetAxes, axes[0]);
    return it != targetAxes.end()
               ? static_cast<size_t>(std::distance(targetAxes.begin(), it))
               : targetAxes.size();
  };

  llvm::stable_sort(permutedGroup, [&](int64_t a, int64_t b) {
    return getSortKey(a) < getSortKey(b);
  });

  // Construct the global permutation vector
  SmallVector<int64_t> outPermutation;
  outPermutation.reserve(outShardingSplit.getRank());
  for (int64_t i = 0; i < *splitDim; ++i) {
    outPermutation.push_back(i);
  }
  llvm::append_range(outPermutation, permutedGroup);
  for (int64_t i = *splitDim + k; i < outShardingSplit.getRank(); ++i) {
    outPermutation.push_back(i);
  }

  // Compute transposed shape and sharding to verify equivalence
  SmallVector<int64_t> transposedSplitShape(splitShape.size());
  for (size_t i = 0; i < splitShape.size(); ++i) {
    transposedSplitShape[i] = splitShape[outPermutation[i]];
  }
  auto transposedSplitTensorType =
      RankedTensorType::get(transposedSplitShape, tensorType.getElementType());

  auto splitDimShardings = outShardingSplit.getDimShardings();
  SmallVector<DimensionShardingAttr> transposedDimShardings(
      splitDimShardings.size());
  for (size_t i = 0; i < splitDimShardings.size(); ++i) {
    transposedDimShardings[i] = splitDimShardings[outPermutation[i]];
  }
  TensorShardingAttr transposedOutShardingSplit = TensorShardingAttr::get(
      ctx, outShardingSplit.getMeshOrRef(), transposedDimShardings,
      outShardingSplit.getReplicatedAxes(),
      outShardingSplit.getUnreducedAxes());

  auto splitTensorType =
      RankedTensorType::get(splitShape, tensorType.getElementType());

  if (!isShardingEquivalentAcrossReshapes(
          inSharding, tensorType, inShardingSplit, splitTensorType, chain.cp) ||
      !isShardingEquivalentAcrossReshapes(
          transposedOutShardingSplit, transposedSplitTensorType, outSharding,
          tensorType, chain.a2as.back())) {
    return std::nullopt;
  }

  return SplitConfig{*splitDim,
                     llvm::to_vector(axesToSplit),
                     std::move(splitShape),
                     inShardingSplit,
                     outShardingSplit,
                     std::move(outPermutation),
                     std::move(transposedSplitShape),
                     transposedOutShardingSplit};
}

// Rewrites a matched chain into: reshape -> AllToAll(s) -> reshape.
void rewriteA2AChain(AllToAllOp terminalOp, const A2AChain& chain,
                     const SplitConfig& config, PatternRewriter& rewriter) {
  Location loc = terminalOp.getLoc();
  auto tensorType = mlir::cast<RankedTensorType>(chain.input.getType());
  auto splitTensorType =
      RankedTensorType::get(config.splitShape, tensorType.getElementType());

  // Reshape Input
  auto reshape1 =
      stablehlo::ReshapeOp::create(rewriter, loc, splitTensorType, chain.input);
  setSharding(reshape1, config.inShardingSplit);

  Value lastVal = reshape1;
  TensorShardingAttr currSharding = config.inShardingSplit;

  for (size_t i = 0; i < chain.a2as.size(); ++i) {
    AllToAllOp oldA2a = chain.a2as[i];
    AllToAllParamAttr param = oldA2a.getParams()[0];
    AxisRefAttr axis = param.getAxes()[0];
    int64_t splitSrcDim =
        *mapDim(param.getSrcDim(), axis, config.splitDim, config.axesToSplit);
    int64_t splitTgtDim =
        *mapDim(param.getTgtDim(), axis, config.splitDim, config.axesToSplit);

    TensorShardingAttr nextSharding =
        (i == chain.a2as.size() - 1)
            ? config.outShardingSplit
            : moveAxis(currSharding, axis, splitSrcDim, splitTgtDim,
                       rewriter.getContext());

    auto newParam = AllToAllParamAttr::get(rewriter.getContext(), {axis},
                                           splitSrcDim, splitTgtDim);
    lastVal = AllToAllOp::create(rewriter, loc, splitTensorType, lastVal,
                                 newParam, nextSharding);
    currSharding = nextSharding;
  }

  // Reshape Output. If the split dimensions are not in stride order, insert a
  // local transpose first to swap them on-device. This ensures the final
  // reshape is communication-free.
  Value outputVal = lastVal;
  if (!isIdentityPermutation(config.outPermutation)) {
    auto transposedSplitTensorType = RankedTensorType::get(
        config.transposedSplitShape, tensorType.getElementType());
    auto transpose =
        stablehlo::TransposeOp::create(rewriter, loc, transposedSplitTensorType,
                                       lastVal, config.outPermutation);
    setSharding(transpose, config.transposedOutShardingSplit);
    outputVal = transpose;
  }

  auto reshape2 =
      stablehlo::ReshapeOp::create(rewriter, loc, tensorType, outputVal);
  setSharding(reshape2, terminalOp.getOutSharding());

  rewriter.replaceAllUsesWith(terminalOp, reshape2);
  for (AllToAllOp a2a : llvm::reverse(chain.a2as)) {
    rewriter.eraseOp(a2a);
  }
  rewriter.eraseOp(chain.cp);
}

struct RewriteA2AChain : public OpRewritePattern<AllToAllOp> {
  using OpRewritePattern<AllToAllOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllToAllOp op,
                                PatternRewriter& rewriter) const override {
    std::optional<A2AChain> chain = extractA2AChain(op);
    if (!chain) {
      return failure();
    }

    std::optional<SplitConfig> config =
        analyzeSplit(*chain, op.getOutSharding(), rewriter.getContext());
    if (!config) {
      return failure();
    }

    rewriteA2AChain(op, *chain, *config, rewriter);
    return success();
  }
};

class OptimizeCollectivesPass
    : public impl::OptimizeCollectivesPassBase<OptimizeCollectivesPass> {
 protected:
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RewriteA2AChain>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace sdy
}  // namespace mlir
