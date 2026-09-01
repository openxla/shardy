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

#include "shardy/dialect/sdy/transforms/export/optimize_collectives_util.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

namespace {

// Returns true if at least one axis in `inAxes` that is permuted in `cpAxes` is
// present in `communicatedAxes`.
//
// Example:
// - inAxes=[{"x"}, {"y"}], cpAxes=[{"y"}, {"x"}], communicatedAxes=[{"x"}]
//   -> returns true
bool communicatesAnyPermutedAxis(ArrayRef<AxisRefAttr> inAxes,
                                 ArrayRef<AxisRefAttr> cpAxes,
                                 ArrayRef<AxisRefAttr> communicatedAxes) {
  for (auto [inAxis, cpAxis] : llvm::zip(inAxes, cpAxes)) {
    // Only axes whose position or presence changed on the split dimension are
    // considered permuted.
    if (inAxis != cpAxis) {
      // Checks whether this permuted axis is communicated off splitDim.
      AxisRefAttr permutedAxis = inAxis;
      if (llvm::any_of(communicatedAxes, [&](AxisRefAttr commAxis) {
            return commAxis.contains(permutedAxis);
          })) {
        return true;
      }
    }
  }
  return false;
}

// Builds an AllToAllStage representing one step in the optimized all-to-all
// sequence.
//
// Example:
// - inSharding: <@mesh, [{"x", "y"}, {}, {}]>, inType: tensor<2x2x8x8xf32>
// - params: [{"x"}: 0->1, {"y"}: 0->2]
// - Returns AllToAllStage with outType=tensor<2x2x8x8xf32> and
//   outSharding=<@mesh, [{}, {"x"}, {"y"}]>.
AllToAllStage computeAllToAllStage(TensorShardingAttr inSharding,
                                   RankedTensorType inType,
                                   AllToAllParamListAttr params) {
  MLIRContext* ctx = inSharding.getContext();
  SmallVector<DimensionShardingAttr> outDimShardings(
      inSharding.getDimShardings().begin(), inSharding.getDimShardings().end());
  for (AllToAllParamAttr param : params) {
    int64_t src = param.getSrcDim();
    int64_t tgt = param.getTgtDim();
    outDimShardings[src] =
        DimensionShardingAttr::get(ctx, {}, /*isClosed=*/true);
    SmallVector<AxisRefAttr> tgtAxes(outDimShardings[tgt].axis_begin(),
                                     outDimShardings[tgt].axis_end());
    for (AxisRefAttr ax : param.getAxes()) {
      tgtAxes.push_back(ax);
    }
    outDimShardings[tgt] = DimensionShardingAttr::get(
        ctx, tgtAxes, outDimShardings[tgt].getIsClosed());
  }
  auto outSharding = TensorShardingAttr::get(
      ctx, inSharding.getMeshOrRef(), outDimShardings,
      inSharding.getReplicatedAxes(), inSharding.getUnreducedAxes());
  return AllToAllStage{params, outSharding, inType};
}

// Validates the fundamental type and sharding preconditions for `cpOp`:
// static shape and rank >= 1 on input tensor (required for shape
// decomposition), presence of input and output shardings on
// CollectivePermuteOp, and matching mesh reference and resolvable MeshAttr
// in the SymbolTable.
//
// Examples:
// - input: tensor<16x8xf32> on @mesh -> returns true
// - input: tensor<?x8xf32> (dynamic)  -> returns false
// - @mesh not in SymbolTable         -> returns false
bool isSupportedTensorAndMesh(CollectivePermuteOp cpOp,
                              const SymbolTable& symbolTable) {
  Value inputTensor = cpOp.getTensor();
  auto inputType = dyn_cast<RankedTensorType>(inputTensor.getType());
  if (!inputType || !inputType.hasStaticShape() || inputType.getRank() == 0) {
    return false;
  }
  TensorShardingAttr inSharding = getSharding(inputTensor);
  TensorShardingAttr cpOutSharding = cpOp.getOutSharding();
  if (!inSharding || !cpOutSharding) {
    return false;
  }
  if (inSharding.getMeshOrRef() != cpOutSharding.getMeshOrRef() ||
      !inSharding.getMesh(symbolTable)) {
    return false;
  }
  return true;
}

// Returns true if `cpAxes` is a pure permutation of `inAxes` (i.e. contains
// the exact same set of at least two axes in a rearranged order).
//
// Examples:
// - inAxes=["x", "y"], cpAxes=["y", "x"] -> returns true
// - inAxes=["x", "y"], cpAxes=["y", "z"] -> returns false (axes differ)
// - inAxes=["x"], cpAxes=["x"]           -> returns false (< 2 axes)
bool isPurePermutation(const AxisList& inAxes, const AxisList& cpAxes) {
  if (inAxes.size() < 2 || inAxes.size() != cpAxes.size()) {
    return false;
  }
  SmallVector<AxisRefAttr> sortedInAxes(inAxes.begin(), inAxes.end());
  SmallVector<AxisRefAttr> sortedCpAxes(cpAxes.begin(), cpAxes.end());
  llvm::sort(sortedInAxes);
  llvm::sort(sortedCpAxes);
  return sortedInAxes == sortedCpAxes;
}

// Extracts all communicated axes from the all-to-all chain.
//
// Example:
// - chain=[a2a [{"x"}: 0->1], a2a [{"y"}: 0->2]] -> returns [{"x"}, {"y"}]
SmallVector<AxisRefAttr> getCommunicatedAxes(const AllToAllChain& chain) {
  SmallVector<AxisRefAttr> communicatedAxes;
  for (AllToAllOp a2aOp : chain.a2aOps) {
    for (AllToAllParamAttr param : a2aOp.getParams()) {
      for (AxisRefAttr axis : param.getAxes()) {
        communicatedAxes.push_back(axis);
      }
    }
  }
  return communicatedAxes;
}

struct DecomposedAxesPerDim {
  SmallVector<AxisList> inAxes;
  SmallVector<AxisList> cpAxes;
};

// Decomposes sub-axes across all dimensions for both input and output shardings
// so they share a common sub-axis granularity.
//
// Example:
// - inSharding: <@mesh, [{"x":(1)4}, {}]>
// - cpOutSharding: <@mesh, [{"x":(1)2}, {}]>
// - Returns: inAxes=[[{"x":(1)2}, {"x":(2)2}], []],
//            cpAxes=[[{"x":(1)2}], []]
DecomposedAxesPerDim getDecomposedAxesPerDim(TensorShardingAttr inSharding,
                                             TensorShardingAttr cpOutSharding,
                                             MeshAttr mesh) {
  DecomposedAxesPerDim result{getAxesPerDim<AxisList>(inSharding),
                              getAxesPerDim<AxisList>(cpOutSharding)};
  alignSubAxesByDecomposition(result.inAxes, result.cpAxes, mesh);
  return result;
}

// Identifies the single tensor dimension whose sharding was modified by
// CollectivePermuteOp.
//
// Examples:
// - inAxes=[[{"x"}], [{}]], cpAxes=[[{"y"}], [{}]] -> returns 0
// - inAxes=[[{"x"}], [{"a"}]], cpAxes=[[{"y"}], [{"b"}]] ->
//   returns std::nullopt (multiple modified dimensions)
std::optional<int64_t> findSplitDimension(ArrayRef<AxisList> inAxesPerDim,
                                          ArrayRef<AxisList> cpAxesPerDim,
                                          TensorShardingAttr inSharding,
                                          TensorShardingAttr cpOutSharding) {
  int64_t rank = inSharding.getRank();
  std::optional<int64_t> splitDim;
  for (int64_t d = 0; d < rank; ++d) {
    if (inAxesPerDim[d] != cpAxesPerDim[d]) {
      if (splitDim.has_value()) {
        return std::nullopt;
      }
      splitDim = d;
    }
    if (inSharding.getDimSharding(d).getIsClosed() !=
        cpOutSharding.getDimSharding(d).getIsClosed()) {
      return std::nullopt;
    }
  }
  return splitDim;
}

struct DecomposedTypeAndSharding {
  RankedTensorType type;
  TensorShardingAttr sharding;
  int64_t numSplitDims;
};

// Constructs the decomposed RankedTensorType by expanding splitDim into
// sub-dimensions for each axis plus any residual slice.
//
// Example:
// - inputType: tensor<16x8xf32>, splitDim: 0, axisSizes: [2, 2],
//   residualSize: 4
// - Returns: tensor<2x2x4x8xf32>
RankedTensorType buildDecomposedShape(RankedTensorType inputType,
                                      int64_t splitDim,
                                      ArrayRef<int64_t> axisSizes,
                                      int64_t residualSize) {
  int64_t rank = inputType.getRank();
  SmallVector<int64_t> shape;
  shape.reserve(rank + axisSizes.size());

  // Copies leading dimensions before splitDim.
  for (int64_t d = 0; d < splitDim; ++d) {
    shape.push_back(inputType.getDimSize(d));
  }
  // Inserts decomposed sub-dimension sizes.
  shape.append(axisSizes.begin(), axisSizes.end());
  if (residualSize > 1) {
    shape.push_back(residualSize);
  }
  // Copies trailing dimensions after splitDim.
  for (int64_t d = splitDim + 1; d < rank; ++d) {
    shape.push_back(inputType.getDimSize(d));
  }
  return RankedTensorType::get(shape, inputType.getElementType());
}

// Constructs the decomposed TensorShardingAttr with 1-to-1 axis assignments
// for each decomposed sub-dimension.
//
// Example:
// - inSharding: <@mesh, [{"x", "y"}, {}]>, splitDim: 0,
//   inAxes: [{"x"}, {"y"}], hasResidual: true
// - Returns: <@mesh, [{"x"}, {"y"}, {}, {}]>
TensorShardingAttr buildDecomposedSharding(TensorShardingAttr inSharding,
                                           int64_t splitDim,
                                           ArrayRef<AxisRefAttr> inAxes,
                                           bool hasResidual) {
  MLIRContext* ctx = inSharding.getContext();
  int64_t rank = inSharding.getRank();
  SmallVector<DimensionShardingAttr> dimShardings;
  dimShardings.reserve(rank + inAxes.size());

  // Copies leading dimension shardings.
  for (int64_t d = 0; d < splitDim; ++d) {
    dimShardings.push_back(inSharding.getDimSharding(d));
  }
  // Assigns each sub-dimension its 1-to-1 axis.
  for (AxisRefAttr axis : inAxes) {
    dimShardings.push_back(
        DimensionShardingAttr::get(ctx, {axis}, /*isClosed=*/true));
  }
  if (hasResidual) {
    dimShardings.push_back(
        DimensionShardingAttr::get(ctx, {}, /*isClosed=*/true));
  }
  // Copies trailing dimension shardings.
  for (int64_t d = splitDim + 1; d < rank; ++d) {
    dimShardings.push_back(inSharding.getDimSharding(d));
  }
  return TensorShardingAttr::get(ctx, inSharding.getMeshOrRef(), dimShardings,
                                 inSharding.getReplicatedAxes(),
                                 inSharding.getUnreducedAxes());
}

// Calculates axis sizes, verifies static divisibility, and constructs the
// decomposed RankedTensorType and 1-to-1 TensorShardingAttr for the split dim.
//
// Example:
// - inputType: tensor<16x8xf32>, inSharding: <@mesh, [{"x", "y"}, {}]>,
//   splitDim: 0, inAxes: [{"x": size 2}, {"y": size 2}]
// - Returns: DecomposedTypeAndSharding{
//     type=tensor<2x2x4x8xf32>,
//     sharding=<@mesh, [{"x"}, {"y"}, {}, {}]>,
//     numSplitDims=3}
std::optional<DecomposedTypeAndSharding> computeDecomposedTypeAndSharding(
    RankedTensorType inputType, TensorShardingAttr inSharding, int64_t splitDim,
    ArrayRef<AxisRefAttr> inAxes, MeshAttr mesh) {
  // Computes axis sizes and validates static divisibility.
  SmallVector<int64_t> axisSizes;
  int64_t totalAxesSize = 1;
  for (AxisRefAttr axis : inAxes) {
    int64_t sz = axis.getSize(mesh);
    axisSizes.push_back(sz);
    totalAxesSize *= sz;
  }

  int64_t splitDimSize = inputType.getDimSize(splitDim);
  if (splitDimSize < totalAxesSize || splitDimSize % totalAxesSize != 0) {
    return std::nullopt;
  }
  int64_t residualSize = splitDimSize / totalAxesSize;
  bool hasResidual = (residualSize > 1);
  int64_t numSplitDims = inAxes.size() + (hasResidual ? 1 : 0);

  RankedTensorType splitType =
      buildDecomposedShape(inputType, splitDim, axisSizes, residualSize);
  TensorShardingAttr splitSharding =
      buildDecomposedSharding(inSharding, splitDim, inAxes, hasResidual);

  return DecomposedTypeAndSharding{splitType, splitSharding, numSplitDims};
}

// Translates each AllToAllOp in the chain to decomposed sub-dimension indices
// and builds sequential AllToAllStages.
SmallVector<AllToAllStage> computeAllToAllStages(
    const AllToAllChain& chain, int64_t splitDim, int64_t numSplitDims,
    ArrayRef<AxisRefAttr> inAxes, RankedTensorType splitInputType,
    TensorShardingAttr splitInputSharding) {
  MLIRContext* ctx = splitInputSharding.getContext();
  int64_t kNumAxes = static_cast<int64_t>(inAxes.size());
  SmallVector<AllToAllStage> a2aStages;
  TensorShardingAttr currentSharding = splitInputSharding;

  for (AllToAllOp a2aOp : chain.a2aOps) {
    SmallVector<AllToAllParamAttr> stageParamsList;
    for (AllToAllParamAttr param : a2aOp.getParams()) {
      for (AxisRefAttr axis : param.getAxes()) {
        for (int64_t k = 0; k < kNumAxes; ++k) {
          if (axis.contains(inAxes[k])) {
            int64_t tgtDim = param.getTgtDim();
            // Shifts dimensions after splitDim by the net change in rank.
            int64_t tgtSubDim =
                tgtDim < splitDim ? tgtDim : tgtDim + numSplitDims - 1;
            int64_t srcSubDim = splitDim + k;
            stageParamsList.push_back(
                AllToAllParamAttr::get(ctx, {inAxes[k]}, srcSubDim, tgtSubDim));
          }
        }
      }
    }
    if (!stageParamsList.empty()) {
      llvm::sort(stageParamsList, [](AllToAllParamAttr a, AllToAllParamAttr b) {
        return a.getSrcDim() < b.getSrcDim();
      });
      auto stageParams = AllToAllParamListAttr::get(ctx, stageParamsList);
      a2aStages.push_back(
          computeAllToAllStage(currentSharding, splitInputType, stageParams));
      currentSharding = a2aStages.back().outSharding;
    }
  }
  return a2aStages;
}

// Returns true if the leading sub-dimensions of `subDimAxes` match
// `targetAxes` in exact order.
bool matchesTargetAxes(ArrayRef<AxisRefAttr> subDimAxes,
                       ArrayRef<AxisRefAttr> targetAxes) {
  for (auto [j, targetAxis] : llvm::enumerate(targetAxes)) {
    if (subDimAxes[j] != targetAxis) {
      return false;
    }
  }
  return true;
}

// Finds the next available axis move whose destination sub-dimension is
// currently empty. Returns std::nullopt if no move can be scheduled or if a
// target axis is missing.
std::optional<AllToAllParamAttr> findAvailableAxisMove(
    ArrayRef<AxisRefAttr> subDimAxes, ArrayRef<AxisRefAttr> targetAxes,
    int64_t splitDim, MLIRContext* ctx) {
  for (auto [tgtIdx, targetAxis] : llvm::enumerate(targetAxes)) {
    // Target sub-dimension already holds the correct target axis or is
    // currently occupied; skip.
    if (subDimAxes[tgtIdx] == targetAxis || subDimAxes[tgtIdx]) {
      continue;
    }
    const auto* it = llvm::find(subDimAxes, targetAxis);
    if (it == subDimAxes.end()) {
      return std::nullopt;
    }
    int64_t srcIdx = std::distance(subDimAxes.begin(), it);
    return AllToAllParamAttr::get(ctx, {targetAxis}, splitDim + srcIdx,
                                  splitDim + tgtIdx);
  }
  return std::nullopt;
}

// Computes AllToAll stages to reorder remaining unscattered axes on splitDim
// into leading sub-dimensions [splitDim .. splitDim + m - 1] in the exact
// order specified by `targetAxes`.
//
// Example:
// - subDimAxes=[{}, {"y"}, {"z"}], targetAxes=[{"y"}, {"z"}]
//   -> Stage 1: {"y"}: 1->0
//   -> Stage 2: {"z"}: 2->1
//   Resulting in leading sub-dimensions matching targetAxes.
std::optional<SmallVector<AllToAllStage>> computeStagesForRemainingAxes(
    TensorShardingAttr currentSharding, RankedTensorType splitType,
    int64_t splitDim, int64_t numSplitDims, ArrayRef<AxisRefAttr> targetAxes) {
  int64_t m = static_cast<int64_t>(targetAxes.size());
  if (m > numSplitDims) {
    return std::nullopt;
  }
  if (targetAxes.empty()) {
    return SmallVector<AllToAllStage>{};
  }

  MLIRContext* ctx = currentSharding.getContext();

  // Each decomposed sub-dimension holds at most one axis (or null if empty).
  SmallVector<AxisRefAttr> subDimAxes(numSplitDims, AxisRefAttr{});
  for (int64_t i = 0; i < numSplitDims; ++i) {
    ArrayRef<AxisRefAttr> axes =
        currentSharding.getDimSharding(splitDim + i).getAxes();
    if (!axes.empty()) {
      subDimAxes[i] = axes.front();
    }
  }

  SmallVector<AllToAllStage> stages;

  // Move remaining axes into leading sub-dimensions until they match
  // targetAxes.
  while (!matchesTargetAxes(subDimAxes, targetAxes)) {
    std::optional<AllToAllParamAttr> param =
        findAvailableAxisMove(subDimAxes, targetAxes, splitDim, ctx);
    if (!param) {
      // Aborts the pattern rewrite and preserves collective_permute if
      // remaining axes cannot be placed due to cyclic dependencies.
      return std::nullopt;
    }

    auto stageParams = AllToAllParamListAttr::get(ctx, {*param});
    stages.push_back(
        computeAllToAllStage(currentSharding, splitType, stageParams));
    currentSharding = stages.back().outSharding;

    int64_t srcIdx = param->getSrcDim() - splitDim;
    int64_t tgtIdx = param->getTgtDim() - splitDim;
    subDimAxes[tgtIdx] = param->getAxes().front();
    subDimAxes[srcIdx] = AxisRefAttr{};
  }

  return stages;
}

}  // namespace

std::optional<AllToAllChain> extractAllToAllChain(CollectivePermuteOp cpOp,
                                                  int64_t splitDim) {
  if (!cpOp.getResult().hasOneUse()) {
    return std::nullopt;
  }

  SmallVector<AllToAllOp> a2as;
  Operation* curr = *cpOp.getResult().user_begin();
  while (auto a2a = dyn_cast<AllToAllOp>(curr)) {
    // Stop if any parameter in this AllToAllOp sources from a dimension
    // other than splitDim.
    bool sourcesFromSplitDim = llvm::all_of(
        a2a.getParams(),
        [&](AllToAllParamAttr param) { return param.getSrcDim() == splitDim; });
    if (!sourcesFromSplitDim) {
      break;
    }

    a2as.push_back(a2a);
    if (!a2a.getResult().hasOneUse()) {
      break;
    }
    curr = *a2a.getResult().user_begin();
  }

  if (a2as.empty()) {
    return std::nullopt;
  }

  return AllToAllChain{cpOp, std::move(a2as)};
}

std::optional<int64_t> getSplitDimension(CollectivePermuteOp cpOp,
                                         const SymbolTable& symbolTable) {
  if (!isSupportedTensorAndMesh(cpOp, symbolTable)) {
    return std::nullopt;
  }

  TensorShardingAttr inSharding = getSharding(cpOp.getTensor());
  TensorShardingAttr cpOutSharding = cpOp.getOutSharding();
  MeshAttr mesh = inSharding.getMesh(symbolTable);

  auto [inAxesPerDim, cpAxesPerDim] =
      getDecomposedAxesPerDim(inSharding, cpOutSharding, mesh);

  std::optional<int64_t> splitDim =
      findSplitDimension(inAxesPerDim, cpAxesPerDim, inSharding, cpOutSharding);
  if (!splitDim ||
      !isPurePermutation(inAxesPerDim[*splitDim], cpAxesPerDim[*splitDim])) {
    return std::nullopt;
  }

  return splitDim;
}

bool isChainOptimizable(const AllToAllChain& chain, int64_t splitDim,
                        const SymbolTable& symbolTable) {
  CollectivePermuteOp cpOp = chain.cpOp;
  TensorShardingAttr inSharding = getSharding(cpOp.getTensor());
  TensorShardingAttr cpOutSharding = cpOp.getOutSharding();
  MeshAttr mesh = inSharding.getMesh(symbolTable);

  auto [inAxesPerDim, cpAxesPerDim] =
      getDecomposedAxesPerDim(inSharding, cpOutSharding, mesh);

  SmallVector<AxisRefAttr> inAxes = llvm::to_vector(inAxesPerDim[splitDim]);
  SmallVector<AxisRefAttr> cpAxes = llvm::to_vector(cpAxesPerDim[splitDim]);
  SmallVector<AxisRefAttr> communicatedAxes = getCommunicatedAxes(chain);
  return communicatesAnyPermutedAxis(inAxes, cpAxes, communicatedAxes);
}

std::optional<AllToAllRewritePlan> computeRewritePlan(
    const AllToAllChain& chain, int64_t splitDim,
    const SymbolTable& symbolTable) {
  CollectivePermuteOp cpOp = chain.cpOp;
  AllToAllOp terminalOp = chain.a2aOps.back();
  Value inputTensor = cpOp.getTensor();
  auto inputType = cast<RankedTensorType>(inputTensor.getType());
  TensorShardingAttr inSharding = getSharding(inputTensor);
  MeshAttr mesh = inSharding.getMesh(symbolTable);

  auto [inAxesPerDim, cpAxesPerDim] =
      getDecomposedAxesPerDim(inSharding, cpOp.getOutSharding(), mesh);
  SmallVector<AxisRefAttr> inAxes(inAxesPerDim[splitDim].begin(),
                                  inAxesPerDim[splitDim].end());

  // Decomposes split dimension shape and sharding.
  std::optional<DecomposedTypeAndSharding> decomposed =
      computeDecomposedTypeAndSharding(inputType, inSharding, splitDim, inAxes,
                                       mesh);
  if (!decomposed) {
    return std::nullopt;
  }

  // Generates all-to-all stages over the decomposed tensor.
  SmallVector<AllToAllStage> a2aStages =
      computeAllToAllStages(chain, splitDim, decomposed->numSplitDims, inAxes,
                            decomposed->type, decomposed->sharding);
  if (a2aStages.empty()) {
    return std::nullopt;
  }

  auto outputType = cast<RankedTensorType>(terminalOp.getType());
  TensorShardingAttr outputSharding = terminalOp.getOutSharding();
  ArrayRef<AxisRefAttr> targetAxes =
      outputSharding.getDimSharding(splitDim).getAxes();
  std::optional<SmallVector<AllToAllStage>> remainingStages =
      computeStagesForRemainingAxes(a2aStages.back().outSharding,
                                    decomposed->type, splitDim,
                                    decomposed->numSplitDims, targetAxes);
  if (!remainingStages) {
    return std::nullopt;
  }
  a2aStages.append(remainingStages->begin(), remainingStages->end());

  // Validates sharding equivalence across the final reshape.
  if (!isShardingEquivalentAcrossReshapes(
          a2aStages.back().outSharding, a2aStages.back().outType,
          outputSharding, outputType, terminalOp,
          /*allowNonDivisible=*/false)) {
    return std::nullopt;
  }

  return AllToAllRewritePlan{decomposed->type, decomposed->sharding,
                             std::move(a2aStages), outputType, outputSharding};
}

LogicalResult rewriteAllToAllChain(const AllToAllChain& chain,
                                   const AllToAllRewritePlan& plan,
                                   PatternRewriter& rewriter) {
  CollectivePermuteOp cpOp = chain.cpOp;
  AllToAllOp terminalOp = chain.a2aOps.back();
  Location loc = cpOp.getLoc();
  Value inputTensor = cpOp.getTensor();

  // Reshapes input to decomposed sub-dimension shape.
  Value reshapedInput = stablehlo::ReshapeOp::create(
      rewriter, loc, plan.splitInputType, inputTensor);
  setSharding(reshapedInput, plan.splitInputSharding);

  // Emits all-to-all stages over decomposed sub-dimensions.
  Value a2aResult = reshapedInput;
  for (const AllToAllStage& stage : plan.a2aStages) {
    a2aResult = AllToAllOp::create(rewriter, loc, stage.outType, a2aResult,
                                   stage.params, stage.outSharding)
                    .getResult();
  }

  // Reshapes back to original output tensor shape.
  Value reshapedOutput =
      stablehlo::ReshapeOp::create(rewriter, loc, plan.outputType, a2aResult);
  setSharding(reshapedOutput, plan.outputSharding);

  // Replaces terminal op with rewritten output and erases obsolete chain ops.
  rewriter.replaceOp(terminalOp, reshapedOutput);
  for (AllToAllOp a2aOp : llvm::reverse(llvm::drop_end(chain.a2aOps))) {
    rewriter.eraseOp(a2aOp);
  }
  rewriter.eraseOp(cpOp);

  return success();
}

}  // namespace sdy
}  // namespace mlir
