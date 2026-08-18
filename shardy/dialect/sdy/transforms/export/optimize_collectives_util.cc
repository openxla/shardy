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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
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

// Returns true if all axes permuted between `inAxes` and `cpAxes` are
// communicated by `commAxes`.
bool arePermutedAxesCommunicated(ArrayRef<AxisRefAttr> inAxes,
                                 ArrayRef<AxisRefAttr> cpAxes,
                                 ArrayRef<AxisRefAttr> commAxes) {
  SmallVector<AxisRefAttr> permutedAxes;
  for (size_t i = 0; i < inAxes.size(); ++i) {
    if (inAxes[i] != cpAxes[i]) {
      permutedAxes.push_back(inAxes[i]);
    }
  }

  for (AxisRefAttr pAxis : permutedAxes) {
    bool communicated = llvm::any_of(
        commAxes, [&](AxisRefAttr cAxis) { return cAxis.contains(pAxis); });
    if (!communicated) {
      return false;
    }
  }
  return true;
}

// Computes an AllToAllStage given an input sharding, input type, and all-to-all
// parameters.
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

}  // namespace

std::optional<A2AChain> extractA2AChain(AllToAllOp terminalOp) {
  // Ensures terminalOp is the end of the chain by verifying no downstream
  // AllToAllOp users exist.
  for (Operation* user : terminalOp.getResult().getUsers()) {
    if (isa<AllToAllOp>(user)) {
      return std::nullopt;
    }
  }

  SmallVector<AllToAllOp> a2as;
  a2as.push_back(terminalOp);
  Operation* curr = terminalOp.getTensor().getDefiningOp();
  // Traverses upward through the chain of single-use AllToAllOp producers.
  while (auto a2a = dyn_cast_or_null<AllToAllOp>(curr)) {
    if (!a2a.getResult().hasOneUse()) {
      return std::nullopt;
    }
    a2as.push_back(a2a);
    curr = a2a.getTensor().getDefiningOp();
  }

  // Verifies the root producer is a single-use CollectivePermuteOp.
  auto cpOp = dyn_cast_or_null<CollectivePermuteOp>(curr);
  if (!cpOp || !cpOp.getResult().hasOneUse()) {
    return std::nullopt;
  }

  // Reverses to establish root-to-terminal order [a2a_0, a2a_1, ..., a2a_k].
  std::reverse(a2as.begin(), a2as.end());
  return A2AChain{cpOp, std::move(a2as)};
}

std::optional<int64_t> getOptimizableSplitDim(const A2AChain& chain) {
  CollectivePermuteOp cpOp = chain.cpOp;
  AllToAllOp terminalOp = chain.a2aOps.back();
  Value inputTensor = cpOp.getTensor();

  // Invariant 1: Requires static shape and non-zero rank on input tensor.
  auto inputType = dyn_cast<RankedTensorType>(inputTensor.getType());
  if (!inputType || !inputType.hasStaticShape()) {
    return std::nullopt;
  }
  int64_t rank = inputType.getRank();
  if (rank == 0) {
    return std::nullopt;
  }

  // Invariant 2: Requires static shape and matching rank on terminal output.
  auto outputType = dyn_cast<RankedTensorType>(terminalOp.getType());
  if (!outputType || !outputType.hasStaticShape() ||
      outputType.getRank() != rank) {
    return std::nullopt;
  }

  // Invariant 3: Requires valid shardings with matching rank.
  TensorShardingAttr inSharding = getSharding(inputTensor);
  TensorShardingAttr cpOutSharding = cpOp.getOutSharding();
  if (!inSharding || !cpOutSharding) {
    return std::nullopt;
  }
  if (inSharding.getRank() != rank || cpOutSharding.getRank() != rank) {
    return std::nullopt;
  }

  // Invariant 3 (continued): Validates mesh identity and device IDs.
  SymbolTable symbolTable(cpOp->getParentOfType<ModuleOp>());
  MeshAttr inMesh = inSharding.getMesh(symbolTable);
  MeshAttr cpOutMesh = cpOutSharding.getMesh(symbolTable);
  if (!inMesh || !cpOutMesh || inMesh != cpOutMesh) {
    return std::nullopt;
  }
  if (inSharding.getMeshOrRef() != cpOutSharding.getMeshOrRef()) {
    return std::nullopt;
  }
  if (inMesh.getDeviceIds() != cpOutMesh.getDeviceIds()) {
    return std::nullopt;
  }

  // Invariant 3 (continued): Validates mesh, replicated/unreduced axes, and
  // non-empty params across all AllToAll operations in the chain.
  for (AllToAllOp a2aOp : chain.a2aOps) {
    TensorShardingAttr a2aOut = a2aOp.getOutSharding();
    if (!a2aOut || a2aOut.getMesh(symbolTable) != inMesh ||
        a2aOut.getMeshOrRef() != inSharding.getMeshOrRef() ||
        a2aOut.getRank() != rank) {
      return std::nullopt;
    }
    if (a2aOut.getReplicatedAxes() != inSharding.getReplicatedAxes() ||
        a2aOut.getUnreducedAxes() != inSharding.getUnreducedAxes()) {
      return std::nullopt;
    }
    if (a2aOp.getParams().empty()) {
      return std::nullopt;
    }
  }

  if (inSharding.getReplicatedAxes() != cpOutSharding.getReplicatedAxes() ||
      inSharding.getUnreducedAxes() != cpOutSharding.getUnreducedAxes()) {
    return std::nullopt;
  }

  // Invariant 4: Aligns sub-axes by decomposition and ensures exactly one
  // dimension is modified by the CollectivePermuteOp.
  SmallVector<AxisList> inAxesPerDim = getAxesPerDim<AxisList>(inSharding);
  SmallVector<AxisList> cpAxesPerDim = getAxesPerDim<AxisList>(cpOutSharding);
  alignSubAxesByDecomposition(inAxesPerDim, cpAxesPerDim, inMesh);

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

  // Invariant 5: Ensures all AllToAllOps source from the single splitDim.
  for (AllToAllOp a2aOp : chain.a2aOps) {
    for (AllToAllParamAttr param : a2aOp.getParams()) {
      if (param.getAxes().empty()) {
        return std::nullopt;
      }
      int64_t srcDim = param.getSrcDim();
      if (srcDim < 0 || srcDim >= rank) {
        return std::nullopt;
      }
      if (splitDim.has_value()) {
        if (srcDim != *splitDim) {
          return std::nullopt;
        }
      } else {
        splitDim = srcDim;
      }
    }
  }

  if (!splitDim.has_value()) {
    return std::nullopt;
  }

  // Invariant 6: Requires at least 2 axes on splitDim and ensures conservation
  // of axes across CollectivePermuteOp.
  SmallVector<AxisRefAttr> inAxes(inAxesPerDim[*splitDim].begin(),
                                  inAxesPerDim[*splitDim].end());
  SmallVector<AxisRefAttr> cpAxes(cpAxesPerDim[*splitDim].begin(),
                                  cpAxesPerDim[*splitDim].end());
  if (inAxes.size() < 2 || cpAxes.size() < 2) {
    return std::nullopt;
  }

  if (inAxes.size() != cpAxes.size()) {
    return std::nullopt;
  }

  SmallVector<AxisRefAttr> sortedInAxes = inAxes;
  SmallVector<AxisRefAttr> sortedCpAxes = cpAxes;
  llvm::sort(sortedInAxes);
  llvm::sort(sortedCpAxes);
  if (sortedInAxes != sortedCpAxes) {
    return std::nullopt;
  }

  // Invariant 7: Verifies all permuted axes are communicated by AllToAllOps.
  SmallVector<AxisRefAttr> communicatedAxes;
  for (AllToAllOp a2aOp : chain.a2aOps) {
    for (AllToAllParamAttr param : a2aOp.getParams()) {
      for (AxisRefAttr axis : param.getAxes()) {
        communicatedAxes.push_back(axis);
      }
    }
  }

  if (!arePermutedAxesCommunicated(inAxes, cpAxes, communicatedAxes)) {
    return std::nullopt;
  }

  // Invariant 8: Ensures all target dimensions are disjoint and distinct from
  // the split dimension.
  llvm::DenseSet<int64_t> targetDims;
  for (AllToAllOp a2aOp : chain.a2aOps) {
    for (AllToAllParamAttr param : a2aOp.getParams()) {
      int64_t tgtDim = param.getTgtDim();
      if (tgtDim == *splitDim || tgtDim < 0 || tgtDim >= rank) {
        return std::nullopt;
      }
      if (!targetDims.insert(tgtDim).second) {
        return std::nullopt;
      }
    }
  }

  return splitDim;
}

std::optional<SplitConfig> computeSplitConfig(const A2AChain& chain,
                                              int64_t splitDim) {
  CollectivePermuteOp cpOp = chain.cpOp;
  AllToAllOp terminalOp = chain.a2aOps.back();
  Value inputTensor = cpOp.getTensor();
  auto inputType = cast<RankedTensorType>(inputTensor.getType());
  TensorShardingAttr inSharding = getSharding(inputTensor);
  MLIRContext* ctx = cpOp.getContext();
  SymbolTable symbolTable(cpOp->getParentOfType<ModuleOp>());
  MeshAttr mesh = inSharding.getMesh(symbolTable);
  int64_t rank = inputType.getRank();

  // Decomposes sub-axes and extracts axes bound to the split dimension.
  SmallVector<AxisList> inAxesPerDim = getAxesPerDim<AxisList>(inSharding);
  SmallVector<AxisList> cpAxesPerDim =
      getAxesPerDim<AxisList>(cpOp.getOutSharding());
  alignSubAxesByDecomposition(inAxesPerDim, cpAxesPerDim, mesh);
  SmallVector<AxisRefAttr> inAxes(inAxesPerDim[splitDim].begin(),
                                  inAxesPerDim[splitDim].end());
  int64_t kNumAxes = static_cast<int64_t>(inAxes.size());

  // Calculates axis sizes and validates static divisibility.
  SmallVector<int64_t> axisSizes;
  int64_t totalAxesSize = 1;
  for (AxisRefAttr axis : inAxes) {
    int64_t sz = axis.getSize(mesh);
    axisSizes.push_back(sz);
    totalAxesSize *= sz;
  }

  int64_t splitDimSize = inputType.getDimSize(splitDim);
  if (splitDimSize % totalAxesSize != 0) {
    return std::nullopt;
  }
  int64_t residualSize = splitDimSize / totalAxesSize;
  bool hasResidual = (residualSize > 1);
  int64_t numSplitDims = kNumAxes + (hasResidual ? 1 : 0);

  // Constructs the decomposed split input shape.
  SmallVector<int64_t> splitInputShape;
  for (int64_t d = 0; d < splitDim; ++d) {
    splitInputShape.push_back(inputType.getDimSize(d));
  }
  for (int64_t k = 0; k < kNumAxes; ++k) {
    splitInputShape.push_back(axisSizes[k]);
  }
  if (hasResidual) {
    splitInputShape.push_back(residualSize);
  }
  for (int64_t d = splitDim + 1; d < rank; ++d) {
    splitInputShape.push_back(inputType.getDimSize(d));
  }
  auto splitInputType =
      RankedTensorType::get(splitInputShape, inputType.getElementType());

  // Constructs 1-to-1 shardings for decomposed sub-dimensions.
  SmallVector<DimensionShardingAttr> splitInDimShardings;
  for (int64_t d = 0; d < splitDim; ++d) {
    splitInDimShardings.push_back(inSharding.getDimSharding(d));
  }
  for (int64_t k = 0; k < kNumAxes; ++k) {
    splitInDimShardings.push_back(
        DimensionShardingAttr::get(ctx, {inAxes[k]}, /*isClosed=*/true));
  }
  if (hasResidual) {
    splitInDimShardings.push_back(
        DimensionShardingAttr::get(ctx, {}, /*isClosed=*/true));
  }
  for (int64_t d = splitDim + 1; d < rank; ++d) {
    splitInDimShardings.push_back(inSharding.getDimSharding(d));
  }
  auto splitInputSharding = TensorShardingAttr::get(
      ctx, inSharding.getMeshOrRef(), splitInDimShardings,
      inSharding.getReplicatedAxes(), inSharding.getUnreducedAxes());

  // Maps each AllToAll axis parameter to its decomposed sub-dimension index.
  SmallVector<AllToAllParamAttr> combinedParamsList;
  for (int64_t k = 0; k < kNumAxes; ++k) {
    std::optional<int64_t> origTgtDim;
    for (AllToAllOp a2aOp : chain.a2aOps) {
      for (AllToAllParamAttr param : a2aOp.getParams()) {
        for (AxisRefAttr axis : param.getAxes()) {
          if (axis.contains(inAxes[k])) {
            origTgtDim = param.getTgtDim();
            break;
          }
        }
        if (origTgtDim.has_value()) {
          break;
        }
      }
      if (origTgtDim.has_value()) {
        break;
      }
    }
    if (origTgtDim.has_value()) {
      int64_t d_tgt = *origTgtDim;
      int64_t tgtSubDim = d_tgt < splitDim ? d_tgt : d_tgt + numSplitDims - 1;
      int64_t srcSubDim = splitDim + k;
      combinedParamsList.push_back(
          AllToAllParamAttr::get(ctx, {inAxes[k]}, srcSubDim, tgtSubDim));
    }
  }

  if (combinedParamsList.empty()) {
    return std::nullopt;
  }

  // Ensures target sub-dimensions are disjoint.
  llvm::DenseSet<int64_t> usedTgtSubDims;
  for (AllToAllParamAttr param : combinedParamsList) {
    if (!usedTgtSubDims.insert(param.getTgtDim()).second) {
      return std::nullopt;
    }
  }

  llvm::sort(combinedParamsList, [](AllToAllParamAttr a, AllToAllParamAttr b) {
    return a.getSrcDim() < b.getSrcDim();
  });
  auto combinedParams = AllToAllParamListAttr::get(ctx, combinedParamsList);

  SmallVector<AllToAllStage> a2aStages;
  a2aStages.push_back(
      computeAllToAllStage(splitInputSharding, splitInputType, combinedParams));

  auto outputType = cast<RankedTensorType>(terminalOp.getType());
  TensorShardingAttr outputSharding = terminalOp.getOutSharding();

  // Validates sharding equivalence across the final reshape.
  if (!isShardingEquivalentAcrossReshapes(
          a2aStages.back().outSharding, a2aStages.back().outType,
          outputSharding, outputType, terminalOp,
          /*allowNonDivisible=*/false)) {
    return std::nullopt;
  }

  return SplitConfig{splitInputType, splitInputSharding, std::move(a2aStages),
                     outputType, outputSharding};
}

LogicalResult rewriteA2AChain(const A2AChain& chain, const SplitConfig& config,
                              PatternRewriter& rewriter) {
  CollectivePermuteOp cpOp = chain.cpOp;
  AllToAllOp terminalOp = chain.a2aOps.back();
  Location loc = cpOp.getLoc();
  Value inputTensor = cpOp.getTensor();

  // Reshapes input to decomposed sub-dimension shape.
  Value reshapedInput = stablehlo::ReshapeOp::create(
      rewriter, loc, config.splitInputType, inputTensor);
  setSharding(reshapedInput, config.splitInputSharding);

  // Emits all-to-all stages over decomposed sub-dimensions.
  Value a2aResult = reshapedInput;
  for (const AllToAllStage& stage : config.a2aStages) {
    a2aResult = AllToAllOp::create(rewriter, loc, stage.outType, a2aResult,
                                   stage.params, stage.outSharding)
                    .getResult();
  }

  // Reshapes back to original output tensor shape.
  Value reshapedOutput =
      stablehlo::ReshapeOp::create(rewriter, loc, config.outputType, a2aResult);
  setSharding(reshapedOutput, config.outputSharding);

  // Replaces terminal op with rewritten output and erases obsolete chain ops.
  rewriter.replaceOp(terminalOp, reshapedOutput);
  for (int i = static_cast<int>(chain.a2aOps.size()) - 2; i >= 0; --i) {
    rewriter.eraseOp(chain.a2aOps[i]);
  }
  rewriter.eraseOp(cpOp);

  return success();
}

}  // namespace sdy
}  // namespace mlir
