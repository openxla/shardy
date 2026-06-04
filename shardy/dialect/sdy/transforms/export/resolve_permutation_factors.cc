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

#include <cstdint>
#include <optional>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/explicit_reshards_util.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_SHARDYRESOLVEPERMUTATIONFACTORSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

using MeshCache = DenseMap<MeshAttr, MeshOp>;

// Decomposes a linear logical index into mesh coordinates.
SmallVector<int64_t> getMeshCoordinates(int64_t index,
                                        ArrayRef<MeshAxisAttr> axes) {
  SmallVector<int64_t> coords(axes.size());
  for (int64_t j = static_cast<int64_t>(axes.size()) - 1; j >= 0; --j) {
    coords[j] = index % axes[j].getSize();
    index /= axes[j].getSize();
  }
  return coords;
}

// Recomposes a linear logical index from mesh coordinates.
int64_t getLinearIndexFromCoordinates(ArrayRef<int64_t> coords,
                                      ArrayRef<MeshAxisAttr> axes) {
  int64_t index = 0;
  int64_t multiplier = 1;
  for (int64_t j = static_cast<int64_t>(axes.size()) - 1; j >= 0; --j) {
    index += coords[j] * multiplier;
    multiplier *= axes[j].getSize();
  }
  return index;
}

// Returns the index of the mesh axis with the given name.
int64_t getMeshAxisIndex(ArrayRef<MeshAxisAttr> axes, StringRef name) {
  for (auto [index, axis] : llvm::enumerate(axes)) {
    if (axis.getName() == name) {
      return static_cast<int64_t>(index);
    }
  }
  return -1;
}

// Returns a new MeshAttr where the ordering of devices along the specified
// 'axesToReverse' is reversed.
MeshAttr getMeshWithReversedAxes(MeshAttr mesh,
                                 ArrayRef<AxisRefAttr> axesToReverse) {
  int64_t totalSize = mesh.getTotalSize();
  SmallVector<int64_t> originalDeviceIds;
  if (mesh.getDeviceIds().empty()) {
    originalDeviceIds.reserve(totalSize);
    for (int64_t i = 0; i < totalSize; ++i) {
      originalDeviceIds.push_back(i);
    }
  } else {
    originalDeviceIds.assign(mesh.getDeviceIds().begin(),
                             mesh.getDeviceIds().end());
  }

  SmallVector<int64_t> newDeviceIds(totalSize);
  ArrayRef<MeshAxisAttr> meshAxes = mesh.getAxes();
  for (int64_t i = 0; i < totalSize; ++i) {
    SmallVector<int64_t> coords = getMeshCoordinates(i, meshAxes);
    SmallVector<int64_t> newCoords = coords;
    for (AxisRefAttr axisRef : axesToReverse) {
      int64_t meshIdx = getMeshAxisIndex(meshAxes, axisRef.getName());
      int64_t fullSize = meshAxes[meshIdx].getSize();
      int64_t subSize = axisRef.getSize(mesh);
      int64_t preSize = axisRef.getSubAxisPreSize();

      // Calculate the physical stride (distance between logical shards) for
      // this sub-axis.
      int64_t postSize = fullSize / (preSize * subSize);
      // Identify the relative coordinate of this device within the specific
      // sub-axis factor.
      int64_t subCoord = (coords[meshIdx] / postSize) % subSize;

      // Calculate the new coordinate by flipping the sub-axis component.
      int64_t newSubCoord = subSize - 1 - subCoord;
      // Update the new coordinate by adding the difference between the new
      // and old sub-axis coordinates, scaled by the stride.
      newCoords[meshIdx] += (newSubCoord - subCoord) * postSize;
    }

    newDeviceIds[i] =
        originalDeviceIds[getLinearIndexFromCoordinates(newCoords, meshAxes)];
  }
  return MeshAttr::get(mesh.getContext(), mesh.getAxes(), newDeviceIds);
}

MeshOp getOrCreateMesh(Location loc, ModuleOp module, MeshAttr meshAttr,
                       StringRef baseName, SymbolTable& symbolTable,
                       MeshCache& meshCache) {
  auto emplaceResult = meshCache.try_emplace(meshAttr, nullptr);
  if (emplaceResult.second) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    MeshOp meshOp = MeshOp::create(moduleBuilder, loc, baseName, meshAttr);
    // Insert the op and rename if needed to avoid name collisions.
    symbolTable.insert(meshOp, module.getBody()->begin());
    emplaceResult.first->second = meshOp;
  }
  return emplaceResult.first->second;
}

// Returns a TensorShardingAttr where all axes in 'manualAxes' are removed.
TensorShardingAttr getLocalSharding(
    TensorShardingAttr sharding,
    const llvm::SmallDenseSet<StringRef>& manualAxes) {
  SmallVector<DimensionShardingAttr> newDimShardings;
  for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
    SmallVector<AxisRefAttr> newAxes;
    for (AxisRefAttr axis : dimSharding.getAxes()) {
      if (!manualAxes.contains(axis.getName())) {
        newAxes.push_back(axis);
      }
    }
    newDimShardings.push_back(DimensionShardingAttr::get(
        sharding.getContext(), newAxes, dimSharding.getIsClosed(),
        dimSharding.getPriority()));
  }
  return TensorShardingAttr::get(sharding.getContext(), sharding.getMeshOrRef(),
                                 newDimShardings, sharding.getReplicatedAxes(),
                                 sharding.getUnreducedAxes());
}

// For each logical device i in the 'manualMesh', computes the logical device
// j that should receive the right-shift data of i when the dimension is
// partitioned by the manual axes in 'manualAxesInDim'. Returns all the (i, j)
// pairs where i is the source device and j is the target device.
SmallVector<int64_t> getRightShiftSourceTargetPairs(
    MeshAttr manualMesh, ArrayRef<AxisRefAttr> manualAxesInDim) {
  ArrayRef<MeshAxisAttr> manualMeshAxes = manualMesh.getAxes();
  int64_t totalSize = manualMesh.getTotalSize();
  SmallVector<int64_t> pairs;
  for (int64_t i = 0; i < totalSize; ++i) {
    SmallVector<int64_t> coords = getMeshCoordinates(i, manualMeshAxes);

    // Compute the (coords + 1) in the hierarchy defined by 'manualAxesInDim'.
    SmallVector<int64_t> nextCoords = coords;
    // Set `carried` to true to add 1 to the most minor axis.
    bool carried = true;
    for (int64_t k = (int64_t)manualAxesInDim.size() - 1; k >= 0 && carried;
         --k) {
      auto axisRef = manualAxesInDim[k];
      int64_t meshIdx = getMeshAxisIndex(manualMeshAxes, axisRef.getName());

      int64_t fullSize = manualMeshAxes[meshIdx].getSize();
      int64_t subSize = axisRef.getSize(manualMesh);
      int64_t preSize = axisRef.getSubAxisPreSize();

      int64_t postSize = fullSize / (preSize * subSize);
      int64_t subCoord = (coords[meshIdx] / postSize) % subSize;

      if (subCoord + 1 < subSize) {
        nextCoords[meshIdx] += postSize;
        carried = false;
      } else {
        nextCoords[meshIdx] -= (subSize - 1) * postSize;
      }
    }

    if (!carried) {
      pairs.push_back(i);
      pairs.push_back(
          getLinearIndexFromCoordinates(nextCoords, manualMeshAxes));
    }
  }
  return pairs;
}

// Generates code to shift the trailing N elements in each local device to the
// right in the partitioning space using HALO exchange.
Value shiftDataRight(Location loc, Value input, int64_t dim, int64_t trailingN,
                     TensorShardingAttr sharding, MeshAttr manualMesh,
                     const llvm::SmallDenseSet<StringRef>& manualAxes,
                     IRRewriter& rewriter) {
  auto localType = cast<RankedTensorType>(input.getType());
  int64_t shardSize = localType.getDimSize(dim);
  int64_t rank = localType.getRank();
  TensorShardingAttr localSharding = getLocalSharding(sharding, manualAxes);

  // Identify the value for HALO exchange.
  SDY_CHECK(trailingN <= shardSize) << "Padding exceeds single shard size";
  int64_t leadingN = shardSize - trailingN;
  Value halo = input;
  if (leadingN > 0) {
    SmallVector<int64_t> haloStarts(rank, 0);
    haloStarts[dim] = leadingN;
    SmallVector<int64_t> haloShape = llvm::to_vector(localType.getShape());
    haloShape[dim] = trailingN;
    auto sliceOp = stablehlo::SliceOp::create(
        rewriter, loc,
        RankedTensorType::get(haloShape, localType.getElementType()), input,
        rewriter.getDenseI64ArrayAttr(haloStarts),
        rewriter.getDenseI64ArrayAttr(localType.getShape()),
        rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(rank, 1)));
    setSharding(sliceOp.getResult(), localSharding);
    halo = sliceOp.getResult();
  }

  // Compute the right-shift source and target device pairs.
  SmallVector<AxisRefAttr> manualAxesInDim;
  for (auto axisRef : sharding.getDimShardings()[dim].getAxes()) {
    if (manualAxes.contains(axisRef.getName())) {
      manualAxesInDim.push_back(axisRef);
    }
  }
  SmallVector<int64_t> pairs =
      getRightShiftSourceTargetPairs(manualMesh, manualAxesInDim);

  // Perform collective permute to shift HALO value to the right.
  auto receivedHalo = stablehlo::CollectivePermuteOp::create(
      rewriter, loc, halo.getType(), halo,
      DenseIntElementsAttr::get(
          RankedTensorType::get({(int64_t)pairs.size() / 2, 2},
                                rewriter.getI64Type()),
          pairs),
      nullptr);
  setSharding(receivedHalo.getResult(), localSharding);

  if (leadingN == 0) {
    return receivedHalo.getResult();
  }

  // Concatenate [receivedHalo, leadingData].
  SmallVector<int64_t> dataLimits = llvm::to_vector(localType.getShape());
  dataLimits[dim] = leadingN;
  auto leadingData = stablehlo::SliceOp::create(
      rewriter, loc,
      RankedTensorType::get(dataLimits, localType.getElementType()), input,
      rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(rank, 0)),
      rewriter.getDenseI64ArrayAttr(dataLimits),
      rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(rank, 1)));
  setSharding(leadingData.getResult(), localSharding);

  auto concatOp = stablehlo::ConcatenateOp::create(
      rewriter, loc, localType,
      ValueRange{receivedHalo.getResult(), leadingData.getResult()},
      static_cast<uint64_t>(dim));
  setSharding(concatOp.getResult(), localSharding);
  return concatOp.getResult();
}

// Generates the local device code wrapped inside a manual computation to
// implement HALO exchange for shifting the padding in the operand of a
// reverse op from high edge to low edge.
Value shiftPaddingToLow(Location loc, Value input, RankedTensorType origType,
                        TensorShardingAttr sharding, MeshAttr mesh,
                        const llvm::SmallDenseSet<StringRef>& manualAxes,
                        ArrayRef<int64_t> paddedShape,
                        ArrayRef<int64_t> dimsToShift,
                        ArrayRef<int64_t> shiftAmounts, IRRewriter& rewriter) {
  // Compute the local tensor shape for the operand inside manual computation.
  SmallVector<int64_t> localShape;
  auto paddedType = cast<RankedTensorType>(input.getType());
  for (auto [dim, dimSharding] : llvm::enumerate(sharding.getDimShardings())) {
    int64_t manualFactor = 1;
    for (auto axis : dimSharding.getAxes()) {
      if (manualAxes.contains(axis.getName())) {
        manualFactor *= axis.getSize(mesh);
      }
    }
    localShape.push_back(paddedShape[dim] / manualFactor);
  }

  // Get the manual axes and mesh axes for the manual computation.
  SmallVector<StringAttr> manualAxesAttrs;
  SmallVector<MeshAxisAttr> manualMeshAxes;
  for (auto axis : mesh.getAxes()) {
    if (manualAxes.contains(axis.getName())) {
      manualAxesAttrs.push_back(rewriter.getStringAttr(axis.getName()));
      manualMeshAxes.push_back(axis);
    }
  }
  MeshAttr manualMesh = MeshAttr::get(rewriter.getContext(), manualMeshAxes);

  // Build the manual computation.
  auto manualComp =
      ManualComputationOp::create(rewriter, loc, input.getType(), input,
                                  {sharding}, {sharding}, manualAxesAttrs);
  Region& body = manualComp.getBody();
  body.emplaceBlock();
  Value shiftedLocal = body.addArgument(
      RankedTensorType::get(localShape, paddedType.getElementType()), loc);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&body.front());

  // Implement the HALO exchange to shift the padding for each padded
  // dimension.
  for (auto [k, dim] : llvm::enumerate(dimsToShift)) {
    shiftedLocal = shiftDataRight(loc, shiftedLocal, dim, shiftAmounts[k],
                                  sharding, manualMesh, manualAxes, rewriter);
  }

  ReturnOp::create(rewriter, loc, shiftedLocal);
  return manualComp.getResult(0);
}

// Information for implementing a reverse operation.
struct ReverseOpInfo {
  // Axes used to shard dimensions being reversed, which are also the axes
  // whose device ordering needs to be reverses.
  llvm::SmallVector<AxisRefAttr> axesToReverse;
  // Axes used to shard dimensions being reversed and also involve in
  // indivisible dimensions.
  llvm::SmallDenseSet<StringRef> manualAxes;
  // The new shape of the input tensor after all indivisible dimensions are
  // padded.
  SmallVector<int64_t> paddedShape;
};

// Returns a ReverseOpInfo for the given reverse operation, or std::nullopt
// if padding size exceeds single shard size and we can't use HALO exchange to
// shift the padding.
std::optional<ReverseOpInfo> getReverseOpInfo(stablehlo::ReverseOp reverseOp,
                                              TensorShardingAttr sharding,
                                              MeshAttr mesh,
                                              RankedTensorType type) {
  ReverseOpInfo info;
  info.paddedShape = llvm::to_vector(type.getShape());

  for (auto [dim, dimSharding] : llvm::enumerate(sharding.getDimShardings())) {
    bool isReversedDim = llvm::is_contained(reverseOp.getDimensions(), dim);
    int64_t shardCount = dimSharding.getShardedSize(mesh);
    if (!isReversedDim || shardCount <= 1) {
      continue;
    }

    for (AxisRefAttr axisRef : dimSharding.getAxes()) {
      info.axesToReverse.push_back(axisRef);
    }

    if (type.getDimSize(dim) % shardCount != 0) {
      for (AxisRefAttr axis : dimSharding.getAxes()) {
        info.manualAxes.insert(axis.getName());
      }
      info.paddedShape[dim] =
          llvm::divideCeil(type.getDimSize(dim), shardCount) * shardCount;

      if (info.paddedShape[dim] - type.getDimSize(dim) >
          info.paddedShape[dim] / shardCount) {
        return std::nullopt;
      }
    }
  }

  return info;
}

// Implements a sharded reverse operation using HALO exchange for indivisible
// dimensions as follows:
//
// 1. Pad indivisible dimensions at the high edge (communication-free).
// 2. Perform HALO exchange to move the high-side padding to the low side.
// 3. Reshard the shifted result to a mesh with reversed device ordering.
// 4. Perform the bulk reverse operation.
// 5. Trim off the padding to produce the final result.
LogicalResult handleReverseOp(stablehlo::ReverseOp reverseOp,
                              IRRewriter& rewriter, SymbolTable& symbolTable,
                              MeshCache& meshCache) {
  Value operand = reverseOp.getOperand();
  TensorShardingAttr inSharding = getSharding(operand);
  auto origType = mlir::dyn_cast<RankedTensorType>(operand.getType());
  if (isFullyReplicated(inSharding) || !origType) {
    return success();
  }

  MeshAttr mesh = inSharding.getMesh(symbolTable);
  if (!mesh || mesh.isMaximal()) {
    return success();
  }

  std::optional<ReverseOpInfo> info =
      getReverseOpInfo(reverseOp, inSharding, mesh, origType);
  if (!info) {
    // Padding size exceeds single shard size, fall back to replication.
    return failure();
  }
  if (info->axesToReverse.empty()) {
    // No dimensions to reverse.
    return success();
  }

  Location loc = reverseOp.getLoc();
  Value input = operand;
  SmallVector<int64_t> dimsToShift, shiftAmounts;

  // Pad indivisible dimensions on the high edge.
  if (!info->manualAxes.empty()) {
    rewriter.setInsertionPoint(reverseOp);
    Type origElementType = origType.getElementType();
    int64_t rank = origType.getRank();
    auto paddedType = RankedTensorType::get(info->paddedShape, origElementType);
    Value zero = stablehlo::ConstantOp::create(
        rewriter, loc,
        rewriter.getZeroAttr(RankedTensorType::get({}, origElementType)));

    SmallVector<int64_t> edgePaddingHigh, zeroPadding(rank, 0);
    for (int64_t i = 0; i < rank; ++i) {
      int64_t padding = info->paddedShape[i] - origType.getDimSize(i);
      edgePaddingHigh.push_back(padding);
      if (llvm::is_contained(reverseOp.getDimensions(), i) && padding > 0) {
        dimsToShift.push_back(i);
        shiftAmounts.push_back(padding);
      }
    }

    auto padOp =
        stablehlo::PadOp::create(rewriter, loc, paddedType, operand, zero,
                                 rewriter.getDenseI64ArrayAttr(zeroPadding),
                                 rewriter.getDenseI64ArrayAttr(edgePaddingHigh),
                                 rewriter.getDenseI64ArrayAttr(zeroPadding));
    setSharding(padOp.getResult(), inSharding);

    // Apply HALO Exchange to shift the padding to the low edge.
    input = shiftPaddingToLow(loc, padOp.getResult(), origType, inSharding,
                              mesh, info->manualAxes, info->paddedShape,
                              dimsToShift, shiftAmounts, rewriter);
  }

  // Reshard to Reversed Mesh.
  MeshAttr newMeshAttr = getMeshWithReversedAxes(mesh, info->axesToReverse);
  MeshOp newMeshOp = getOrCreateMesh(
      loc, reverseOp->getParentOfType<ModuleOp>(), newMeshAttr,
      (inSharding.getMeshName() + "_reversed").str(), symbolTable, meshCache);
  TensorShardingAttr reversedSharding = TensorShardingAttr::get(
      reverseOp->getContext(),
      FlatSymbolRefAttr::get(reverseOp->getContext(), newMeshOp.getName()),
      inSharding.getDimShardings(), inSharding.getReplicatedAxes(),
      inSharding.getUnreducedAxes());
  rewriter.setInsertionPoint(reverseOp);
  input = ReshardOp::create(rewriter, loc, input, reversedSharding);

  // Reverse the data on each device.
  rewriter.modifyOpInPlace(reverseOp, [&]() {
    reverseOp->setOperand(0, input);
    setSharding(reverseOp.getResult(), inSharding);
    if (!info->manualAxes.empty()) {
      reverseOp.getResult().setType(cast<RankedTensorType>(input.getType()));
    }
  });

  // Trim off the padding.
  if (!info->manualAxes.empty()) {
    rewriter.setInsertionPointAfter(reverseOp);
    SmallVector<int64_t> sliceStarts(origType.getRank(), 0);
    auto sliceOp = stablehlo::SliceOp::create(
        rewriter, loc, origType, reverseOp.getResult(),
        rewriter.getDenseI64ArrayAttr(sliceStarts),
        rewriter.getDenseI64ArrayAttr(origType.getShape()),
        rewriter.getDenseI64ArrayAttr(
            SmallVector<int64_t>(origType.getRank(), 1)));

    setSharding(sliceOp.getResult(), inSharding);
    rewriter.replaceAllUsesExcept(reverseOp.getResult(), sliceOp, sliceOp);
  }

  return success();
}

struct ShardyResolvePermutationFactorsPass
    : public impl::ShardyResolvePermutationFactorsPassBase<
          ShardyResolvePermutationFactorsPass> {
  using ShardyResolvePermutationFactorsPassBase::
      ShardyResolvePermutationFactorsPassBase;

 protected:
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(moduleOp);
    SymbolTable symbolTable(moduleOp);
    MeshCache meshCache;

    // Populate the cache with existing meshes in the module.
    for (auto meshOp : moduleOp.getOps<MeshOp>()) {
      meshCache[meshOp.getMesh()] = meshOp;
    }

    // Walk the module to resolve permutation factors for each op.
    moduleOp.walk([&](Operation* op) {
      // Skip terminators and any operations not in the StableHLO dialect.
      // This prevents "unknown op" warnings for Shardy collectives or return
      // ops.
      if (op->hasTrait<OpTrait::IsTerminator>() ||
          !inDialect<stablehlo::StablehloDialect>(op)) {
        return;
      }

      OpShardingRuleAttr rule = getOrCreateShardingRule(op, false, false);
      if (!rule || rule.isCustom()) {
        return;
      }

      // Identify if the op defines any permutation factors.
      auto isPermutation = [&](int64_t i) {
        return rule.getFactorType(i) == FactorType::kPermutation;
      };
      if (llvm::none_of(llvm::seq<int64_t>(0, rule.getNumFactors()),
                        isPermutation)) {
        return;
      }

      // Dispatch to HALO exchange if enabled and implemented for the op.
      if (enableHaloExchange) {
        if (auto reverseOp = dyn_cast<stablehlo::ReverseOp>(op)) {
          // If HALO exchange failed, fall back to explicit reshards below.
          if (succeeded(handleReverseOp(reverseOp, rewriter, symbolTable,
                                        meshCache))) {
            return;
          }
        }
      }

      // Otherwise, use a generic resolution based on explicit reshards.
      SmallVector<TensorShardingAttr> inShardings =
          getShardings(op->getOperands());
      SmallVector<TensorShardingAttr> outShardings =
          getShardings(op->getResults());
      std::optional<StringRef> meshName =
          getCommonMeshName(inShardings, outShardings, symbolTable, true);
      if (!meshName) {
        return;
      }
      MeshOp meshOp = getMeshOp(symbolTable, *meshName);
      if (!meshOp || meshOp.getMesh().isMaximal()) {
        return;
      }

      ShardingProjection projection =
          ShardingProjection::build(inShardings, outShardings, rule,
                                    meshOp.getMesh(), /*closedIfMissing=*/true);
      UpdateTensorShardings update(op->getNumOperands(), op->getNumResults());

      for (int64_t i = 0; i < rule.getNumFactors(); ++i) {
        if (rule.getFactorType(i) != FactorType::kPermutation) {
          continue;
        }
        if (auto sliceOp = dyn_cast<stablehlo::SliceOp>(op)) {
          SDY_CHECK(inShardings[0] == outShardings[0]);
          if (isCommunicationFreeSliceDim(i, sliceOp, inShardings[0],
                                          meshOp.getMesh())) {
            continue;
          }
        }

        update |=
            projection.updateSharding(i, /*axes=*/{}, /*overflowAxes=*/{});
      }

      if (update.updateOperands.any() || update.updateResults.any()) {
        insertExplicitReshards(op, inShardings, outShardings, projection,
                               update, rewriter, rule, symbolTable, meshOp);
      }
    });
  }
};

}  // namespace
}  // namespace sdy
}  // namespace mlir
