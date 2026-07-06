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

// -----------------------------------------------------------------------------
// Data structures.
// -----------------------------------------------------------------------------

using MeshCache = DenseMap<MeshAttr, MeshOp>;

// State maintained during resolving permutation factors.
struct ResolutionState {
  IRRewriter& rewriter;
  SymbolTable& symbolTable;
  MeshCache& meshCache;
  int64_t& nextChannelId;
};

// -----------------------------------------------------------------------------
// Coordinate math & index conversion helpers.
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Mesh & sharding configuration helpers.
// -----------------------------------------------------------------------------

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

// Returns a TensorShardingAttr where all axes in 'axes' are removed.
TensorShardingAttr removeAxesFromSharding(
    TensorShardingAttr sharding, const llvm::SmallDenseSet<StringRef>& axes) {
  SmallVector<DimensionShardingAttr> newDimShardings;
  for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
    SmallVector<AxisRefAttr> newAxes;
    for (AxisRefAttr axis : dimSharding.getAxes()) {
      if (!axes.contains(axis.getName())) {
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

// -----------------------------------------------------------------------------
// Pad and slice helpers for dimension alignment.
// -----------------------------------------------------------------------------

inline int64_t getPaddedDimSize(int64_t dimSize, int64_t shardCount) {
  return llvm::divideCeil(dimSize, shardCount) * shardCount;
}

// Pads the high side of `operand` with `paddingValue` (default zero if null)
// to match the target `paddedShape`. When used to align dimensions for the
// input sharding's own divisibility, the created pad op is a
// communication-free operation. However, when this routine is used to pad
// the input based on the output sharding's divisibility, such as to resolve
// the Reshape op, the created pad op shifts data across device boundaries
// and requires a HALO exchange. In this case, `opsToResolve` can be used to
// track the created op for later resolution.
Value padHighSideToShape(Location loc, IRRewriter& rewriter, Value operand,
                         ArrayRef<int64_t> paddedShape,
                         TensorShardingAttr sharding,
                         Value paddingValue = nullptr,
                         SmallVectorImpl<Operation*>* opsToResolve = nullptr) {
  auto type = cast<RankedTensorType>(operand.getType());
  if (paddedShape == type.getShape()) {
    return operand;
  }
  auto paddedType = RankedTensorType::get(paddedShape, type.getElementType());
  SmallVector<int64_t> edgePaddingHigh;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    edgePaddingHigh.push_back(paddedShape[i] - type.getDimSize(i));
  }
  Value padVal = paddingValue;
  if (!padVal) {
    auto zeroAttr =
        rewriter.getZeroAttr(RankedTensorType::get({}, type.getElementType()));
    padVal = stablehlo::ConstantOp::create(rewriter, loc, zeroAttr);
  }
  auto padHighOp = stablehlo::PadOp::create(
      rewriter, loc, paddedType, operand, padVal,
      rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(type.getRank(), 0)),
      rewriter.getDenseI64ArrayAttr(edgePaddingHigh),
      rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(type.getRank(), 0)));
  if (opsToResolve) {
    opsToResolve->push_back(padHighOp);
  }
  setSharding(padHighOp.getResult(), sharding);
  return padHighOp.getResult();
}

// Slices the high side of `operand` (e.g., to trim away high-side padding
// elements previously inserted by `padHighSideToShape`) to match `targetType`.
// When used to align dimensions after local operations complete, the created
// slice op is a communication-free operation. However, when this routine is
// used to trim the padded output based on the output sharding's divisibility,
// such as to resolve the Reshape op, the created slice op shifts data across
// device boundaries and requires a HALO exchange. In this case, `opsToResolve`
// can be used to track the created op for later resolution.
Value sliceHighSideToShape(
    Location loc, IRRewriter& rewriter, Value operand, Type targetType,
    TensorShardingAttr sharding,
    SmallVectorImpl<Operation*>* opsToResolve = nullptr) {
  auto rankedTargetType = cast<RankedTensorType>(targetType);
  if (operand.getType() == rankedTargetType) {
    return operand;
  }
  int64_t rank = rankedTargetType.getRank();
  SmallVector<int64_t> sliceStarts(rank, 0);
  SmallVector<int64_t> sliceStrides(rank, 1);
  auto sliceOp = stablehlo::SliceOp::create(
      rewriter, loc, rankedTargetType, operand,
      rewriter.getDenseI64ArrayAttr(sliceStarts),
      rewriter.getDenseI64ArrayAttr(rankedTargetType.getShape()),
      rewriter.getDenseI64ArrayAttr(sliceStrides));
  if (opsToResolve) {
    opsToResolve->push_back(sliceOp);
  }
  setSharding(sliceOp.getResult(), sharding);
  return sliceOp.getResult();
}

// -----------------------------------------------------------------------------
// Core HALO exchange helpers.
// -----------------------------------------------------------------------------

// For each logical device i in the 'manualMesh', computes the logical device
// j that should receive the shifted data of i if the dimension is
// partitioned by the manual axes in 'manualAxesInDim'. The number of devices
// to shift is given by 'shardOffset'. Returns all the (i, j) pairs where i is
// the source device and j is the target device.
SmallVector<int64_t> getRightShiftSourceTargetPairs(
    MeshAttr manualMesh, ArrayRef<AxisRefAttr> manualAxesInDim,
    int64_t shardOffset) {
  ArrayRef<MeshAxisAttr> manualMeshAxes = manualMesh.getAxes();
  int64_t totalSize = manualMesh.getTotalSize();
  SmallVector<int64_t> pairs;

  for (int64_t i = 0; i < totalSize; ++i) {
    SmallVector<int64_t> coords = getMeshCoordinates(i, manualMeshAxes);

    SmallVector<int64_t> nextCoords = coords;
    int64_t currentCarry = shardOffset;
    for (int64_t k = (int64_t)manualAxesInDim.size() - 1;
         k >= 0 && currentCarry != 0; --k) {
      auto axisRef = manualAxesInDim[k];
      int64_t meshIdx = getMeshAxisIndex(manualMeshAxes, axisRef.getName());

      int64_t fullSize = manualMeshAxes[meshIdx].getSize();
      int64_t subSize = axisRef.getSize(manualMesh);
      int64_t preSize = axisRef.getSubAxisPreSize();
      int64_t postSize = fullSize / (preSize * subSize);

      int64_t subCoord = (coords[meshIdx] / postSize) % subSize;
      int64_t newSubCoord = subCoord + currentCarry;

      currentCarry = llvm::divideFloorSigned(newSubCoord, subSize);
      int64_t wrappedSubCoord = llvm::mod(newSubCoord, subSize);

      nextCoords[meshIdx] += (wrappedSubCoord - subCoord) * postSize;
    }

    if (currentCarry == 0) {
      ArrayRef<int64_t> deviceIds = manualMesh.getDeviceIds();
      int64_t nextFlatIndex =
          getLinearIndexFromCoordinates(nextCoords, manualMeshAxes);
      int64_t src = deviceIds.empty() ? i : deviceIds[i];
      int64_t tgt =
          deviceIds.empty() ? nextFlatIndex : deviceIds[nextFlatIndex];
      pairs.push_back(src);
      pairs.push_back(tgt);
    }
  }
  return pairs;
}

// =============================================================================
// Implementation of handleXYZOps routines in alphabetical order.
// =============================================================================

// -----------------------------------------------------------------------------
// stablehlo.reverse
// -----------------------------------------------------------------------------

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

// Returns a ReverseOpInfo for the given reverse operation. Returns nullopt if
// there is no need to apply the sharded reverse operation.
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
          getPaddedDimSize(type.getDimSize(dim), shardCount);
    }
  }

  if (info.axesToReverse.empty()) {
    return std::nullopt;
  }
  return info;
}

// Generates code to right shift elements logically in each local device across
// the partitioning space using HALO exchange and support multi-hop shifts.
Value rightShiftData(Location loc, Value input, int64_t dim, int64_t totalShift,
                     TensorShardingAttr sharding, MeshAttr mesh,
                     const llvm::SmallDenseSet<StringRef>& manualAxes,
                     ResolutionState& state) {
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t shardSize = inputType.getDimSize(dim);
  int64_t rank = inputType.getRank();
  removeAxesFromSharding(sharding, manualAxes);
  TensorShardingAttr localSharding =
      removeAxesFromSharding(sharding, manualAxes);

  // Parse manual axes for the sharded dimension.
  SmallVector<AxisRefAttr> manualAxesInDim;
  for (auto axisRef : sharding.getDimShardings()[dim].getAxes()) {
    if (manualAxes.contains(axisRef.getName())) {
      manualAxesInDim.push_back(axisRef);
    }
  }

  int64_t hops = totalShift / shardSize;
  int64_t fraction = totalShift % shardSize;

  // Helper to collective permute a sliced piece by a given device offset.
  auto permutePiece = [&](Value piece, int64_t deviceOffset) -> Value {
    if (deviceOffset == 0) {
      return piece;
    }
    SmallVector<int64_t> pairs =
        getRightShiftSourceTargetPairs(mesh, manualAxesInDim, deviceOffset);
    if (pairs.empty()) {
      auto zeroAttr = state.rewriter.getZeroAttr(
          RankedTensorType::get({}, inputType.getElementType()));
      Value zeroConst =
          stablehlo::ConstantOp::create(state.rewriter, loc, zeroAttr);
      auto bcast = stablehlo::BroadcastInDimOp::create(
          state.rewriter, loc, cast<RankedTensorType>(piece.getType()),
          zeroConst, state.rewriter.getDenseI64ArrayAttr({}));
      setSharding(bcast, localSharding);
      return bcast;
    }
    auto pairType =
        RankedTensorType::get({static_cast<int64_t>(pairs.size()) / 2, 2},
                              state.rewriter.getI64Type());
    auto channelAttr = stablehlo::ChannelHandleAttr::get(
        state.rewriter.getContext(), state.nextChannelId++, 1);
    auto permOp = stablehlo::CollectivePermuteOp::create(
        state.rewriter, loc, piece.getType(), piece,
        DenseIntElementsAttr::get(pairType, pairs), channelAttr);
    setSharding(permOp.getResult(), localSharding);
    return permOp.getResult();
  };

  Value head = permutePiece(input, hops);
  Value tail = permutePiece(input, hops + 1);

  int64_t headSliceSize = shardSize - fraction;

  // tailSlice slices tail[shardSize - fraction : shardSize]
  SmallVector<int64_t> tailStarts(rank, 0);
  tailStarts[dim] = headSliceSize;

  SmallVector<int64_t> tailShape = llvm::to_vector(inputType.getShape());
  tailShape[dim] = fraction;
  auto tailType = RankedTensorType::get(tailShape, inputType.getElementType());
  auto tailSlice = stablehlo::SliceOp::create(
      state.rewriter, loc, tailType, tail,
      state.rewriter.getDenseI64ArrayAttr(tailStarts),
      state.rewriter.getDenseI64ArrayAttr(inputType.getShape()),
      state.rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(rank, 1)));
  setSharding(tailSlice.getResult(), localSharding);

  // headSlice slices head[0 : headSliceSize]
  SmallVector<int64_t> headLimits = llvm::to_vector(inputType.getShape());
  headLimits[dim] = headSliceSize;

  auto headType = RankedTensorType::get(headLimits, inputType.getElementType());
  auto headSlice = stablehlo::SliceOp::create(
      state.rewriter, loc, headType, head,
      state.rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(rank, 0)),
      state.rewriter.getDenseI64ArrayAttr(headLimits),
      state.rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(rank, 1)));
  setSharding(headSlice.getResult(), localSharding);

  // Concat tailSlice followed by headSlice
  Value concat = stablehlo::ConcatenateOp::create(
      state.rewriter, loc,
      ValueRange{tailSlice.getResult(), headSlice.getResult()}, dim);
  setSharding(concat, localSharding);
  return concat;
}

// Generates the local device code wrapped inside a manual computation to
// implement HALO exchange for shifting data to the right. This effectively
// moves the padding in the operand of a reverse op from high edge to low edge.
Value haloRightShiftData(Location loc, Value input, RankedTensorType origType,
                         TensorShardingAttr sharding, MeshAttr mesh,
                         const llvm::SmallDenseSet<StringRef>& manualAxes,
                         ArrayRef<int64_t> paddedShape,
                         ArrayRef<int64_t> dimsToShift,
                         ArrayRef<int64_t> shiftAmounts,
                         ResolutionState& state) {
  // Compute the local tensor shape for the operand inside manual computation.
  SmallVector<int64_t> localShape;
  localShape.reserve(sharding.getDimShardings().size());
  auto inputType = cast<RankedTensorType>(input.getType());
  for (auto [dim, dimSharding] : llvm::enumerate(sharding.getDimShardings())) {
    int64_t manualFactor = 1;
    for (auto axis : dimSharding.getAxes()) {
      if (manualAxes.contains(axis.getName())) {
        manualFactor *= axis.getSize(mesh);
      }
    }
    localShape.push_back(paddedShape[dim] / manualFactor);
  }

  // Get the manual axes for the manual computation.
  SmallVector<StringAttr> manualAxesAttrs;
  for (auto axis : mesh.getAxes()) {
    if (manualAxes.contains(axis.getName())) {
      manualAxesAttrs.push_back(state.rewriter.getStringAttr(axis.getName()));
    }
  }

  // Build the manual computation.
  auto manualComp =
      ManualComputationOp::create(state.rewriter, loc, input.getType(), input,
                                  {sharding}, {sharding}, manualAxesAttrs);

  Region& body = manualComp.getBody();
  body.emplaceBlock();
  Value shiftedLocal = body.addArgument(
      RankedTensorType::get(localShape, inputType.getElementType()), loc);

  OpBuilder::InsertionGuard guard(state.rewriter);
  state.rewriter.setInsertionPointToStart(&body.front());

  for (auto [k, dim] : llvm::enumerate(dimsToShift)) {
    shiftedLocal = rightShiftData(loc, shiftedLocal, dim, shiftAmounts[k],
                                  sharding, mesh, manualAxes, state);
  }

  ReturnOp::create(state.rewriter, loc, shiftedLocal);
  return manualComp.getResult(0);
}

// Implements a sharded reverse operation using HALO exchange for indivisible
// dimensions as follows:
//
// 1. Pad indivisible dimensions at the high edge (communication-free).
// 2. Perform HALO exchange to shift data so the padding moves to the low side.
// 3. Perform the local reverse operation on the HALO exchanged result.
// 4. Reshard the reversed result back to the original mesh sharding.
// 5. Trim off the padding to produce the final result.
LogicalResult handleReverseOp(stablehlo::ReverseOp reverseOp,
                              ResolutionState& state) {
  Value operand = reverseOp.getOperand();
  TensorShardingAttr inSharding = getSharding(operand);
  TensorShardingAttr outSharding = getSharding(reverseOp.getResult());
  auto origType = mlir::dyn_cast<RankedTensorType>(operand.getType());
  if (isFullyReplicated(inSharding) || !origType) {
    return success();
  }

  MeshAttr mesh = inSharding.getMesh(state.symbolTable);
  if (!mesh || mesh.isMaximal()) {
    return success();
  }

  std::optional<ReverseOpInfo> info =
      getReverseOpInfo(reverseOp, inSharding, mesh, origType);
  if (!info) {
    // No sharded dimensions to reverse.
    return success();
  }

  Location loc = reverseOp.getLoc();
  Value input = operand;

  // Pad indivisible dimensions on the high edge.
  if (!info->manualAxes.empty()) {
    state.rewriter.setInsertionPoint(reverseOp);
    input = padHighSideToShape(loc, state.rewriter, operand, info->paddedShape,
                               inSharding);

    SmallVector<int64_t> dimsToShift, shiftAmounts;
    int64_t rank = origType.getRank();
    for (int64_t i = 0; i < rank; ++i) {
      int64_t padding = info->paddedShape[i] - origType.getDimSize(i);
      if (llvm::is_contained(reverseOp.getDimensions(), i) && padding > 0) {
        dimsToShift.push_back(i);
        shiftAmounts.push_back(padding);
      }
    }

    // Apply HALO Exchange to shift the padding to the low edge.
    input = haloRightShiftData(loc, input, origType, inSharding, mesh,
                               info->manualAxes, info->paddedShape, dimsToShift,
                               shiftAmounts, state);
  }

  // Construct Reversed Mesh to represent the reverse op result sharding.
  MeshAttr newMeshAttr = getMeshWithReversedAxes(mesh, info->axesToReverse);
  MeshOp newMeshOp =
      getOrCreateMesh(loc, reverseOp->getParentOfType<ModuleOp>(), newMeshAttr,
                      inSharding.getMeshName().str() + "_reversed",
                      state.symbolTable, state.meshCache);
  TensorShardingAttr reversedSharding = TensorShardingAttr::get(
      reverseOp->getContext(),
      FlatSymbolRefAttr::get(reverseOp->getContext(), newMeshOp.getName()),
      inSharding.getDimShardings(), inSharding.getReplicatedAxes(),
      inSharding.getUnreducedAxes());
  // Replace the reverse op input with the padded/shifted input and update the
  // result sharding to reversedSharding.
  state.rewriter.modifyOpInPlace(reverseOp, [&]() {
    reverseOp->setOperand(0, input);
    setSharding(reverseOp.getResult(), reversedSharding);
    if (!info->manualAxes.empty()) {
      reverseOp.getResult().setType(cast<RankedTensorType>(input.getType()));
    }
  });

  // Reshard the reversed result back to the original result sharding.
  state.rewriter.setInsertionPointAfter(reverseOp);
  Value reversedResult = ReshardOp::create(state.rewriter, loc,
                                           reverseOp.getResult(), outSharding);

  // Trim off the padding.
  if (!info->manualAxes.empty()) {
    Value slicedResult = sliceHighSideToShape(
        loc, state.rewriter, reversedResult, origType, inSharding);
    state.rewriter.replaceAllUsesExcept(reverseOp.getResult(), slicedResult,
                                        reversedResult.getDefiningOp());
  } else {
    state.rewriter.replaceAllUsesExcept(reverseOp.getResult(), reversedResult,
                                        reversedResult.getDefiningOp());
  }

  return success();
}

// Returns the maximum channel ID in `moduleOp` plus one.
int64_t getNextChannelId(ModuleOp moduleOp) {
  int64_t maxChannelId = 0;
  moduleOp->walk([&](Operation* op) {
    if (auto channelHandle =
            op->getAttrOfType<stablehlo::ChannelHandleAttr>("channel_handle")) {
      maxChannelId = std::max(maxChannelId, channelHandle.getHandle());
    }
  });
  return maxChannelId + 1;
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

    int64_t nextChannelId = getNextChannelId(moduleOp);
    ResolutionState state{rewriter, symbolTable, meshCache, nextChannelId};

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
          if (succeeded(handleReverseOp(reverseOp, state))) {
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
