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
#include <functional>
#include <limits>
#include <optional>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"  // IWYU pragma: keep
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
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
  int64_t replicaCount = 1;
  int64_t partitionCount = 1;
  SmallVector<Operation*> opsToErase;
};

// Returns a channel handle with handle = nextChannelId++ and type = 1
// (CROSS_PARTITION) if usePartitionId is true. Otherwise returns nullptr (to
// perform cross-replica communication without channel handle).
stablehlo::ChannelHandleAttr getChannelHandle(ResolutionState& state) {
  return sdy::getChannelHandle(state.rewriter.getContext(), state.replicaCount,
                               state.partitionCount, state.nextChannelId);
}

// Describes the data exchange needed to perform explicit resharding for a
// single shard in a dimension.
struct ShardExchangeInfo {
  int64_t sourceId;     // Source shard index.
  int64_t localOffset;  // Local start index in that source shard.
};

// Dimension exchange info specific to a reshape operation.
struct ReshapeDimExchangeInfo {
  // Combined dimension sharding attribute of the target output dimensions
  // (groupOutDims) used to convert global partition IDs into local group
  // partition coordinates.
  DimensionShardingAttr targetDimSharding;

  // Returns the target dimension sharding attribute for partition coordinate
  // conversion.
  DimensionShardingAttr getTargetDimSharding() const {
    SDY_CHECK(targetDimSharding);
    return targetDimSharding;
  }
};

// Describes the data exchange needed to perform explicit resharding for a
// single dimension of a tensor operand with permutation factors.
struct DimExchangeInfo {
  // shardExchanges[i] contains the source shard index and local offset for the
  // i-th shard in the dimension.
  SmallVector<ShardExchangeInfo> shardExchanges;
  // Aligned logical output shard size for the dimension.
  int64_t sOut;
  // The length of the local slice (input footprint) in the physical/un-dilated
  // input space needed by the device, calculated as:
  //    sFootprint = divideCeil((sOut - 1) * stride + dilatedWindowSize - 1,
  //                 baseDilation) + 1
  int64_t sFootprint;
  // Max left and right hops required for dimension exchange.
  int64_t leftHops = 0;
  int64_t rightHops = 0;
  // Indicates if the exchange for this dimension requires cross-device
  // communication.
  bool needsComm = false;
  // Indicates if the exchange for this dimension is supported by HALO exchange.
  // We cannot use HALO exchange if the number of gathered shards (leftHops +
  // rightHops + 1) is equal to or larger than the total shard count in the
  // dimension, AND we need more than 1 hop in at least one direction.
  bool canUseHalo = true;
  // Reshape-specific data, present if this exchange is for a reshape op.
  std::optional<ReshapeDimExchangeInfo> reshapeInfo;
};

// Describes the data exchange needed to perform explicit resharding for a
// tensor operand with permutation factors.
struct DataExchangeInfo {
  // The result shape after padding the input for sharding divisibility.
  SmallVector<int64_t> divisibleInputShape;
  // Contains the axes sharding all sharded dimensions (i.e. dim size > 1).
  llvm::SmallDenseSet<AxisRefAttr> manualAxes;
  // dimExchanges[i] contains the exchange info for the i-th sharded
  // dimension of the sharded tensor operand.
  SmallVector<DimExchangeInfo, 2> dimExchanges;

  // Returns true if any of the sharded dimensions require cross-device
  // communication.
  bool needsComm() const {
    return llvm::any_of(dimExchanges, [](const DimExchangeInfo& dimExchange) {
      return dimExchange.needsComm;
    });
  }
};

// Returns true if the data exchange can be resolved using HALO exchange.
bool canUseHalo(const DataExchangeInfo& info) {
  return llvm::all_of(info.dimExchanges,
                      [](const DimExchangeInfo& dimExchange) {
                        return dimExchange.canUseHalo;
                      });
}

// Sharded dimension exchange context.
struct ShardedDimContext {
  // Aligned logical input shard size for the dimension.
  int64_t sIn;
  // The number of devices used for sharding this dimension.
  int64_t shardCount;
};

// Values used to dynamic-slice the assembled HALO buffer.
//
// For sharded dimensions, we concatenate neighbor shards and pad them to
// assemble the HALO buffer. For pad and slice ops, the buffer is dilated,
// so we use dilatedOffset to slice the segment (as dilation is 1 for slice,
// dilatedOffset is equal to undilatedOffset). For window ops, the buffer is
// un-dilated, so we use undilatedOffset to slice the segment.
struct DeviceOffsetInfo {
  Value dilatedOffset;
  // max(dilatedOffset / baseDilation, 0).
  Value undilatedOffset;
  // A constant 0 of type i64, used to initialize offsets of other dimensions
  // in the dynamic slice array.
  Value zeroConst;
};

using PostProcessFn =
    std::function<Value(Value, Value, RankedTensorType, ResolutionState&)>;
using VariadicPostProcessFn = std::function<SmallVector<Value>(
    ValueRange, ValueRange, ArrayRef<RankedTensorType>, ResolutionState&)>;

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

// Decomposes `shardIndex` into a multi-dimensional coordinate in
// `groupOutDims`, then uses this coordinate and `groupOutStrides` to compute
// the linear start offset in unpadded output space.
int64_t getLinearDataStartOffsetForShardIdx(int64_t shardIndex,
                                            ArrayRef<int64_t> groupOutDims,
                                            ArrayRef<int64_t> groupOutStrides,
                                            TensorShardingAttr outSharding,
                                            RankedTensorType resultType,
                                            MeshAttr mesh) {
  int64_t linearStartOffset = 0;
  int64_t remainingShardIdx = shardIndex;
  for (int64_t idx = static_cast<int64_t>(groupOutDims.size()) - 1; idx >= 0;
       --idx) {
    int64_t outDim = groupOutDims[idx];
    int64_t numShardsPerDim =
        outSharding.getDimShardings()[outDim].getShardedSize(mesh);
    int64_t shardCoord = remainingShardIdx % numShardsPerDim;
    remainingShardIdx /= numShardsPerDim;

    int64_t shardTileSize =
        llvm::divideCeil(resultType.getDimSize(outDim), numShardsPerDim);
    int64_t dimStride = groupOutStrides[idx];
    linearStartOffset += shardCoord * shardTileSize * dimStride;
  }
  return linearStartOffset;
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

Value createDeviceId(Location loc, ResolutionState& state) {
  return getDeviceId(loc, state.replicaCount, state.partitionCount,
                     state.rewriter);
}

// Compute and return the following values from the input:
//
// dilatedOffset = partitionId * diffSize + baseOffset
// physicalOffset = max(dilatedOffset / baseDilation, 0)
//
// Derivation of dilatedOffset:
// 1. In a global coordinate system, the start index of the partition's
//    needed input segment is:
//      globalOffset = partitionId * sOut * stride - padLow
// 2. The partition's own local input starts at:
//      globalInputStart = partitionId * sInDilated
// 3. However, inside the locally assembled HALO buffer, the left neighbor
//    segments and low-padding shift the local input's start position to:
//      localInputStart = leftHops * sInDilated + sFootprint * baseDilation
// 4. Therefore, the coordinate translation shift to map from the global space
//    to the local HALO buffer is:
//      shift = localInputStart - globalInputStart
//            = leftHops * sInDilated + sFootprint * baseDilation
//              - partitionId * sInDilated
// 5. Applying this shift to get the local offset inside the HALO buffer:
//      localOffset = globalOffset + shift
//                  = partitionId * sOut * stride - padLow
//                    + leftHops * sInDilated + sFootprint * baseDilation
//                    - partitionId * sInDilated
//                  = partitionId * (sOut * stride - sInDilated) - padLow
//                    + leftHops * sInDilated + sFootprint * baseDilation
//                  = partitionId * diffSize + baseOffset
//    where:
//      diffSize = sOut * stride - sInDilated
//      baseOffset = -padLow + leftHops * sInDilated
//                                 + sFootprint * baseDilation
DeviceOffsetInfo getDeviceOffsetInfo(Location loc, Value partitionId,
                                     int64_t diffSize, int64_t baseOffset,
                                     int64_t baseDilation,
                                     ResolutionState& state,
                                     bool needUndilated = true,
                                     bool clampToZero = true) {
  IRRewriter& rewriter = state.rewriter;
  Type i32Ty = rewriter.getI32Type();
  Value dilatedOffset;
  Value zeroConst;
  // When diffSize == 0 and baseOffset >= 0, every shard has the same
  // non-negative offset in the HALO buffer, so fold directly to a constant.
  if (diffSize == 0 && baseOffset >= 0) {
    dilatedOffset = createScalarIntConstant(loc, i32Ty, baseOffset, rewriter);
    zeroConst =
        createZeroConstant(loc, RankedTensorType::get({}, i32Ty), rewriter);
  } else {
    Value diffSizeVal = createScalarIntConstant(loc, i32Ty, diffSize, rewriter);
    Value baseOffsetVal =
        createScalarIntConstant(loc, i32Ty, baseOffset, rewriter);
    Value offsetInPartition =
        stablehlo::MulOp::create(rewriter, loc, partitionId, diffSizeVal);
    dilatedOffset = stablehlo::AddOp::create(rewriter, loc, offsetInPartition,
                                             baseOffsetVal);
    zeroConst =
        createZeroConstant(loc, RankedTensorType::get({}, i32Ty), rewriter);
    // Only clamp dilatedOffset to 0 when the minimum offset across all shards
    // can be negative.
    if (clampToZero) {
      dilatedOffset =
          stablehlo::MaxOp::create(rewriter, loc, dilatedOffset, zeroConst);
    }
  }

  Value undilatedOffset;
  if (needUndilated) {
    if (baseDilation == 1) {
      undilatedOffset = dilatedOffset;
    } else if (diffSize == 0 && baseOffset >= 0) {
      undilatedOffset = createScalarIntConstant(
          loc, i32Ty, baseOffset / baseDilation, rewriter);
    } else {
      Value baseDilationVal =
          createScalarIntConstant(loc, i32Ty, baseDilation, rewriter);
      undilatedOffset =
          stablehlo::DivOp::create(rewriter, loc, dilatedOffset.getType(),
                                   dilatedOffset, baseDilationVal);
    }
  }
  return {dilatedOffset, undilatedOffset, zeroConst};
}

// -----------------------------------------------------------------------------
// Window and padding attribute helpers.
// -----------------------------------------------------------------------------

// Returns the value at 'dim' for an optional dimension attribute array,
// returning the specified default value if the attribute is absent. This
// routine is used to inspect attributes such as window_strides and
// window_dilations, where the absence of these attributes implies a default
// value of 1.
template <typename T>
int64_t getDimProperty(const std::optional<T>& propertyAttr, int64_t dim,
                       int64_t defaultValue = 1) {
  return propertyAttr ? (*propertyAttr)[dim] : defaultValue;
}

// Unpacks an optional property attribute array into a list of values for each
// dimension, defaulting to 'defaultValue' when the attribute is absent.
template <typename T>
SmallVector<int64_t> getDimProperties(const std::optional<T>& propertyAttr,
                                      int64_t rank, int64_t defaultValue = 1) {
  SmallVector<int64_t> properties;
  properties.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    properties.push_back(getDimProperty(propertyAttr, i, defaultValue));
  }
  return properties;
}

// Unpacks an optional property attribute array and maps it to spatial
// dimensions, defaulting to 'defaultValue' on other (non-spatial) dimensions.
template <typename T>
SmallVector<int64_t> getSpatialProperties(const std::optional<T>& propertyAttr,
                                          int64_t rank,
                                          ArrayRef<int64_t> spatialDimensions,
                                          int64_t defaultValue = 1) {
  SmallVector<int64_t> properties(rank, defaultValue);
  for (auto [i, dim] : llvm::enumerate(spatialDimensions)) {
    properties[dim] = getDimProperty(propertyAttr, i, defaultValue);
  }
  return properties;
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

// Returns a TensorShardingAttr where all specific axis references in
// axesToRemove are removed.
TensorShardingAttr removeAxisRefsFromSharding(
    TensorShardingAttr sharding, ArrayRef<AxisRefAttr> axesToRemove,
    MeshAttr mesh) {
  if (axesToRemove.empty()) {
    return sharding;
  }
  SmallVector<DimensionShardingAttr> newDimShardings(
      sharding.getDimShardings());
  for (DimensionShardingAttr& dimSharding : newDimShardings) {
    SmallVector<AxisRefAttr> newAxes =
        getAxisSetDiff(dimSharding.getAxes(), axesToRemove, mesh);
    dimSharding = DimensionShardingAttr::get(sharding.getContext(), newAxes,
                                             dimSharding.getIsClosed(),
                                             dimSharding.getPriority());
  }
  return TensorShardingAttr::get(sharding.getContext(), sharding.getMeshOrRef(),
                                 newDimShardings, sharding.getReplicatedAxes(),
                                 sharding.getUnreducedAxes(),
                                 sharding.getReductionOp());
}

// Returns a TensorShardingAttr where all axes overlapping `axes` are removed.
TensorShardingAttr removeAxesFromSharding(
    TensorShardingAttr sharding, const llvm::SmallDenseSet<AxisRefAttr>& axes,
    MeshAttr mesh) {
  SmallVector<AxisRefAttr> axesToRemove(axes.begin(), axes.end());
  return removeAxisRefsFromSharding(sharding, axesToRemove, mesh);
}

int64_t getManualShardCount(
    DimensionShardingAttr dimSharding, MeshAttr mesh,
    const llvm::SmallDenseSet<AxisRefAttr>& manualAxes) {
  int64_t count = 1;
  for (AxisRefAttr axis : dimSharding.getAxes()) {
    for (AxisRefAttr manualAxis : manualAxes) {
      if (std::optional<AxisRefAttr> overlap = axis.getOverlap(manualAxis)) {
        count *= overlap->getSize(mesh);
      }
    }
  }
  return count;
}

// Given the padded global dimension size and the manual computation axes,
// returns the local dimension size for the dimension inside manual computation.
int64_t getLocalDimSize(DimensionShardingAttr dimSharding, int64_t dimSize,
                        MeshAttr mesh,
                        const llvm::SmallDenseSet<AxisRefAttr>& manualAxes) {
  int64_t manualFactor = getManualShardCount(dimSharding, mesh, manualAxes);
  SDY_CHECK(dimSize % manualFactor == 0);
  return dimSize / manualFactor;
}

SmallVector<AxisRefAttr> getManualAxesAttrs(
    MeshAttr mesh, const llvm::SmallDenseSet<AxisRefAttr>& manualAxes) {
  SmallVector<AxisRefAttr> manualAxesAttrs(manualAxes.begin(),
                                           manualAxes.end());
  sortAndMergeAxes(manualAxesAttrs, mesh);
  return manualAxesAttrs;
}

SmallVector<AxisRefAttr> getManualAxesInDim(
    DimensionShardingAttr dimSharding, MeshAttr mesh,
    const llvm::SmallDenseSet<AxisRefAttr>& manualAxes) {
  SmallVector<AxisRefAttr> sortedManualAxes =
      getManualAxesAttrs(mesh, manualAxes);
  SmallVector<AxisRefAttr> manualAxesInDim;
  for (AxisRefAttr axisRef : dimSharding.getAxes()) {
    for (AxisRefAttr manualAxis : sortedManualAxes) {
      if (std::optional<AxisRefAttr> overlap = axisRef.getOverlap(manualAxis)) {
        manualAxesInDim.push_back(*overlap);
      }
    }
  }
  return manualAxesInDim;
}

// -----------------------------------------------------------------------------
// Pad and slice helpers for dimension alignment.
// -----------------------------------------------------------------------------

inline int64_t getPaddedDimSize(int64_t dimSize, int64_t shardCount) {
  return llvm::alignTo(dimSize, shardCount);
}

// Creates a zero scalar constant of element type `elemType`.
inline Value createZeroScalarConstant(Location loc, Type elemType,
                                      IRRewriter& rewriter) {
  return createZeroConstant(loc, RankedTensorType::get({}, elemType), rewriter);
}

// Pads the high side of `operand` with `paddingValue` (default zero if null)
// to match the target `paddedShape`. When used to align dimensions for the
// input sharding's own divisibility, the created pad op is a
// communication-free operation.
Value padHighSideToShape(Location loc, Value operand,
                         ArrayRef<int64_t> paddedShape,
                         TensorShardingAttr sharding, IRRewriter& rewriter,
                         Value paddingValue = nullptr) {
  auto type = cast<RankedTensorType>(operand.getType());
  if (paddedShape == type.getShape()) {
    return operand;
  }
  auto paddedType = RankedTensorType::get(paddedShape, type.getElementType());
  return padHighSideToType(loc, operand, paddedType, sharding, rewriter,
                           paddingValue);
}

// Slices val along dimension dim on [sliceStart, sliceLimit) with stride 1 and
// attaches sharding to the result. Returns val directly if skipFullDimSlice is
// true and the slice covers the full dimension.
Value sliceAlongDim(Location loc, Value val, int64_t dim, int64_t sliceStart,
                    int64_t sliceLimit, TensorShardingAttr sharding,
                    IRRewriter& rewriter, bool skipFullDimSlice = true) {
  auto valType = cast<RankedTensorType>(val.getType());
  if (skipFullDimSlice && sliceStart == 0 &&
      sliceLimit == valType.getDimSize(dim)) {
    return val;
  }
  int64_t rank = valType.getRank();
  SmallVector<int64_t> starts(rank, 0);
  SmallVector<int64_t> limits = llvm::to_vector(valType.getShape());
  SmallVector<int64_t> strides(rank, 1);
  starts[dim] = sliceStart;
  limits[dim] = sliceLimit;
  SmallVector<int64_t> slicedShape = llvm::to_vector(valType.getShape());
  slicedShape[dim] = sliceLimit - sliceStart;
  auto slicedType =
      RankedTensorType::get(slicedShape, valType.getElementType());
  auto sliceOp = stablehlo::SliceOp::create(
      rewriter, loc, slicedType, val, rewriter.getDenseI64ArrayAttr(starts),
      rewriter.getDenseI64ArrayAttr(limits),
      rewriter.getDenseI64ArrayAttr(strides));
  setSharding(sliceOp.getResult(), sharding);
  return sliceOp.getResult();
}

// -----------------------------------------------------------------------------
// Core HALO exchange helpers.
// -----------------------------------------------------------------------------

// For each logical device i in the 'manualMesh', computes the logical device
// j that should receive the shifted data of i if the dimension is
// partitioned by the manual axes in 'manualAxesInDim'. The number of devices
// to shift is given by 'shardOffset'. If cyclicParticipatingShards is set,
// shifts cyclically across the first cyclicParticipatingShards shards along
// manualAxesInDim. Returns all the (i, j) pairs where i is the source device
// and j is the target device.
SmallVector<int64_t> getDataShiftSourceTargetPairs(
    MeshAttr manualMesh, ArrayRef<AxisRefAttr> manualAxesInDim,
    int64_t shardOffset,
    std::optional<int64_t> cyclicParticipatingShards = std::nullopt) {
  ArrayRef<MeshAxisAttr> manualMeshAxes = manualMesh.getAxes();
  int64_t totalSize = manualMesh.getTotalSize();
  SmallVector<int64_t> pairs;

  struct AxisShiftInfo {
    int64_t meshIdx;
    int64_t subSize;
    int64_t postSize;
  };
  SmallVector<AxisShiftInfo> axisInfos;
  axisInfos.reserve(manualAxesInDim.size());
  int64_t totalShardsInDim = 1;
  for (AxisRefAttr axisRef : manualAxesInDim) {
    int64_t meshIdx = getMeshAxisIndex(manualMeshAxes, axisRef.getName());
    int64_t fullSize = manualMeshAxes[meshIdx].getSize();
    int64_t subSize = axisRef.getSize(manualMesh);
    int64_t preSize = axisRef.getSubAxisPreSize();
    int64_t postSize = fullSize / (preSize * subSize);
    axisInfos.push_back({meshIdx, subSize, postSize});
    totalShardsInDim *= subSize;
  }

  ArrayRef<int64_t> deviceIds = manualMesh.getDeviceIds();
  for (int64_t i = 0; i < totalSize; ++i) {
    SmallVector<int64_t> coords = getMeshCoordinates(i, manualMeshAxes);
    int64_t coordInDim = 0;
    for (const AxisShiftInfo& info : axisInfos) {
      int64_t subCoord = (coords[info.meshIdx] / info.postSize) % info.subSize;
      coordInDim = coordInDim * info.subSize + subCoord;
    }

    int64_t targetCoordInDim = coordInDim + shardOffset;
    if (cyclicParticipatingShards) {
      if (coordInDim >= *cyclicParticipatingShards) {
        continue;
      }
      targetCoordInDim =
          llvm::mod(targetCoordInDim, *cyclicParticipatingShards);
    } else if (targetCoordInDim < 0 || targetCoordInDim >= totalShardsInDim) {
      continue;
    }

    SmallVector<int64_t> nextCoords = coords;
    for (const AxisShiftInfo& info : llvm::reverse(axisInfos)) {
      int64_t subCoord = (coords[info.meshIdx] / info.postSize) % info.subSize;
      int64_t targetSubCoord = targetCoordInDim % info.subSize;
      targetCoordInDim /= info.subSize;
      nextCoords[info.meshIdx] += (targetSubCoord - subCoord) * info.postSize;
    }

    int64_t nextFlatIndex =
        getLinearIndexFromCoordinates(nextCoords, manualMeshAxes);
    int64_t src = deviceIds.empty() ? i : deviceIds[i];
    int64_t tgt = deviceIds.empty() ? nextFlatIndex : deviceIds[nextFlatIndex];
    pairs.push_back(src);
    pairs.push_back(tgt);
  }
  return pairs;
}

// Generates code to right shift elements logically in each local device across
// the partitioning space using HALO exchange and support multi-hop shifts.
Value rightShiftData(Location loc, Value input, int64_t dim, int64_t totalShift,
                     TensorShardingAttr sharding, MeshAttr mesh,
                     const llvm::SmallDenseSet<AxisRefAttr>& manualAxes,
                     ResolutionState& state) {
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t shardSize = inputType.getDimSize(dim);
  TensorShardingAttr localSharding =
      removeAxesFromSharding(sharding, manualAxes, mesh);

  // Parse manual axes for the sharded dimension.
  SmallVector<AxisRefAttr> manualAxesInDim =
      getManualAxesInDim(sharding.getDimShardings()[dim], mesh, manualAxes);

  int64_t hops = totalShift / shardSize;
  int64_t fraction = totalShift % shardSize;

  // Helper to collective permute a sliced piece by a given device offset.
  auto permutePiece = [&](Value piece, int64_t deviceOffset) -> Value {
    if (deviceOffset == 0) {
      return piece;
    }
    SmallVector<int64_t> pairs =
        getDataShiftSourceTargetPairs(mesh, manualAxesInDim, deviceOffset);
    if (pairs.empty()) {
      Value zeroConst = createZeroScalarConstant(
          loc, inputType.getElementType(), state.rewriter);
      auto bcast = stablehlo::BroadcastInDimOp::create(
          state.rewriter, loc, cast<RankedTensorType>(piece.getType()),
          zeroConst, state.rewriter.getDenseI64ArrayAttr({}));
      setSharding(bcast, localSharding);
      return bcast;
    }
    auto pairType =
        RankedTensorType::get({static_cast<int64_t>(pairs.size()) / 2, 2},
                              state.rewriter.getI64Type());
    auto channelAttr = getChannelHandle(state);
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
  Value tailSlice =
      sliceAlongDim(loc, tail, dim, headSliceSize, shardSize, localSharding,
                    state.rewriter, /*skipFullDimSlice=*/false);

  // headSlice slices head[0 : headSliceSize]
  Value headSlice = sliceAlongDim(loc, head, dim, 0, headSliceSize,
                                  localSharding, state.rewriter,
                                  /*skipFullDimSlice=*/false);

  // Concat tailSlice followed by headSlice
  Value concat = stablehlo::ConcatenateOp::create(
      state.rewriter, loc, ValueRange{tailSlice, headSlice}, dim);
  setSharding(concat, localSharding);
  return concat;
}

// Generates the local device code wrapped inside a manual computation to
// implement HALO exchange for shifting data to the right. This effectively
// moves the padding in the operand of a reverse op from high edge to low edge.
Value haloRightShiftData(Location loc, Value input, TensorShardingAttr sharding,
                         MeshAttr mesh,
                         const llvm::SmallDenseSet<AxisRefAttr>& manualAxes,
                         ArrayRef<int64_t> dimsToShift,
                         ArrayRef<int64_t> shiftAmounts,
                         ResolutionState& state) {
  // Compute the local tensor shape for the operand inside manual computation.
  SmallVector<int64_t> localShape;
  localShape.reserve(sharding.getDimShardings().size());
  auto inputType = cast<RankedTensorType>(input.getType());
  for (auto [dim, dimSharding] : llvm::enumerate(sharding.getDimShardings())) {
    localShape.push_back(getLocalDimSize(dimSharding, inputType.getDimSize(dim),
                                         mesh, manualAxes));
  }

  // Get the manual axes for the manual computation.
  SmallVector<AxisRefAttr> manualAxesAttrs =
      getManualAxesAttrs(mesh, manualAxes);

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

// Returns the given input if sourceTargetPairs is empty, otherwise returns a
// collective-permute operation with the given sourceTargetPairs.
Value mayCollectivePermute(Location loc, Value input,
                           ArrayRef<int64_t> sourceTargetPairs,
                           ResolutionState& state) {
  if (sourceTargetPairs.empty()) {
    return input;
  }

  auto pairType = RankedTensorType::get(
      {static_cast<int64_t>(sourceTargetPairs.size()) / 2, 2},
      state.rewriter.getI64Type());
  auto channelAttr = getChannelHandle(state);
  auto permOp = stablehlo::CollectivePermuteOp::create(
      state.rewriter, loc, input.getType(), input,
      DenseIntElementsAttr::get(pairType, sourceTargetPairs), channelAttr);
  return permOp.getResult();
}

// Gathers boundary data from left and right neighboring shards along a single
// sharded dimension, concatenates them with the local shard, and pads the
// accumulated buffer on both sides to prepare for dynamic slicing to assemble
// the HALO exchanged buffer for the dimension.
//
// For an undilated HALO exchange buffer, interiorPad is 0. Otherwise,
// interiorPad is baseDilation - 1. Parameter edgePadSize already considers
// interior padding. Optional callbacks shouldMaskNonTarget and
// sliceSegment allow skipping non-target masking and trimming segment
// bounds prior to collective-permute for a given hop offset.
Value assembleHaloExchangeBuffer(
    Location loc, Value input, int64_t dim, int64_t edgePadSize,
    Value paddingValue, int64_t leftHops, int64_t rightHops,
    TensorShardingAttr sharding, MeshAttr mesh,
    const llvm::SmallDenseSet<AxisRefAttr>& manualAxes,
    TensorShardingAttr localSharding, int64_t interiorPad,
    const DimExchangeInfo& dimExchange, Value idInPartitionGroup,
    llvm::function_ref<bool(int64_t)> shouldMaskNonTarget,
    llvm::function_ref<Value(Value, int64_t)> sliceSegment,
    ResolutionState& state) {
  DimensionShardingAttr activeDimSharded =
      dimExchange.reshapeInfo ? dimExchange.reshapeInfo->getTargetDimSharding()
                              : sharding.getDimShardings()[dim];
  SmallVector<AxisRefAttr> manualAxesInDim =
      getManualAxesInDim(activeDimSharded, mesh, manualAxes);

  SmallVector<Value> concatSegments;
  concatSegments.reserve(leftHops + 1 + rightHops);

  auto getSegmentInput = [&](Value val, int64_t offset) -> Value {
    return sliceSegment ? sliceSegment(val, offset) : val;
  };

  auto fetchNeighbors = [&](int64_t start, int64_t end) {
    for (int64_t offset = start; offset >= end; --offset) {
      SmallVector<int64_t> pairs =
          getDataShiftSourceTargetPairs(mesh, manualAxesInDim, offset);
      Value segInput = getSegmentInput(input, offset);
      Value neighborData = mayCollectivePermute(loc, segInput, pairs, state);
      if (!isFullyReplicated(localSharding)) {
        setSharding(neighborData, localSharding);
      }
      if (paddingValue && idInPartitionGroup && !manualAxesInDim.empty() &&
          (!shouldMaskNonTarget || shouldMaskNonTarget(offset))) {
        // Ensure devices not in target set use 0 instead of garbage values.
        int64_t shardCount = activeDimSharded.getShardedSize(mesh);
        int64_t thresholdVal = offset > 0 ? offset : shardCount + offset;
        Value thresholdConst = createScalarIntConstant(
            loc, state.rewriter.getI32Type(), thresholdVal, state.rewriter);
        Value isTargetPred = stablehlo::CompareOp::create(
            state.rewriter, loc, idInPartitionGroup, thresholdConst,
            offset > 0 ? stablehlo::ComparisonDirection::GE
                       : stablehlo::ComparisonDirection::LT);
        auto segType = cast<RankedTensorType>(neighborData.getType());
        Value broadcastPadding = stablehlo::BroadcastInDimOp::create(
            state.rewriter, loc, segType, paddingValue,
            state.rewriter.getDenseI64ArrayAttr({}));
        Value broadcastPred = stablehlo::BroadcastInDimOp::create(
            state.rewriter, loc,
            RankedTensorType::get(segType.getShape(),
                                  state.rewriter.getI1Type()),
            isTargetPred, state.rewriter.getDenseI64ArrayAttr({}));
        neighborData = stablehlo::SelectOp::create(state.rewriter, loc, segType,
                                                   broadcastPred, neighborData,
                                                   broadcastPadding);
        if (!isFullyReplicated(localSharding)) {
          setSharding(broadcastPadding, localSharding);
          setSharding(broadcastPred, localSharding);
          setSharding(neighborData, localSharding);
        }
      }
      concatSegments.push_back(neighborData);
    }
  };

  // Fetch Left Neighbors (hop count h = leftHops down to 1)
  fetchNeighbors(leftHops, 1);
  // Local Shard
  concatSegments.push_back(getSegmentInput(input, /*offset=*/0));
  // Fetch Right Neighbors (hop count h = 1 up to rightHops, offset maps to -1
  // down to -rightHops)
  fetchNeighbors(-1, -rightHops);

  // Concatenate all segments.
  Value concat = stablehlo::ConcatenateOp::create(state.rewriter, loc,
                                                  concatSegments, dim);
  setSharding(concat, localSharding);

  if (edgePadSize == 0 && interiorPad == 0) {
    return concat;
  }

  // Pad the concatenated tensor on both sides by edgePadSize.
  auto concatType = cast<RankedTensorType>(concat.getType());
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t rank = inputType.getRank();
  SmallVector<int64_t> concatLow(rank, 0), concatHigh(rank, 0),
      concatInterior(rank, 0);
  concatLow[dim] = edgePadSize;
  concatHigh[dim] = edgePadSize;
  concatInterior[dim] = interiorPad;

  SmallVector<int64_t> paddedConcatShape =
      llvm::to_vector(concatType.getShape());
  int64_t dilatedSize =
      (concatType.getShape()[dim] - 1) * (interiorPad + 1) + 1;
  paddedConcatShape[dim] = dilatedSize + 2 * edgePadSize;
  auto paddedConcatType =
      RankedTensorType::get(paddedConcatShape, concatType.getElementType());

  Value paddedConcat = stablehlo::PadOp::create(
      state.rewriter, loc, paddedConcatType, concat, paddingValue,
      state.rewriter.getDenseI64ArrayAttr(concatLow),
      state.rewriter.getDenseI64ArrayAttr(concatHigh),
      state.rewriter.getDenseI64ArrayAttr(concatInterior));
  setSharding(paddedConcat, localSharding);

  return paddedConcat;
}

// Returns a dynamic slice of the halo exchange buffer.
Value dynamicSliceHaloExchangeBuffer(Location loc, Value haloBuffer,
                                     Value offset, Value zeroConst, int64_t dim,
                                     int64_t sliceSize,
                                     TensorShardingAttr localSharding,
                                     ResolutionState& state) {
  auto inputType = cast<RankedTensorType>(haloBuffer.getType());
  SmallVector<int64_t> resultPieceShape = llvm::to_vector(inputType.getShape());
  resultPieceShape[dim] = sliceSize;
  auto resultPieceType =
      RankedTensorType::get(resultPieceShape, inputType.getElementType());

  SmallVector<Value> dynamicOffsets(inputType.getRank(), zeroConst);
  dynamicOffsets[dim] = offset;

  auto dynamicSliceOp = stablehlo::DynamicSliceOp::create(
      state.rewriter, loc, resultPieceType, haloBuffer, dynamicOffsets,
      state.rewriter.getDenseI64ArrayAttr(resultPieceShape));
  setSharding(dynamicSliceOp.getResult(), localSharding);
  return dynamicSliceOp.getResult();
}

// Performs HALO data exchange for a single sharded dimension and slices the
// HALO exchange buffer to retrieve the needed local segment.
//
// For window ops, such as reduce_window and convolution, the HALO exchange
// buffer is not dilated, indicated by dilatedHaloBuffer=false. For pad and
// slice ops, the buffer is dilated by the base dilation of the dimension.
Value exchangeDimWithDynamicOffset(
    Location loc, Value operand, Value paddingValue, int64_t dim,
    const DimExchangeInfo& dimExchange, TensorShardingAttr sharding,
    MeshAttr mesh, const llvm::SmallDenseSet<AxisRefAttr>& manualAxes,
    TensorShardingAttr localSharding, int64_t padLow, int64_t stride,
    int64_t sOut, int64_t baseDilation, bool dilatedHaloBuffer,
    ResolutionState& state) {
  Value partitionId = createDeviceId(loc, state);
  DimensionShardingAttr activeDimShardedAttr =
      dimExchange.reshapeInfo ? dimExchange.reshapeInfo->getTargetDimSharding()
                              : sharding.getDimShardings()[dim];
  Value idInPartitionGroup = convertPartitionIdToIdInGroup(
      loc, partitionId, mesh, activeDimShardedAttr, state.rewriter);

  auto inputType = cast<RankedTensorType>(operand.getType());
  int64_t sIn = inputType.getShape()[dim];
  int64_t sInDilated = sIn * baseDilation;
  int64_t activeSOut = sOut == 0 ? dimExchange.sFootprint : sOut;
  int64_t diffSize = activeSOut * stride - sInDilated;
  int64_t unpaddedBaseOffset = -padLow + dimExchange.leftHops * sInDilated;
  int64_t interiorPad = dilatedHaloBuffer ? baseDilation - 1 : 0;

  int64_t edgePadSize = dilatedHaloBuffer
                            ? dimExchange.sFootprint * baseDilation
                            : dimExchange.sFootprint;
  int64_t shardCount = activeDimShardedAttr.getShardedSize(mesh);
  int64_t endOffset = (shardCount - 1) * diffSize + unpaddedBaseOffset;
  int64_t minUnpaddedOff = std::min(unpaddedBaseOffset, endOffset);
  int64_t maxUnpaddedOff = std::max(unpaddedBaseOffset, endOffset);
  int64_t concatDimSize =
      (dimExchange.leftHops + 1 + dimExchange.rightHops) * sIn;
  int64_t dilatedConcatSize = (concatDimSize - 1) * (interiorPad + 1) + 1;
  // Skip edge padding when every shard's slice window [offset, offset +
  // dilatedFootprint] lies entirely within the concatenated segments [0,
  // dilatedConcatSize].
  if (!dimExchange.reshapeInfo) {
    int64_t dilatedFootprint = dilatedHaloBuffer
                                   ? dimExchange.sFootprint
                                   : dimExchange.sFootprint * baseDilation;
    if (minUnpaddedOff >= 0 &&
        maxUnpaddedOff + dilatedFootprint <= dilatedConcatSize) {
      edgePadSize = 0;
    }
  }
  int64_t bufferDimSize =
      (dilatedHaloBuffer ? dilatedConcatSize : concatDimSize) + 2 * edgePadSize;
  // When there is no edge or interior padding and baseDilation == 1, if the
  // union of all shards' slice windows starts inside the first segment and ends
  // inside the last segment, we can trim the unused prefix of the first segment
  // and unused suffix of the last segment before collective-permute.
  int64_t totalSegments = dimExchange.leftHops + 1 + dimExchange.rightHops;
  int64_t maxEnd = maxUnpaddedOff + dimExchange.sFootprint;
  bool canTrimSegments =
      !dimExchange.reshapeInfo && edgePadSize == 0 && interiorPad == 0 &&
      baseDilation == 1 && totalSegments > 1 && minUnpaddedOff >= 0 &&
      minUnpaddedOff < sIn && maxEnd > (totalSegments - 1) * sIn &&
      maxEnd <= totalSegments * sIn;
  int64_t untrimmedOffsetInConcat =
      unpaddedBaseOffset + edgePadSize * (dilatedHaloBuffer ? 1 : baseDilation);
  // Returns true if any non-target shard for hop offset has a dynamic slice
  // window overlapping that hop's segment [segStart, segEnd) in the buffer.
  auto shouldMaskNonTarget = [&](int64_t offset) -> bool {
    if (dimExchange.reshapeInfo) {
      return true;
    }
    int64_t segStart =
        edgePadSize + (dimExchange.leftHops - offset) * sIn * (interiorPad + 1);
    int64_t segEnd =
        edgePadSize +
        (dimExchange.leftHops - offset + 1) * sIn * (interiorPad + 1) -
        interiorPad;
    int64_t nonTargetStart =
        offset > 0 ? 0 : std::max<int64_t>(0, shardCount + offset);
    int64_t nonTargetEnd =
        offset > 0 ? std::min(offset, shardCount) : shardCount;
    for (int64_t t = nonTargetStart; t < nonTargetEnd; ++t) {
      int64_t rawDilatedOff =
          std::max<int64_t>(t * diffSize + untrimmedOffsetInConcat, 0);
      int64_t sliceStart =
          dilatedHaloBuffer ? rawDilatedOff : rawDilatedOff / baseDilation;
      sliceStart = std::min(sliceStart, bufferDimSize - dimExchange.sFootprint);
      int64_t sliceEnd = sliceStart + dimExchange.sFootprint;
      if (sliceStart < segEnd && sliceEnd > segStart) {
        return true;
      }
    }
    return false;
  };
  auto sliceSegment = [&](Value val, int64_t offset) -> Value {
    if (!canTrimSegments) {
      return val;
    }
    int64_t k = dimExchange.leftHops - offset;
    int64_t sliceStart = (k == 0) ? minUnpaddedOff : 0;
    int64_t sliceLimit = (k == totalSegments - 1) ? (maxEnd - k * sIn) : sIn;
    return sliceAlongDim(loc, val, dim, sliceStart, sliceLimit, localSharding,
                         state.rewriter);
  };
  Value paddedConcat = assembleHaloExchangeBuffer(
      loc, operand, dim, edgePadSize, paddingValue, dimExchange.leftHops,
      dimExchange.rightHops, sharding, mesh, manualAxes, localSharding,
      interiorPad, dimExchange, idInPartitionGroup, shouldMaskNonTarget,
      sliceSegment, state);
  int64_t bufferBaseOffset =
      untrimmedOffsetInConcat - (canTrimSegments ? minUnpaddedOff : 0);

  Value offset;
  Value zeroConst;
  if (dimExchange.reshapeInfo) {
    auto i32Ty = state.rewriter.getI32Type();
    SDY_CHECK(dilatedHaloBuffer);
    SDY_CHECK_EQ(baseDilation, 1);
    SmallVector<int32_t> offsets(dimExchange.shardExchanges.size(), 0);
    for (int64_t t = 0; t < dimExchange.shardExchanges.size(); ++t) {
      const auto& ex = dimExchange.shardExchanges[t];
      int64_t edgePad = dimExchange.sFootprint;
      int64_t sourceId = ex.sourceId != -1 ? ex.sourceId : t;
      int64_t relativeHop = dimExchange.leftHops + sourceId - t;
      int64_t off = -padLow + edgePad + relativeHop * sIn + ex.localOffset;
      SDY_CHECK_GE(off, std::numeric_limits<int32_t>::min());
      SDY_CHECK_LE(off, std::numeric_limits<int32_t>::max());
      offsets[t] = static_cast<int32_t>(off);
    }
    int64_t n = offsets.size();
    auto offsetsConst = stablehlo::ConstantOp::create(
        state.rewriter, loc,
        DenseIntElementsAttr::get(RankedTensorType::get({n}, i32Ty), offsets));
    auto sliceOp = stablehlo::DynamicSliceOp::create(
        state.rewriter, loc, RankedTensorType::get({1}, i32Ty), offsetsConst,
        idInPartitionGroup, state.rewriter.getDenseI64ArrayAttr({1}));
    offset = stablehlo::ReshapeOp::create(
        state.rewriter, loc, RankedTensorType::get({}, i32Ty), sliceOp);
    zeroConst = createZeroConstant(loc, RankedTensorType::get({}, i32Ty),
                                   state.rewriter);
  } else {
    int64_t minOffsetInBuffer = std::min(
        bufferBaseOffset, (shardCount - 1) * diffSize + bufferBaseOffset);
    DeviceOffsetInfo offsetInfo =
        getDeviceOffsetInfo(loc, idInPartitionGroup, diffSize, bufferBaseOffset,
                            baseDilation, state,
                            /*needUndilated=*/!dilatedHaloBuffer,
                            /*clampToZero=*/minOffsetInBuffer < 0);
    offset = dilatedHaloBuffer ? offsetInfo.dilatedOffset
                               : offsetInfo.undilatedOffset;
    zeroConst = offsetInfo.zeroConst;
  }

  return dynamicSliceHaloExchangeBuffer(loc, paddedConcat, offset, zeroConst,
                                        dim, dimExchange.sFootprint,
                                        localSharding, state);
}

// Performs HALO exchange for a sharded operand inside a manual computation
// block.
Value exchangeDataWithDynamicOffset(
    Location loc, Value operand, Value paddingValue,
    TensorShardingAttr sharding, MeshAttr mesh,
    const llvm::SmallDenseSet<AxisRefAttr>& manualAxes,
    ArrayRef<DimExchangeInfo> dimExchanges, ArrayRef<int64_t> edgePaddingLow,
    ArrayRef<int64_t> windowStrides, ArrayRef<int64_t> baseDilations,
    bool dilatedHaloBuffer, ResolutionState& state) {
  TensorShardingAttr localSharding =
      removeAxesFromSharding(sharding, manualAxes, mesh);
  int64_t dimExchangeIdx = -1;
  for (int64_t dim = 0; dim < sharding.getDimShardings().size(); ++dim) {
    if (sharding.getDimShardings()[dim].getShardedSize(mesh) <= 1) {
      continue;
    }
    dimExchangeIdx++;
    if (!dimExchanges[dimExchangeIdx].needsComm) {
      continue;
    }

    int64_t stride = windowStrides.empty() ? 1 : windowStrides[dim];
    int64_t sOut = dimExchanges[dimExchangeIdx].sOut;
    int64_t padLow = edgePaddingLow.empty() ? 0 : edgePaddingLow[dim];
    int64_t baseDilation = baseDilations.empty() ? 1 : baseDilations[dim];
    operand = exchangeDimWithDynamicOffset(
        loc, operand, paddingValue, dim, dimExchanges[dimExchangeIdx], sharding,
        mesh, manualAxes, localSharding, padLow, stride, sOut, baseDilation,
        dilatedHaloBuffer, state);
  }
  return operand;
}

// Performs HALO exchange over the sharded input inside a newly created
// sdy.manual_computation block.
//
// Inside the block, it shifts local shards along the sharded dimensions
// specified in dimExchanges to align boundary requirements. It then runs
// the user-provided postProcess callback to process the local result
// before returning the output of the manual computation.
SmallVector<Value> haloDataExchangeVariadic(
    Location loc, ValueRange divisibleInputs,
    ArrayRef<int64_t> divisibleInputShape,
    ArrayRef<RankedTensorType> origInputTypes,
    ArrayRef<RankedTensorType> origOutputTypes, TensorShardingAttr inSharding,
    TensorShardingAttr outSharding, MeshAttr mesh,
    const llvm::SmallDenseSet<AxisRefAttr>& manualAxes,
    ValueRange paddingValues, ArrayRef<int64_t> edgePaddingLow,
    ArrayRef<DimExchangeInfo> dimExchanges, ArrayRef<int64_t> windowStrides,
    ArrayRef<int64_t> baseDilations, const VariadicPostProcessFn& postProcess,
    ResolutionState& state) {
  SmallVector<AxisRefAttr> manualAxesAttrs =
      getManualAxesAttrs(mesh, manualAxes);

  SmallVector<int64_t> localShape;
  for (auto [dim, dimSharding] :
       llvm::enumerate(inSharding.getDimShardings())) {
    localShape.push_back(getLocalDimSize(dimSharding, divisibleInputShape[dim],
                                         mesh, manualAxes));
  }

  SmallVector<Value> operands;
  operands.append(divisibleInputs.begin(), divisibleInputs.end());
  operands.append(paddingValues.begin(), paddingValues.end());

  SmallVector<TensorShardingAttr> inShardings;
  inShardings.reserve(operands.size());
  inShardings.append(divisibleInputs.size(), inSharding);
  inShardings.append(
      paddingValues.size(),
      TensorShardingAttr::getFullyReplicated(state.rewriter.getContext(), 0,
                                             inSharding.getMeshOrRef(),
                                             /*isClosed=*/false));

  SmallVector<TensorShardingAttr> outShardings(origOutputTypes.size(),
                                               outSharding);

  SmallVector<Type> resultTypes;
  SmallVector<RankedTensorType> paddedGlobalResultTypes;
  for (RankedTensorType origOutputType : origOutputTypes) {
    SmallVector<int64_t> paddedGlobalResultShape;
    paddedGlobalResultShape.reserve(origOutputType.getRank());
    for (auto [i, dimSize] : llvm::enumerate(origOutputType.getShape())) {
      paddedGlobalResultShape.push_back(getPaddedDimSize(
          dimSize, outSharding.getDimShardings()[i].getShardedSize(mesh)));
    }
    RankedTensorType paddedGlobalResultType = RankedTensorType::get(
        paddedGlobalResultShape, origOutputType.getElementType());
    paddedGlobalResultTypes.push_back(paddedGlobalResultType);
    resultTypes.push_back(paddedGlobalResultType);
  }

  auto manualComp = ManualComputationOp::create(
      state.rewriter, loc, resultTypes, operands,
      TensorShardingPerValueAttr::get(state.rewriter.getContext(), inShardings),
      TensorShardingPerValueAttr::get(state.rewriter.getContext(),
                                      outShardings),
      manualAxesAttrs);

  Region& body = manualComp.getBody();
  body.emplaceBlock();
  SmallVector<Value> localInputs;
  for (RankedTensorType inputType : origInputTypes) {
    localInputs.push_back(body.addArgument(
        RankedTensorType::get(localShape, inputType.getElementType()), loc));
  }
  SmallVector<Value> localPaddingValues;
  for (Value paddingValue : paddingValues) {
    localPaddingValues.push_back(body.addArgument(paddingValue.getType(), loc));
  }

  OpBuilder::InsertionGuard guard(state.rewriter);
  state.rewriter.setInsertionPointToStart(&body.front());

  SmallVector<Value> exchangedLocals;
  for (auto [localInput, localPadVal] :
       llvm::zip_equal(localInputs, localPaddingValues)) {
    Value exchangedLocal = exchangeDataWithDynamicOffset(
        loc, localInput, localPadVal, inSharding, mesh, manualAxes,
        dimExchanges, edgePaddingLow, windowStrides, baseDilations,
        /*dilatedHaloBuffer=*/windowStrides.empty(), state);
    exchangedLocals.push_back(exchangedLocal);
  }

  SmallVector<Value> localResults = postProcess(
      exchangedLocals, localPaddingValues, paddedGlobalResultTypes, state);

  ReturnOp::create(state.rewriter, loc, localResults);
  return SmallVector<Value>(manualComp.getResults().begin(),
                            manualComp.getResults().end());
}

Value haloDataExchange(
    Location loc, Value divisibleInput, ArrayRef<int64_t> divisibleInputShape,
    RankedTensorType origInputType, RankedTensorType origOutputType,
    TensorShardingAttr inSharding, TensorShardingAttr outSharding,
    MeshAttr mesh, const llvm::SmallDenseSet<AxisRefAttr>& manualAxes,
    Value paddingValue, ArrayRef<int64_t> edgePaddingLow,
    ArrayRef<DimExchangeInfo> dimExchanges, ArrayRef<int64_t> windowStrides,
    ArrayRef<int64_t> baseDilations, const PostProcessFn& postProcess,
    ResolutionState& state) {
  auto variadicPostProcess = [&](ValueRange exchangedLocals,
                                 ValueRange localPaddingValues,
                                 ArrayRef<RankedTensorType> resultTypes,
                                 ResolutionState& state) -> SmallVector<Value> {
    return {postProcess(exchangedLocals[0], localPaddingValues[0],
                        resultTypes[0], state)};
  };
  SmallVector<Value> results = haloDataExchangeVariadic(
      loc, {divisibleInput}, divisibleInputShape, {origInputType},
      {origOutputType}, inSharding, outSharding, mesh, manualAxes,
      {paddingValue}, edgePaddingLow, dimExchanges, windowStrides,
      baseDilations, variadicPostProcess, state);
  return results[0];
}

// Returns the data exchange info for a single sharded dimension.
DimExchangeInfo getDimExchangeInfo(int64_t shardCount, int64_t sIn,
                                   int64_t sOut, int64_t sFootprint,
                                   int64_t padLow, int64_t stride,
                                   int64_t baseDilation, int64_t inputDimSize) {
  DimExchangeInfo dimInfo;
  dimInfo.sFootprint = sFootprint;
  dimInfo.sOut = sOut;

  bool dimNeedsComm = false;
  for (int64_t t = 0; t < shardCount; ++t) {
    // In windowed-ops, the input is dilated and then padded, meaning the stride
    // and window parameters are defined in this processed input space. In
    // contrast, the derived sFootprint is defined in the un-dilated (physical)
    // input space. For this reason, we map "start" and "limit" to the original
    // input space to find out which shards of the input are needed.
    int64_t start =
        llvm::divideFloorSigned(t * sOut * stride - padLow, baseDilation);
    int64_t limit = start + sFootprint - 1;

    int64_t validStart = std::max<int64_t>(0, start);
    int64_t validLimit = std::min<int64_t>(inputDimSize - 1, limit);

    int64_t sourceId = -1;
    int64_t localOffset = 0;
    int64_t lastSourceId = -1;

    if (validStart <= validLimit) {
      sourceId = validStart / sIn;
      localOffset = start - sourceId * sIn;
      lastSourceId = validLimit / sIn;
    }

    if ((sourceId != -1 && sourceId != t) ||
        (lastSourceId != -1 && lastSourceId != t) || sourceId == -1) {
      dimNeedsComm = true;
    }
    if (sourceId != -1 && sourceId < t) {
      dimInfo.leftHops = std::max(dimInfo.leftHops, t - sourceId);
    }
    if (lastSourceId != -1 && lastSourceId > t) {
      dimInfo.rightHops = std::max(dimInfo.rightHops, lastSourceId - t);
    }
    dimInfo.shardExchanges.push_back({sourceId, localOffset});
  }

  dimInfo.needsComm = dimNeedsComm;
  if (dimInfo.leftHops + dimInfo.rightHops + 1 >= shardCount &&
      (dimInfo.leftHops > 1 || dimInfo.rightHops > 1)) {
    dimInfo.canUseHalo = false;
  }
  return dimInfo;
}

// Checks if a dimension is sharded, populates the manual axes, pads the logical
// shape in DataExchangeInfo, and returns the sharding context.
std::optional<ShardedDimContext> prepareShardedDimForExchange(
    int64_t dim, DimensionShardingAttr dimSharding, MeshAttr mesh,
    DataExchangeInfo& info) {
  int64_t shardCount = dimSharding.getShardedSize(mesh);
  if (shardCount <= 1) {
    return std::nullopt;
  }
  for (AxisRefAttr axis : dimSharding.getAxes()) {
    info.manualAxes.insert(axis);
  }
  info.divisibleInputShape[dim] =
      getPaddedDimSize(info.divisibleInputShape[dim], shardCount);
  int64_t sIn = info.divisibleInputShape[dim] / shardCount;
  return ShardedDimContext{sIn, shardCount};
}

// Master loop builder that walks through all sharded dimensions of an op and
// constructs the DataExchangeInfo.
//
// `processDim` is a callable with signature
//   DimExchangeInfo(int64_t dim, int64_t sIn, int64_t shardCount)
// and handles the explicit dimension exchange logic for each axis for a given
// op.
template <typename F>
DataExchangeInfo buildDataExchangeInfo(TensorShardingAttr sharding,
                                       MeshAttr mesh,
                                       RankedTensorType inputType,
                                       F&& processDim) {
  DataExchangeInfo info;
  info.divisibleInputShape = llvm::to_vector(inputType.getShape());

  for (auto [dim, dimSharding] : llvm::enumerate(sharding.getDimShardings())) {
    std::optional<ShardedDimContext> context =
        prepareShardedDimForExchange(dim, dimSharding, mesh, info);
    if (!context) {
      // Dimension is not sharded.
      continue;
    }
    DimExchangeInfo dimInfo =
        processDim(dim, context->sIn, context->shardCount);
    info.dimExchanges.push_back(dimInfo);
  }
  return info;
}

// =============================================================================
// Implementation of handleXYZOps routines in alphabetical order.
// =============================================================================

// -----------------------------------------------------------------------------
// stablehlo.concatenate
// -----------------------------------------------------------------------------

// Returns the right-rotation amount if concatOp matches the rotate-right
// pattern concatenate(slice(x, [k:N]), slice(x, [0:k])) along a sharded
// dimension where both slices have a single use and share the sharding of x
// and concatOp.
std::optional<int64_t> findRotateRightPattern(stablehlo::ConcatenateOp concatOp,
                                              const SymbolTable& symbolTable) {
  if (concatOp.getInputs().size() != 2) {
    return std::nullopt;
  }
  auto lhs = concatOp.getInputs()[0].getDefiningOp<stablehlo::SliceOp>();
  auto rhs = concatOp.getInputs()[1].getDefiningOp<stablehlo::SliceOp>();
  if (!lhs || !rhs || !lhs->hasOneUse() || !rhs->hasOneUse() ||
      lhs.getOperand() != rhs.getOperand()) {
    return std::nullopt;
  }
  Value toRotate = lhs.getOperand();
  auto origType = dyn_cast<RankedTensorType>(toRotate.getType());
  if (!origType || origType != concatOp.getType()) {
    return std::nullopt;
  }
  TensorShardingAttr sharding = getSharding(concatOp.getResult());
  if (isFullyReplicated(sharding) || !sharding.getUnreducedAxes().empty() ||
      getSharding(toRotate) != sharding ||
      getSharding(lhs.getResult()) != sharding ||
      getSharding(rhs.getResult()) != sharding) {
    return std::nullopt;
  }
  MeshAttr mesh = sharding.getMesh(symbolTable);
  int64_t dim = concatOp.getDimension();
  if (!mesh || mesh.isSingleDevice() ||
      sharding.getDimShardings()[dim].getShardedSize(mesh) <= 1) {
    return std::nullopt;
  }
  ArrayRef<int64_t> lhsStarts = lhs.getStartIndices();
  ArrayRef<int64_t> lhsLimits = lhs.getLimitIndices();
  ArrayRef<int64_t> lhsStrides = lhs.getStrides();
  ArrayRef<int64_t> rhsStarts = rhs.getStartIndices();
  ArrayRef<int64_t> rhsLimits = rhs.getLimitIndices();
  ArrayRef<int64_t> rhsStrides = rhs.getStrides();
  for (int64_t d = 0; d < origType.getRank(); ++d) {
    if (lhsStrides[d] != 1 || rhsStrides[d] != 1) {
      return std::nullopt;
    }
    int64_t dimSize = origType.getDimSize(d);
    if (d == dim) {
      if (rhsStarts[d] != 0 || rhsLimits[d] != lhsStarts[d] ||
          lhsLimits[d] != dimSize) {
        return std::nullopt;
      }
    } else if (lhsStarts[d] != 0 || lhsLimits[d] != dimSize ||
               rhsStarts[d] != 0 || rhsLimits[d] != dimSize) {
      return std::nullopt;
    }
  }
  int64_t amount = lhs.getType().getDimSize(dim);
  int64_t fullSize = origType.getDimSize(dim);
  if (amount <= 0 || amount >= fullSize) {
    return std::nullopt;
  }
  return amount;
}

// Rotates the first `participatingShards` shards of `localInput` right by
// `rotateAmount` elements along `dim` using cyclic collective-permutes.
Value rotatePaddedShardRight(Location loc, Value localInput, int64_t dim,
                             int64_t rotateAmount, int64_t participatingShards,
                             MeshAttr mesh,
                             ArrayRef<AxisRefAttr> manualAxesInDim,
                             TensorShardingAttr localSharding,
                             ResolutionState& state) {
  auto localType = cast<RankedTensorType>(localInput.getType());
  ArrayRef<int64_t> localShape = localType.getShape();
  int64_t shardSize = localShape[dim];
  rotateAmount %= (shardSize * participatingShards);
  SDY_CHECK_GT(rotateAmount, 0);

  IRRewriter& rewriter = state.rewriter;
  int64_t currentSize = 0;
  SmallVector<Value> concatPieces;
  while (currentSize < shardSize) {
    int64_t shardDistance =
        llvm::divideCeil(rotateAmount - currentSize, shardSize);
    int64_t offsetInShard =
        shardDistance * shardSize - rotateAmount + currentSize;
    int64_t haloSize =
        std::min(shardSize - offsetInShard, shardSize - currentSize);
    currentSize += haloSize;

    Value halo =
        sliceAlongDim(loc, localInput, dim, offsetInShard,
                      offsetInShard + haloSize, localSharding, rewriter);
    int64_t effectiveDistance = shardDistance % participatingShards;
    if (effectiveDistance != 0) {
      SmallVector<int64_t> pairs = getDataShiftSourceTargetPairs(
          mesh, manualAxesInDim, effectiveDistance, participatingShards);
      halo = mayCollectivePermute(loc, halo, pairs, state);
      setSharding(halo, localSharding);
    }
    concatPieces.push_back(halo);
  }
  if (concatPieces.size() == 1) {
    return concatPieces[0];
  }
  Value concat =
      stablehlo::ConcatenateOp::create(rewriter, loc, concatPieces, dim);
  setSharding(concat, localSharding);
  return concat;
}

// When `rightPadding > 0`, the `participatingShards` shards hold
// `participatingShards * shardSize = fullSize + rightPadding` elements, with
// `rightPadding` padding elements at the end of the last participating shard
// `[fullSize : fullSize + rightPadding)`. In this padded cyclic buffer:
// - The `[0:k]` suffix (`k = fullSize - amount`), which lands at global output
//   indices `[amount:fullSize)`, does not wrap around the end of the buffer and
//   therefore shifts right by `amount` (`rotated0`).
// - The `[k:fullSize]` prefix, which wraps around the end of the original
//   tensor to land at global output indices `[0:amount)`, must also jump over
//   the `rightPadding` trailing padding elements and therefore shifts right by
//   `amount + rightPadding` (`rotated1`).
// On each shard with global start offset `shardOffset = shardId * shardSize`,
// local index `i` corresponds to global output index `shardOffset + i`, so
// `iota < amount - shardOffset` selects `rotated1` for `[0:amount)` and
// `rotated0` for `[amount:fullSize)`.
Value selectRotatedWithRightPadding(Location loc, Value rotated0,
                                    Value rotated1, int64_t dim, int64_t amount,
                                    DimensionShardingAttr dimSharding,
                                    MeshAttr mesh,
                                    TensorShardingAttr localSharding,
                                    ResolutionState& state) {
  IRRewriter& rewriter = state.rewriter;
  auto localType = cast<RankedTensorType>(rotated0.getType());
  ArrayRef<int64_t> localShape = localType.getShape();
  int64_t shardSize = localShape[dim];

  Value idInPartitionGroup = convertPartitionIdToIdInGroup(
      loc, createDeviceId(loc, state), mesh, dimSharding, rewriter);
  Type i32Ty = rewriter.getI32Type();
  Value shardSizeConst =
      createScalarIntConstant(loc, i32Ty, shardSize, rewriter);
  Value shardOffset = stablehlo::MulOp::create(
      rewriter, loc, idInPartitionGroup, shardSizeConst);
  Value amountConst = createScalarIntConstant(loc, i32Ty, amount, rewriter);
  Value boundary =
      stablehlo::SubtractOp::create(rewriter, loc, amountConst, shardOffset);
  auto localI32Type = RankedTensorType::get(localShape, i32Ty);
  Value bcastBoundary = stablehlo::BroadcastInDimOp::create(
      rewriter, loc, localI32Type, boundary, rewriter.getDenseI64ArrayAttr({}));
  setSharding(bcastBoundary, localSharding);
  Value iota = stablehlo::IotaOp::create(rewriter, loc, localI32Type,
                                         rewriter.getI64IntegerAttr(dim));
  setSharding(iota, localSharding);
  Value pred = stablehlo::CompareOp::create(rewriter, loc, iota, bcastBoundary,
                                            stablehlo::ComparisonDirection::LT);
  setSharding(pred, localSharding);
  Value selected = stablehlo::SelectOp::create(rewriter, loc, localType, pred,
                                               rotated1, rotated0);
  setSharding(selected, localSharding);
  return selected;
}

// Implements rotate-right concatenate(slice(x, [k:N]), slice(x, [0:k])) along a
// sharded dimension using cyclic collective-permutes:
//
// 1. Align logical input shard sizes via high-side padding.
// 2. Inside `sdy.manual_computation`, cyclically shift the
//    `participatingShards` shards right by `amount` (`rotated0`).
// 3. If `rightPadding > 0`, also cyclically shift right by
//    `amount + rightPadding` (`rotated1`) to skip the trailing padding gap for
//    wrapped-around elements, and select between `rotated1` and `rotated0` at
//    global boundary `amount`.
// 4. Trim high-side padding to match the original tensor shape.
LogicalResult handleConcatenateOp(stablehlo::ConcatenateOp concatOp,
                                  ResolutionState& state) {
  std::optional<int64_t> amountOpt =
      findRotateRightPattern(concatOp, state.symbolTable);
  if (!amountOpt) {
    return failure();
  }
  int64_t amount = *amountOpt;
  int64_t dim = concatOp.getDimension();
  auto lhs = concatOp.getInputs()[0].getDefiningOp<stablehlo::SliceOp>();
  auto rhs = concatOp.getInputs()[1].getDefiningOp<stablehlo::SliceOp>();
  Value toRotate = lhs.getOperand();
  auto origType = cast<RankedTensorType>(toRotate.getType());
  TensorShardingAttr sharding = getSharding(concatOp.getResult());
  MeshAttr mesh = sharding.getMesh(state.symbolTable);
  int64_t rank = origType.getRank();
  int64_t fullSize = origType.getDimSize(dim);
  int64_t shardCount = sharding.getDimShardings()[dim].getShardedSize(mesh);
  int64_t shardSize = llvm::divideCeil(fullSize, shardCount);
  int64_t participatingShards = llvm::divideCeil(fullSize, shardSize);
  int64_t rightPadding = participatingShards * shardSize - fullSize;

  DataExchangeInfo exchangeInfo;
  exchangeInfo.divisibleInputShape = llvm::to_vector(origType.getShape());
  SmallVector<int64_t> localShape;
  localShape.reserve(rank);
  for (auto [d, dimSharding] : llvm::enumerate(sharding.getDimShardings())) {
    if (std::optional<ShardedDimContext> ctx =
            prepareShardedDimForExchange(d, dimSharding, mesh, exchangeInfo)) {
      localShape.push_back(ctx->sIn);
    } else {
      localShape.push_back(exchangeInfo.divisibleInputShape[d]);
    }
  }

  IRRewriter& rewriter = state.rewriter;
  Location loc = concatOp.getLoc();
  rewriter.setInsertionPoint(concatOp);
  Value divisibleInput = padHighSideToShape(
      loc, toRotate, exchangeInfo.divisibleInputShape, sharding, rewriter);
  SmallVector<AxisRefAttr> manualAxesAttrs =
      getManualAxesAttrs(mesh, exchangeInfo.manualAxes);
  TensorShardingAttr localSharding =
      removeAxesFromSharding(sharding, exchangeInfo.manualAxes, mesh);
  SmallVector<AxisRefAttr> manualAxesInDim = getManualAxesInDim(
      sharding.getDimShardings()[dim], mesh, exchangeInfo.manualAxes);

  auto manualComp = ManualComputationOp::create(
      rewriter, loc, divisibleInput.getType(), divisibleInput, {sharding},
      {sharding}, manualAxesAttrs);
  Region& body = manualComp.getBody();
  body.emplaceBlock();
  auto localType = RankedTensorType::get(localShape, origType.getElementType());
  Value localInput = body.addArgument(localType, loc);

  rewriter.setInsertionPointToStart(&body.front());
  Value localResult =
      rotatePaddedShardRight(loc, localInput, dim, amount, participatingShards,
                             mesh, manualAxesInDim, localSharding, state);
  if (rightPadding > 0) {
    Value rotatedWithPadding = rotatePaddedShardRight(
        loc, localInput, dim, amount + rightPadding, participatingShards, mesh,
        manualAxesInDim, localSharding, state);
    localResult = selectRotatedWithRightPadding(
        loc, localResult, rotatedWithPadding, dim, amount,
        sharding.getDimShardings()[dim], mesh, localSharding, state);
  }
  ReturnOp::create(rewriter, loc, localResult);

  rewriter.setInsertionPoint(concatOp);
  Value result = sliceHighSideToType(loc, manualComp.getResult(0), origType,
                                     sharding, rewriter);
  rewriter.replaceOp(concatOp, result);
  state.opsToErase.push_back(lhs.getOperation());
  state.opsToErase.push_back(rhs.getOperation());
  return success();
}

// -----------------------------------------------------------------------------
// stablehlo.convolution
// -----------------------------------------------------------------------------

struct ShardedConvSpatialDimInfo {
  int64_t spatialIdx;
  int64_t lhsDim;
  int64_t outDim;
  int64_t padLow;
  int64_t padHigh;
  int64_t stride;
  int64_t sOut;
  int64_t sFootprint;
};

// Lowers a spatially sharded `stablehlo.convolution` by all-gathering the
// sharded spatial dimensions of the LHS, then padding, dynamically slicing each
// device's `[id * sOut * stride : +W]` window footprint, and running a
// zero-padded local convolution inside an `sdy.manual_computation`, matching
// GSPMD's fallback in `PartitionConvolutionBaseCase`.
LogicalResult handleConvolutionOp(stablehlo::ConvolutionOp convOp,
                                  ResolutionState& state) {
  if (convOp.getBatchGroupCount() != 1 || convOp.getFeatureGroupCount() != 1 ||
      convOp->getParentOfType<ManualComputationOp>()) {
    return failure();
  }

  Value lhs = convOp.getLhs();
  Value rhs = convOp.getRhs();
  Value out = convOp.getResult();
  TensorShardingAttr lhsSharding = getSharding(lhs);
  TensorShardingAttr rhsSharding = getSharding(rhs);
  TensorShardingAttr outSharding = getSharding(out);
  if (!lhsSharding || !rhsSharding || !outSharding ||
      (isFullyReplicated(lhsSharding) && isFullyReplicated(outSharding))) {
    return failure();
  }
  if (!lhsSharding.getUnreducedAxes().empty() ||
      !rhsSharding.getUnreducedAxes().empty() ||
      !outSharding.getUnreducedAxes().empty()) {
    return failure();
  }

  MeshAttr mesh = lhsSharding.getMesh(state.symbolTable);
  if (!mesh || mesh.isSingleDevice()) {
    return failure();
  }

  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  auto outType = dyn_cast<RankedTensorType>(out.getType());
  if (!lhsType || !rhsType || !outType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || !outType.hasStaticShape()) {
    return failure();
  }

  auto isEvenlyDivisible = [&](RankedTensorType type,
                               TensorShardingAttr sharding) {
    for (const auto& [dimSize, dimSharding] :
         llvm::zip_equal(type.getShape(), sharding.getDimShardings())) {
      int64_t shardCount = dimSharding.getShardedSize(mesh);
      if (dimSize <= 0 || dimSize % shardCount != 0) {
        return false;
      }
    }
    return true;
  };
  if (!isEvenlyDivisible(lhsType, lhsSharding) ||
      !isEvenlyDivisible(rhsType, rhsSharding) ||
      !isEvenlyDivisible(outType, outSharding)) {
    return failure();
  }

  stablehlo::ConvDimensionNumbersAttr dimNums = convOp.getDimensionNumbers();
  if (lhsSharding.getDimShardings()[dimNums.getInputBatchDimension()]
              .getAxes() !=
          outSharding.getDimShardings()[dimNums.getOutputBatchDimension()]
              .getAxes() ||
      lhsSharding.getDimShardings()[dimNums.getInputFeatureDimension()]
              .getAxes() !=
          rhsSharding
              .getDimShardings()[dimNums.getKernelInputFeatureDimension()]
              .getAxes() ||
      rhsSharding.getDimShardings()[dimNums.getKernelOutputFeatureDimension()]
              .getAxes() !=
          outSharding.getDimShardings()[dimNums.getOutputFeatureDimension()]
              .getAxes()) {
    return failure();
  }

  SmallVector<AxisRefAttr> rhsAxes;
  rhsSharding.forEachAxisRef(
      [&](AxisRefAttr axis) { rhsAxes.push_back(axis); });

  ArrayRef<int64_t> inputSpatialDims = dimNums.getInputSpatialDimensions();
  ArrayRef<int64_t> kernelSpatialDims = dimNums.getKernelSpatialDimensions();
  ArrayRef<int64_t> outputSpatialDims = dimNums.getOutputSpatialDimensions();
  std::optional<ArrayRef<bool>> windowReversal = convOp.getWindowReversal();
  int64_t numSpatialDims = inputSpatialDims.size();
  SmallVector<int64_t> paddingValues(2 * numSpatialDims, 0);
  if (auto paddingAttr = convOp.getPadding()) {
    llvm::copy(paddingAttr->getValues<int64_t>(), paddingValues.begin());
  }

  SmallVector<ShardedConvSpatialDimInfo> shardedSpatialDims;
  llvm::SmallDenseSet<AxisRefAttr> manualAxes;
  for (auto [i, dims] : llvm::enumerate(llvm::zip_equal(
           inputSpatialDims, kernelSpatialDims, outputSpatialDims))) {
    auto [lhsDim, rhsDim, outDim] = dims;
    DimensionShardingAttr lhsDimSharding =
        lhsSharding.getDimShardings()[lhsDim];
    DimensionShardingAttr rhsDimSharding =
        rhsSharding.getDimShardings()[rhsDim];
    DimensionShardingAttr outDimSharding =
        outSharding.getDimShardings()[outDim];
    int64_t lhsShardCount = lhsDimSharding.getShardedSize(mesh);
    int64_t rhsShardCount = rhsDimSharding.getShardedSize(mesh);
    int64_t outShardCount = outDimSharding.getShardedSize(mesh);
    if (lhsShardCount == 1 && rhsShardCount == 1 && outShardCount == 1) {
      continue;
    }
    if (rhsShardCount != 1 || lhsShardCount <= 1 ||
        lhsDimSharding.getAxes() != outDimSharding.getAxes()) {
      return failure();
    }
    for (AxisRefAttr axis : lhsDimSharding.getAxes()) {
      if (axis.getSubAxisInfo() ||
          llvm::any_of(rhsAxes, [&](AxisRefAttr rhsAxis) {
            return axis.overlaps(rhsAxis);
          })) {
        return failure();
      }
    }
    int64_t spatialIdx = static_cast<int64_t>(i);
    if ((windowReversal && (*windowReversal)[spatialIdx]) ||
        getDimProperty(convOp.getLhsDilation(), spatialIdx, 1) != 1) {
      return failure();
    }
    int64_t stride = getDimProperty(convOp.getWindowStrides(), spatialIdx, 1);
    int64_t rhsDilation =
        getDimProperty(convOp.getRhsDilation(), spatialIdx, 1);
    int64_t windowSize = rhsType.getDimSize(rhsDim);
    int64_t effectiveWindowSize = (windowSize - 1) * rhsDilation + 1;
    int64_t padLow = paddingValues[2 * spatialIdx];
    int64_t padHigh = paddingValues[2 * spatialIdx + 1];
    if (padLow < 0 || padHigh < 0) {
      return failure();
    }
    int64_t sIn = lhsType.getDimSize(lhsDim) / lhsShardCount;
    int64_t sOut = outType.getDimSize(outDim) / outShardCount;
    int64_t sFootprint = (sOut - 1) * stride + effectiveWindowSize;
    bool needsComm =
        padLow > 0 || padHigh > 0 || sFootprint != sIn || stride != 1;
    // Only apply the optimization when all sharded spatial dims require
    // communication, assuming typical convolutions have spatial dims that are
    // either all communication-free or all require communication.
    if (!needsComm) {
      return failure();
    }
    for (AxisRefAttr axis : lhsDimSharding.getAxes()) {
      manualAxes.insert(axis);
    }
    shardedSpatialDims.push_back({spatialIdx, lhsDim, outDim, padLow, padHigh,
                                  stride, sOut, sFootprint});
  }

  if (shardedSpatialDims.empty()) {
    return failure();
  }

  TensorShardingAttr localLhsSharding =
      removeAxesFromSharding(lhsSharding, manualAxes, mesh);
  TensorShardingAttr localOutSharding =
      removeAxesFromSharding(outSharding, manualAxes, mesh);
  // If replicating the spatial dimensions matches both the input of a
  // single-use LHS reshard and a single-use result reshard, fall back to
  // replication so CanonicalizerPass(kReshardLabel) folds both reshards.
  auto lhsReshard = lhs.getDefiningOp<ReshardOp>();
  auto userReshard = convOp->hasOneUse()
                         ? dyn_cast<ReshardOp>(*convOp->user_begin())
                         : nullptr;
  if (lhsReshard && lhsReshard->hasOneUse() && userReshard &&
      getSharding(lhsReshard.getInput()) == localLhsSharding &&
      userReshard.getSharding() == localOutSharding) {
    return failure();
  }

  IRRewriter& rewriter = state.rewriter;
  Location loc = convOp.getLoc();
  rewriter.setInsertionPoint(convOp);
  SmallVector<AxisRefListAttr> gatheringAxesPerDim(
      lhsType.getRank(), AxisRefListAttr::get(rewriter.getContext(), {}));
  for (const ShardedConvSpatialDimInfo& d : shardedSpatialDims) {
    gatheringAxesPerDim[d.lhsDim] =
        AxisRefListAttr::get(rewriter.getContext(),
                             lhsSharding.getDimShardings()[d.lhsDim].getAxes());
  }
  Value gatheredLhs = sdy::AllGatherOp::create(
      rewriter, loc, lhs, gatheringAxesPerDim, localLhsSharding);

  SmallVector<AxisRefAttr> manualAxesAttrs =
      getManualAxesAttrs(mesh, manualAxes);
  auto manualComp = ManualComputationOp::create(
      rewriter, loc, TypeRange{outType}, ValueRange{gatheredLhs, rhs},
      TensorShardingPerValueAttr::get(rewriter.getContext(),
                                      {localLhsSharding, rhsSharding}),
      TensorShardingPerValueAttr::get(rewriter.getContext(), {outSharding}),
      manualAxesAttrs);

  Region& body = manualComp.getBody();
  body.emplaceBlock();
  Value blockLhs = body.addArgument(lhsType, loc);
  Value blockRhs = body.addArgument(rhsType, loc);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&body.front());

  int64_t rank = lhsType.getRank();
  SmallVector<int64_t> edgePaddingLow(rank, 0);
  SmallVector<int64_t> edgePaddingHigh(rank, 0);
  SmallVector<int64_t> interiorPadding(rank, 0);
  SmallVector<int64_t> paddedLhsShape(lhsType.getShape());
  bool hasSpatialPadding = false;
  for (const ShardedConvSpatialDimInfo& d : shardedSpatialDims) {
    edgePaddingLow[d.lhsDim] = d.padLow;
    edgePaddingHigh[d.lhsDim] = d.padHigh;
    paddedLhsShape[d.lhsDim] += d.padLow + d.padHigh;
    if (d.padLow > 0 || d.padHigh > 0) {
      hasSpatialPadding = true;
    }
  }

  Value paddedLhs = blockLhs;
  if (hasSpatialPadding) {
    Value zeroPad =
        createZeroScalarConstant(loc, lhsType.getElementType(), rewriter);
    auto paddedLhsType =
        RankedTensorType::get(paddedLhsShape, lhsType.getElementType());
    paddedLhs = stablehlo::PadOp::create(
        rewriter, loc, paddedLhsType, blockLhs, zeroPad,
        rewriter.getDenseI64ArrayAttr(edgePaddingLow),
        rewriter.getDenseI64ArrayAttr(edgePaddingHigh),
        rewriter.getDenseI64ArrayAttr(interiorPadding));
    setSharding(paddedLhs, localLhsSharding);
  }

  Value partitionId = createDeviceId(loc, state);
  int64_t totalDevices = mesh.getTotalSize();
  Type i32Ty = rewriter.getI32Type();
  Value zeroIdx =
      createZeroConstant(loc, RankedTensorType::get({}, i32Ty), rewriter);
  SmallVector<Value> startIndices(rank, zeroIdx);
  SmallVector<int64_t> sliceSizes(paddedLhsShape);
  for (const ShardedConvSpatialDimInfo& d : shardedSpatialDims) {
    int64_t step = d.sOut * d.stride;
    SmallVector<int32_t> offsetsTable;
    offsetsTable.reserve(totalDevices);
    for (int64_t devId = 0; devId < totalDevices; ++devId) {
      int64_t offset =
          getShardIndex(devId, mesh, lhsSharding.getDimShardings()[d.lhsDim]) *
          step;
      SDY_CHECK_LE(offset, std::numeric_limits<int32_t>::max());
      offsetsTable.push_back(static_cast<int32_t>(offset));
    }
    auto offsetsConst = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseIntElementsAttr::get(RankedTensorType::get({totalDevices}, i32Ty),
                                  offsetsTable));
    auto sliceOp = stablehlo::DynamicSliceOp::create(
        rewriter, loc, RankedTensorType::get({1}, i32Ty), offsetsConst,
        partitionId, rewriter.getDenseI64ArrayAttr({1}));
    startIndices[d.lhsDim] = stablehlo::ReshapeOp::create(
        rewriter, loc, RankedTensorType::get({}, i32Ty), sliceOp);
    sliceSizes[d.lhsDim] = d.sFootprint;
  }

  auto slicedLhsType =
      RankedTensorType::get(sliceSizes, lhsType.getElementType());
  Value slicedLhs = stablehlo::DynamicSliceOp::create(
      rewriter, loc, slicedLhsType, paddedLhs, startIndices,
      rewriter.getDenseI64ArrayAttr(sliceSizes));
  setSharding(slicedLhs, localLhsSharding);

  for (const ShardedConvSpatialDimInfo& d : shardedSpatialDims) {
    paddingValues[2 * d.spatialIdx] = 0;
    paddingValues[2 * d.spatialIdx + 1] = 0;
  }
  auto newPaddingAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({numSpatialDims, 2}, rewriter.getI64Type()),
      paddingValues);

  SmallVector<int64_t> localOutShape(outType.getShape());
  for (const ShardedConvSpatialDimInfo& d : shardedSpatialDims) {
    localOutShape[d.outDim] = d.sOut;
  }
  auto localOutType =
      RankedTensorType::get(localOutShape, outType.getElementType());

  auto localConv = stablehlo::ConvolutionOp::create(
      rewriter, loc, localOutType, slicedLhs, blockRhs,
      convOp.getWindowStridesAttr(), newPaddingAttr,
      convOp.getLhsDilationAttr(), convOp.getRhsDilationAttr(),
      convOp.getWindowReversalAttr(), dimNums, convOp.getFeatureGroupCount(),
      convOp.getBatchGroupCount(), convOp.getPrecisionConfigAttr());
  setSharding(localConv.getResult(), localOutSharding);

  ReturnOp::create(rewriter, loc, localConv.getResult());
  rewriter.replaceOp(convOp, manualComp.getResult(0));
  return success();
}

// -----------------------------------------------------------------------------
// stablehlo.pad
// -----------------------------------------------------------------------------

// Returns a DataExchangeInfo to represent the needed data exchange for the
// given pad op.
DataExchangeInfo getPadDataExchangeInfo(stablehlo::PadOp padOp,
                                        TensorShardingAttr sharding,
                                        MeshAttr mesh,
                                        RankedTensorType inputType) {
  return buildDataExchangeInfo(
      sharding, mesh, inputType,
      [&](int64_t dim, int64_t sIn, int64_t shardCount) {
        int64_t sOut =
            llvm::divideCeil(padOp.getType().getDimSize(dim), shardCount);
        int64_t pLow = padOp.getEdgePaddingLow()[dim];
        int64_t baseDilation = padOp.getInteriorPadding()[dim] + 1;
        DimExchangeInfo dimInfo = getDimExchangeInfo(shardCount, sIn, sOut,
                                                     /*sFootprint=*/sOut, pLow,
                                                     /*stride=*/1, baseDilation,
                                                     inputType.getDimSize(dim));
        if (!isCommunicationFreePadDim(dim, padOp, sharding, mesh)) {
          // This is a zero-hop-non-communication-free case, which doesn't
          // require collective-permute but dynamic-slice.
          dimInfo.needsComm = true;
        }
        return dimInfo;
      });
}

// Implements the non-trivial padding operation on replicated dimensions.
Value handleReplicatedPadDims(
    Location loc, Value exchangedLocal, Value localPaddingValue,
    RankedTensorType paddedGlobalType, TensorShardingAttr sharding,
    MeshAttr mesh, const llvm::SmallDenseSet<AxisRefAttr>& manualAxes,
    ArrayRef<int64_t> edgePaddingLow, ArrayRef<int64_t> edgePaddingHigh,
    ArrayRef<int64_t> interiorPadding, ResolutionState& state) {
  auto exchangedLocalType = cast<RankedTensorType>(exchangedLocal.getType());
  int64_t rank = exchangedLocalType.getRank();
  TensorShardingAttr localSharding =
      removeAxesFromSharding(sharding, manualAxes, mesh);

  SmallVector<int64_t> localResultShape(rank, 0), localLow(rank, 0),
      localHigh(rank, 0), localInterior(rank, 0);
  bool needsLocalPad = false;
  for (int64_t i = 0; i < rank; ++i) {
    int64_t sOutLocal =
        getLocalDimSize(sharding.getDimShardings()[i],
                        paddedGlobalType.getDimSize(i), mesh, manualAxes);
    localResultShape[i] = sOutLocal;

    if (sharding.getDimShardings()[i].getShardedSize(mesh) > 1) {
      continue;
    }

    int64_t low = edgePaddingLow[i];
    int64_t interior = interiorPadding[i];
    int64_t high = edgePaddingHigh[i];
    localLow[i] = low;
    localInterior[i] = interior;
    localHigh[i] = high;
    int64_t sInLocal = exchangedLocalType.getShape()[i];
    int64_t expandedInputSize =
        (sInLocal - 1) * (interior + 1) + 1 + low + high;
    SDY_CHECK(expandedInputSize == sOutLocal);
    if (low != 0 || high != 0 || interior != 0) {
      needsLocalPad = true;
    }
  }

  if (!needsLocalPad) {
    return exchangedLocal;
  }

  auto localResultType = RankedTensorType::get(
      localResultShape, paddedGlobalType.getElementType());
  exchangedLocal = stablehlo::PadOp::create(
      state.rewriter, loc, localResultType, exchangedLocal, localPaddingValue,
      state.rewriter.getDenseI64ArrayAttr(localLow),
      state.rewriter.getDenseI64ArrayAttr(localHigh),
      state.rewriter.getDenseI64ArrayAttr(localInterior));
  setSharding(exchangedLocal, localSharding);
  return exchangedLocal;
}

// Implements a sharded pad operation using HALO exchange for non-uniform
// dimensions as follows:
//
// 1. Align logical input shard sizes via high-side padding.
// 2. Use HALO exchange to implement sharded padding dimensions:
//    - Concatenate the left neighbor shard, self shard, and right neighbor
//      shard along the sharded dimension.
//    - Pad the concatenated buffer by sOut.
//    - Slice out a local segment of shape sOut using the dynamic offset:
//        offset = max(partitionId * diffSize + baseOffset, 0)
//        where diffSize = sOut - sIn,
//              baseOffset = -padLow + sIn + sOut.
// 3. Implement replicated padding dimensions.
// 4. Trim final tensor shape to match expected output shape.
LogicalResult handlePadOp(stablehlo::PadOp padOp, ResolutionState& state) {
  Value origInput = padOp.getOperand();
  TensorShardingAttr inSharding = getSharding(origInput);
  auto origInputType = mlir::dyn_cast<RankedTensorType>(origInput.getType());
  if (isFullyReplicated(inSharding) || !origInputType) {
    return success();
  }

  SymbolTable& symbolTable = state.symbolTable;
  MeshAttr mesh = inSharding.getMesh(symbolTable);
  if (!mesh || mesh.isSingleDevice()) {
    return success();
  }

  DataExchangeInfo info =
      getPadDataExchangeInfo(padOp, inSharding, mesh, origInputType);
  if (info.manualAxes.empty() || !info.needsComm()) {
    return success();
  }
  if (!canUseHalo(info)) {
    return failure();
  }

  Location loc = padOp.getLoc();
  IRRewriter& rewriter = state.rewriter;
  rewriter.setInsertionPoint(padOp);
  // Align logical shard sizes via high-side padding (Comm-free).
  Value divisibleInput =
      padHighSideToShape(loc, origInput, info.divisibleInputShape, inSharding,
                         rewriter, padOp.getPaddingValue());

  SmallVector<int64_t> baseDilations = llvm::map_to_vector(
      padOp.getInteriorPadding(), [](int64_t val) { return val + 1; });

  // Define step 3 as post processing inside haloDataExchange.
  auto postProcess = [&](Value exchangedLocal, Value localPaddingValue,
                         RankedTensorType paddedGlobalResultType,
                         ResolutionState& state) -> Value {
    return handleReplicatedPadDims(
        loc, exchangedLocal, localPaddingValue, paddedGlobalResultType,
        inSharding, mesh, info.manualAxes, padOp.getEdgePaddingLow(),
        padOp.getEdgePaddingHigh(), padOp.getInteriorPadding(), state);
  };

  Value result = haloDataExchange(
      loc, divisibleInput, info.divisibleInputShape, origInputType,
      padOp.getType(), inSharding, inSharding, mesh, info.manualAxes,
      padOp.getPaddingValue(), padOp.getEdgePaddingLow(), info.dimExchanges,
      /*windowStrides=*/{}, baseDilations, postProcess, state);

  // Trim final tensor shape to match expected output shape.
  result =
      sliceHighSideToType(loc, result, padOp.getType(), inSharding, rewriter);

  rewriter.replaceOp(padOp, result);
  return success();
}

// -----------------------------------------------------------------------------
// stablehlo.reshape
// -----------------------------------------------------------------------------

struct ReshapeHaloInfo {
  SmallVector<ReshapeGroupInfo> activeGroups;
};

struct ReshapeShardingAnalysis {
  bool isEquivalent = false;
  bool isCommunicationFree = false;
  llvm::SmallDenseSet<AxisRefAttr> nonDivisibleAxes;
  SmallVector<int64_t> paddedInputShape;
  SmallVector<int64_t> paddedOutputShape;
  SmallVector<ReshapeGroupInfo> reshapeGroups;
  std::optional<ReshapeHaloInfo> haloInfo;

  static ReshapeShardingAnalysis run(stablehlo::ReshapeOp reshapeOp,
                                     TensorShardingAttr inSharding,
                                     TensorShardingAttr outSharding,
                                     MeshAttr mesh,
                                     RankedTensorType origInputType,
                                     RankedTensorType resultType,
                                     bool enableHaloExchange);
};

// Returns ReshapeHaloInfo for the given reshape groups for HALO exchange or
// std::nullopt if HALO exchange cannot be performed.
std::optional<ReshapeHaloInfo> getReshapeHaloInfo(
    ArrayRef<ReshapeGroupInfo> reshapeGroups, RankedTensorType inputType,
    RankedTensorType resultType, ArrayRef<int64_t> paddedInputShape,
    ArrayRef<int64_t> paddedOutputShape) {
  if (reshapeGroups.empty()) {
    return std::nullopt;
  }

  ReshapeHaloInfo info;
  // Gather the ReshapeGroup for HALO exchange, may return early when there is
  // indivisible group that can't be handle by HALO exchange.
  for (const ReshapeGroupInfo& g : reshapeGroups) {
    if (g.isPassthrough()) {
      continue;
    }
    if (!g.hasIndivisibility(paddedInputShape, paddedOutputShape, inputType,
                             resultType)) {
      continue;
    }
    if (g.isNeither()) {
      return std::nullopt;
    }

    // Fall back to replication if an active group with indivisibility contains
    // size-1 dimensions. We can support these cases through removing dimensions
    // 1 from the reshape with indivisibility and then reshape to add them back.
    if (g.numInNontrivialDims != (g.inLastDim - g.inStartDim) ||
        g.numOutNontrivialDims != (g.outLastDim - g.outStartDim)) {
      return std::nullopt;
    }

    int64_t inSize = inputType.getDimSize(g.inStartDim);
    int64_t outSize = resultType.getDimSize(g.outStartDim);
    if (inSize % outSize != 0 && outSize % inSize != 0) {
      return std::nullopt;
    }

    if (g.getInVolume(paddedInputShape) != g.getOutVolume(paddedOutputShape)) {
      return std::nullopt;
    }

    if (!g.isSplit()) {
      return std::nullopt;
    }

    info.activeGroups.push_back(g);
  }

  if (info.activeGroups.empty()) {
    return std::nullopt;
  }

  return info;
}

// Returns a ReshapeShardingAnalysis for the given reshape operation.
//
// The analysis determines if the reshape operation can be handled by
// shardy's HALO exchange mechanism or if it must fall back to replication.
// It also computes the padded output and input shapes and the reshape groups
// for the given reshape operation.
ReshapeShardingAnalysis ReshapeShardingAnalysis::run(
    stablehlo::ReshapeOp reshapeOp, TensorShardingAttr inSharding,
    TensorShardingAttr outSharding, MeshAttr mesh,
    RankedTensorType origInputType, RankedTensorType resultType,
    bool enableHaloExchange) {
  ReshapeShardingAnalysis a;
  a.reshapeGroups = buildReshapeGroupInfos(origInputType, resultType);
  a.isEquivalent = isShardingEquivalentAcrossReshapes(
      inSharding, origInputType, outSharding, resultType, reshapeOp,
      /*allowNonDivisible=*/true);
  if (!a.isEquivalent) {
    return a;
  }

  a.isCommunicationFree =
      isCommunicationFreeReshape(reshapeOp, inSharding, outSharding, mesh,
                                 origInputType, resultType, a.reshapeGroups);
  if (a.isCommunicationFree) {
    return a;
  }

  a.paddedOutputShape = llvm::to_vector(resultType.getShape());
  a.paddedInputShape = llvm::to_vector(origInputType.getShape());

  // Inspect each ReshapeGroup to see HALO can handle it. If there is any group
  // that cannot be handled with HALO, we fall back to replication for the
  // entire reshape operation.
  for (ReshapeGroupInfo& g : a.reshapeGroups) {
    bool groupHasIndivisibility = false;

    auto processDims = [&](int64_t startDim, int64_t lastDim,
                           TensorShardingAttr sharding, RankedTensorType type,
                           SmallVectorImpl<int64_t>& paddedShape) {
      for (int64_t d = startDim; d < lastDim; ++d) {
        DimensionShardingAttr dimSharding = sharding.getDimShardings()[d];
        int64_t sc = dimSharding.getShardedSize(mesh);
        if (sc > 1 && type.getDimSize(d) % sc != 0) {
          groupHasIndivisibility = true;
          paddedShape[d] = getPaddedDimSize(type.getDimSize(d), sc);
        }
      }
    };

    processDims(g.inStartDim, g.inLastDim, inSharding, origInputType,
                a.paddedInputShape);
    processDims(g.outStartDim, g.outLastDim, outSharding, resultType,
                a.paddedOutputShape);

    if (groupHasIndivisibility) {
      for (int64_t d = g.inStartDim; d < g.inLastDim; ++d) {
        for (AxisRefAttr axis : inSharding.getDimShardings()[d].getAxes()) {
          a.nonDivisibleAxes.insert(axis);
        }
      }
    }
  }

  if (a.nonDivisibleAxes.empty() || !enableHaloExchange) {
    return a;
  }

  a.haloInfo = getReshapeHaloInfo(a.reshapeGroups, origInputType, resultType,
                                  a.paddedInputShape, a.paddedOutputShape);
  return a;
}

DimExchangeInfo getDimExchangeInfoForReshapeGroup(
    stablehlo::ReshapeOp reshapeOp, TensorShardingAttr inSharding,
    TensorShardingAttr outSharding, MeshAttr mesh,
    const ReshapeGroupInfo& groupInfo, int64_t activeDim, int64_t shardCount,
    ArrayRef<int64_t> divisibleInputShape) {
  auto origInputType = reshapeOp.getOperand().getType();
  auto resultType = reshapeOp.getResult().getType();

  int64_t sOut = divisibleInputShape[activeDim] / shardCount;
  ReshapeDimExchangeInfo reshapeInfo;
  SDY_CHECK_EQ(groupInfo.inLastDim - groupInfo.inStartDim, 1);
  auto groupOutDims = llvm::to_vector(
      llvm::seq<int64_t>(groupInfo.outStartDim, groupInfo.outLastDim));
  SmallVector<AxisRefAttr> combinedAxes;
  for (int64_t outDim : groupOutDims) {
    llvm::append_range(combinedAxes,
                       outSharding.getDimShardings()[outDim].getAxes());
  }
  reshapeInfo.targetDimSharding =
      DimensionShardingAttr::get(reshapeOp.getContext(), combinedAxes, {});

  // Pre-calculate row-major strides of the output dimensions in groupOutDims.
  SmallVector<int64_t> groupOutStrides(groupOutDims.size(), 1);
  int64_t stride = 1;
  for (int64_t idx = static_cast<int64_t>(groupOutDims.size()) - 1; idx >= 0;
       --idx) {
    groupOutStrides[idx] = stride;
    stride *= resultType.getDimSize(groupOutDims[idx]);
  }

  // The linear start offset for each source device.
  SmallVector<int64_t> deviceStart(shardCount, 0);
  // The valid elements count for each source device.
  SmallVector<int64_t> deviceCount(shardCount, 0);
  int64_t dimSize = origInputType.getDimSize(activeDim);
  int64_t tileSize = llvm::divideCeil(dimSize, shardCount);
  for (int64_t s = 0; s < shardCount; ++s) {
    deviceStart[s] = s * tileSize;
    deviceCount[s] =
        std::max<int64_t>(0, std::min(tileSize, dimSize - s * tileSize));
  }

  DimExchangeInfo dimInfo;
  dimInfo.sFootprint = sOut;
  dimInfo.sOut = sOut;
  bool dimNeedsComm = false;

  // For each target shard compute ShardExchangeInfo.
  for (int64_t t = 0; t < shardCount; ++t) {
    int64_t sourceId = -1;
    int64_t localOffset = 0;
    int64_t lastSourceId = -1;

    int64_t targetStart = getLinearDataStartOffsetForShardIdx(
        t, groupOutDims, groupOutStrides, outSharding, resultType, mesh);
    int64_t targetLimit = targetStart + sOut - 1;

    for (int64_t s = 0; s < shardCount; ++s) {
      if (deviceCount[s] > 0) {
        if (targetStart >= deviceStart[s] &&
            targetStart < deviceStart[s] + deviceCount[s]) {
          sourceId = s;
          localOffset = targetStart - deviceStart[s];
        }
        if (targetLimit >= deviceStart[s] &&
            targetLimit < deviceStart[s] + deviceCount[s]) {
          lastSourceId = s;
        }
      }
    }

    if ((sourceId != -1 && sourceId != t) ||
        (lastSourceId != -1 && lastSourceId != t)) {
      dimNeedsComm = true;
    }
    if (sourceId != -1 && sourceId < t) {
      dimInfo.leftHops = std::max(dimInfo.leftHops, t - sourceId);
    }
    if (lastSourceId != -1 && lastSourceId > t) {
      dimInfo.rightHops = std::max(dimInfo.rightHops, lastSourceId - t);
    }
    dimInfo.shardExchanges.push_back({sourceId, localOffset});
  }

  dimInfo.needsComm = dimNeedsComm;
  // Only support a hops for reshape.
  if (dimInfo.leftHops > 1 || dimInfo.rightHops > 1) {
    dimInfo.canUseHalo = false;
  }
  dimInfo.reshapeInfo = reshapeInfo;
  return dimInfo;
}

// Computes DataExchangeInfo for a reshape operation.
DataExchangeInfo getReshapeDataExchangeInfo(
    stablehlo::ReshapeOp reshapeOp, TensorShardingAttr inSharding,
    TensorShardingAttr outSharding, MeshAttr mesh,
    ArrayRef<int64_t> divisibleInputShape, const ReshapeHaloInfo& haloInfo,
    const llvm::SmallDenseSet<AxisRefAttr>& nonDivisibleAxes) {
  DataExchangeInfo info;
  info.manualAxes = nonDivisibleAxes;
  auto origInputType = reshapeOp.getOperand().getType();
  int64_t activeRank = origInputType.getRank();
  TensorShardingAttr activeSharding = inSharding;

  const auto* activeGroupIt = haloInfo.activeGroups.begin();
  for (int64_t i = 0; i < activeRank; ++i) {
    int64_t sc = activeSharding.getDimShardings()[i].getShardedSize(mesh);
    if (sc <= 1) {
      continue;
    }

    if (activeGroupIt != haloInfo.activeGroups.end() &&
        activeGroupIt->getInIndivisibleDim() == i) {
      const ReshapeGroupInfo& groupInfo = *activeGroupIt++;
      DimExchangeInfo dimInfo = getDimExchangeInfoForReshapeGroup(
          reshapeOp, inSharding, outSharding, mesh, groupInfo, i, sc,
          divisibleInputShape);
      info.dimExchanges.push_back(dimInfo);
    } else {
      int64_t sLocal = divisibleInputShape[i] / sc;
      DimExchangeInfo noCommInfo;
      noCommInfo.sOut = sLocal;
      noCommInfo.sFootprint = sLocal;
      info.dimExchanges.push_back(noCommInfo);
    }
  }
  info.divisibleInputShape = llvm::to_vector(divisibleInputShape);
  return info;
}

// Resolves non-divisible dimensions across a stablehlo.reshape by falling back
// to replication for non-passthrough dimensions.
LogicalResult handleReshapeOpReplicate(stablehlo::ReshapeOp reshapeOp,
                                       TensorShardingAttr inSharding,
                                       TensorShardingAttr outSharding,
                                       RankedTensorType resultType,
                                       ArrayRef<ReshapeGroupInfo> reshapeGroups,
                                       ResolutionState& state) {
  Value operand = reshapeOp.getOperand();
  IRRewriter& rewriter = state.rewriter;
  rewriter.setInsertionPoint(reshapeOp);

  auto origInputType = mlir::cast<RankedTensorType>(operand.getType());
  MeshAttr mesh = inSharding.getMesh(state.symbolTable);

  // Find the sharding axes for which we want to remove indivisibility through
  // replication.
  SmallVector<AxisRefAttr> toRemoveAxisRefs;
  for (const ReshapeGroupInfo& g : reshapeGroups) {
    if (g.isPassthrough()) {
      continue;
    }
    if (g.hasIndivisibility(inSharding, outSharding, mesh, origInputType,
                            resultType)) {
      SmallVector<AxisRefAttr> groupAxes =
          g.getAllAxisRefs(inSharding, outSharding);
      toRemoveAxisRefs.append(groupAxes.begin(), groupAxes.end());
    }
  }

  Location loc = reshapeOp.getLoc();
  TensorShardingAttr newInSharding =
      removeAxisRefsFromSharding(inSharding, toRemoveAxisRefs, mesh);
  Value reshardInput = ReshardOp::create(rewriter, loc, operand.getType(),
                                         operand, newInSharding);

  TensorShardingAttr newOutSharding =
      removeAxisRefsFromSharding(outSharding, toRemoveAxisRefs, mesh);

  rewriter.setInsertionPointAfter(reshapeOp);
  auto newReshapeOp =
      stablehlo::ReshapeOp::create(rewriter, loc, resultType, reshardInput);
  setSharding(newReshapeOp.getResult(), newOutSharding);

  Value restoredResult = ReshardOp::create(
      rewriter, loc, resultType, newReshapeOp.getResult(), outSharding);

  rewriter.replaceOp(reshapeOp, restoredResult);
  return success();
}

// Implements splitting reshape operation with indivisible sharding using HALO
// data exchange.
LogicalResult handleReshapeHaloExchange(
    stablehlo::ReshapeOp reshapeOp, TensorShardingAttr inSharding,
    TensorShardingAttr outSharding, MeshAttr mesh,
    ArrayRef<ReshapeGroupInfo> reshapeGroups,
    ArrayRef<int64_t> divisibleInputShape, ArrayRef<int64_t> paddedOutputShape,
    const ReshapeHaloInfo& haloInfo,
    const llvm::SmallDenseSet<AxisRefAttr>& nonDivisibleAxes,
    ResolutionState& state) {
  SDY_CHECK(!haloInfo.activeGroups.empty());

  DataExchangeInfo info = getReshapeDataExchangeInfo(
      reshapeOp, inSharding, outSharding, mesh, divisibleInputShape, haloInfo,
      nonDivisibleAxes);

  RankedTensorType resultType = reshapeOp.getResult().getType();
  if (!canUseHalo(info)) {
    return handleReshapeOpReplicate(reshapeOp, inSharding, outSharding,
                                    resultType, reshapeGroups, state);
  }

  Location loc = reshapeOp.getLoc();
  IRRewriter& rewriter = state.rewriter;
  rewriter.setInsertionPoint(reshapeOp);
  Value paddingValue =
      createZeroScalarConstant(loc, resultType.getElementType(), rewriter);
  Value operand = reshapeOp.getOperand();
  auto origInputType = cast<RankedTensorType>(operand.getType());
  SmallVector<int64_t> edgePaddingLow(origInputType.getRank(), 0);
  Value paddedInput = padHighSideToShape(loc, operand, divisibleInputShape,
                                         inSharding, rewriter);

  auto postProcess = [&](Value exchangedLocal, Value localPaddingValue,
                         RankedTensorType paddedGlobalResultType,
                         ResolutionState& state) -> Value {
    // Reshape the local value after data exchange to the output shape.
    SmallVector<int64_t> localOutputShape;
    for (auto [dim, dimSharding] :
         llvm::enumerate(outSharding.getDimShardings())) {
      localOutputShape.push_back(getLocalDimSize(
          dimSharding, paddedOutputShape[dim], mesh, info.manualAxes));
    }
    auto localOutputType =
        RankedTensorType::get(localOutputShape, resultType.getElementType());
    TensorShardingAttr localOutSharding =
        removeAxesFromSharding(outSharding, info.manualAxes, mesh);
    auto localReshape = stablehlo::ReshapeOp::create(
        state.rewriter, loc, localOutputType, exchangedLocal);
    setSharding(localReshape.getResult(), localOutSharding);
    return localReshape;
  };

  Value result = haloDataExchange(
      loc, paddedInput, divisibleInputShape, origInputType, resultType,
      inSharding, outSharding, mesh, info.manualAxes, paddingValue,
      edgePaddingLow, info.dimExchanges, /*windowStrides=*/{},
      /*baseDilations=*/{}, postProcess, state);
  result = sliceHighSideToType(loc, result, resultType, outSharding, rewriter);
  rewriter.replaceOp(reshapeOp, result);
  return success();
}

// Resolves stablehlo.reshape operations across sharded dimensions where the
// dimension size is not evenly divisible by the sharding axis size.
LogicalResult handleReshapeOp(stablehlo::ReshapeOp reshapeOp,
                              bool enableHaloExchange, ResolutionState& state) {
  Value operand = reshapeOp.getOperand();
  TensorShardingAttr inSharding = getSharding(operand);
  TensorShardingAttr outSharding = getSharding(reshapeOp.getResult());
  // Insert-explicit-reshard pass should have made sharding consistent.
  SDY_CHECK(isFullyReplicated(inSharding) == isFullyReplicated(outSharding));
  if (isFullyReplicated(inSharding)) {
    return success();
  }

  auto origInputType = mlir::dyn_cast<RankedTensorType>(operand.getType());
  RankedTensorType resultType = reshapeOp.getResult().getType();
  if (!origInputType || !resultType) {
    return success();
  }

  MeshAttr mesh = inSharding.getMesh(state.symbolTable);
  if (!mesh) {
    return success();
  }

  // Verify the assumption that the input and output shardings have the same
  // ordered normalized axes, guranteed by the insert-explicit-reshard pass.
  auto getOrderedNormalizedAxes = [mesh](TensorShardingAttr sharding) {
    SmallVector<AxisRefAttr> axes;
    for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
      for (AxisRefAttr axis : dimSharding.getAxes()) {
        addAxisOrMerge(axes, axis, mesh);
      }
    }
    return axes;
  };
  SDY_CHECK(getOrderedNormalizedAxes(inSharding) ==
            getOrderedNormalizedAxes(outSharding));

  ReshapeShardingAnalysis analysis = ReshapeShardingAnalysis::run(
      reshapeOp, inSharding, outSharding, mesh, origInputType, resultType,
      enableHaloExchange);

  if (analysis.isCommunicationFree) {
    return success();
  }

  if (analysis.haloInfo.has_value()) {
    return handleReshapeHaloExchange(
        reshapeOp, inSharding, outSharding, mesh, analysis.reshapeGroups,
        analysis.paddedInputShape, analysis.paddedOutputShape,
        *analysis.haloInfo, analysis.nonDivisibleAxes, state);
  }

  return handleReshapeOpReplicate(reshapeOp, inSharding, outSharding,
                                  resultType, analysis.reshapeGroups, state);
}

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
  llvm::SmallDenseSet<AxisRefAttr> manualAxes;
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
        info.manualAxes.insert(axis);
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
  if (!mesh || mesh.isSingleDevice()) {
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
    input = padHighSideToShape(loc, operand, info->paddedShape, inSharding,
                               state.rewriter);

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
    input = haloRightShiftData(loc, input, inSharding, mesh, info->manualAxes,
                               dimsToShift, shiftAmounts, state);
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
      inSharding.getUnreducedAxes(), inSharding.getReductionOp());
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
    Value slicedResult = sliceHighSideToType(loc, reversedResult, origType,
                                             inSharding, state.rewriter);
    state.rewriter.replaceAllUsesExcept(reverseOp.getResult(), slicedResult,
                                        reversedResult.getDefiningOp());
  } else {
    state.rewriter.replaceAllUsesExcept(reverseOp.getResult(), reversedResult,
                                        reversedResult.getDefiningOp());
  }

  return success();
}

// -----------------------------------------------------------------------------
// stablehlo.rng_bit_generator
// -----------------------------------------------------------------------------

void handleRngBitGeneratorOp(stablehlo::RngBitGeneratorOp rngOp,
                             bool rngBitGeneratorUnsafe,
                             ResolutionState& state) {
  Type stateElemTy = rngOp.getInitialState().getType().getElementType();
  if (rngBitGeneratorUnsafe &&
      (stateElemTy.isInteger(32) || stateElemTy.isInteger(64))) {
    // Support partitioned rng op output.
    return;
  }
  TensorShardingAttr outSharding = getSharding(rngOp.getOutput());
  if (!outSharding || isFullyReplicated(outSharding)) {
    return;
  }
  // Otherwise, replicate the rng op output.
  setSharding(rngOp.getOutput(), TensorShardingAttr::getFullyClosed(
                                     rngOp.getContext(), outSharding.getRank(),
                                     outSharding.getMeshOrRef()));
  state.rewriter.setInsertionPointAfter(rngOp);
  auto reshardOp = sdy::ReshardOp::create(state.rewriter, rngOp.getLoc(),
                                          rngOp.getOutput(), outSharding);
  state.rewriter.replaceAllUsesExcept(rngOp.getOutput(), reshardOp.getResult(),
                                      reshardOp);
}

// -----------------------------------------------------------------------------
// stablehlo.slice
// -----------------------------------------------------------------------------

DataExchangeInfo getSliceDataExchangeInfo(stablehlo::SliceOp sliceOp,
                                          TensorShardingAttr sharding,
                                          MeshAttr mesh,
                                          RankedTensorType inputType) {
  return buildDataExchangeInfo(
      sharding, mesh, inputType,
      [&](int64_t dim, int64_t sIn, int64_t shardCount) {
        if (isCommunicationFreeSliceDim(dim, sliceOp, sharding, mesh)) {
          return DimExchangeInfo{};
        }
        int64_t sliceSize = sliceOp.getType().getDimSize(dim);
        int64_t sOut = llvm::divideCeil(sliceSize, shardCount);
        int64_t start = sliceOp.getStartIndices()[dim];
        DimExchangeInfo dimInfo = getDimExchangeInfo(
            shardCount, sIn, sOut,
            /*sFootprint=*/sOut, /*padLow=*/-start,
            /*stride=*/1, /*baseDilation=*/1, inputType.getDimSize(dim));
        // This is a zero-hop-non-communication-free case, which doesn't
        // require collective-permute but dynamic-slice.
        dimInfo.needsComm = true;
        return dimInfo;
      });
}

// Implements the slicing operation on replicated dimensions.
Value handleReplicatedSliceDims(
    Location loc, Value exchangedLocal, TensorShardingAttr sharding,
    MeshAttr mesh, const llvm::SmallDenseSet<AxisRefAttr>& manualAxes,
    ArrayRef<int64_t> startIndices, ArrayRef<int64_t> limitIndices,
    ArrayRef<int64_t> strides, ResolutionState& state) {
  auto exchangedLocalType = cast<RankedTensorType>(exchangedLocal.getType());
  int64_t rank = exchangedLocalType.getRank();
  TensorShardingAttr localSharding =
      removeAxesFromSharding(sharding, manualAxes, mesh);

  SmallVector<int64_t> localSliceStarts(rank, 0);
  SmallVector<int64_t> localSliceLimits =
      llvm::to_vector(exchangedLocalType.getShape());
  SmallVector<int64_t> localSliceStrides(rank, 1);
  bool needsLocalSlice = false;
  for (int64_t i = 0; i < rank; ++i) {
    if (sharding.getDimShardings()[i].getShardedSize(mesh) > 1) {
      continue;
    }
    int64_t start = startIndices[i];
    int64_t limit = limitIndices[i];
    int64_t stride = strides.empty() ? 1 : strides[i];
    if (start != 0 || limit != exchangedLocalType.getDimSize(i) ||
        stride != 1) {
      localSliceStarts[i] = start;
      localSliceLimits[i] = limit;
      localSliceStrides[i] = stride;
      needsLocalSlice = true;
    }
  }

  if (!needsLocalSlice) {
    return exchangedLocal;
  }

  SmallVector<int64_t> localResultShape =
      llvm::to_vector(exchangedLocalType.getShape());
  for (int64_t i = 0; i < rank; ++i) {
    localResultShape[i] = llvm::divideCeil(
        localSliceLimits[i] - localSliceStarts[i], localSliceStrides[i]);
  }

  auto localResultType = RankedTensorType::get(
      localResultShape, exchangedLocalType.getElementType());
  exchangedLocal = stablehlo::SliceOp::create(
      state.rewriter, loc, localResultType, exchangedLocal,
      state.rewriter.getDenseI64ArrayAttr(localSliceStarts),
      state.rewriter.getDenseI64ArrayAttr(localSliceLimits),
      state.rewriter.getDenseI64ArrayAttr(localSliceStrides));
  setSharding(exchangedLocal, localSharding);
  return exchangedLocal;
}

// Implements a sharded slice operation using HALO exchange for non-uniform
// dimensions as follows:
//
// 1. Align logical input shard sizes via high-side padding.
// 2. Use HALO exchange to implement sharded slice dimensions:
//    - Concatenate the left neighbor shard, self shard, and right neighbor
//      shard along the sharded dimension.
//    - Pad the concatenated buffer by sOut.
//    - Slice out a local segment of shape sOut using a dynamic offset computed
//      on-device via partitionId.
// 3. Implement replicated slice dimensions:
// 4. Trim final tensor shape to match expected output shape.
LogicalResult handleSliceOp(stablehlo::SliceOp sliceOp,
                            ResolutionState& state) {
  IRRewriter& rewriter = state.rewriter;
  SymbolTable& symbolTable = state.symbolTable;

  Value origInput = sliceOp.getOperand();
  TensorShardingAttr inSharding = getSharding(origInput);
  auto origInputType = mlir::dyn_cast<RankedTensorType>(origInput.getType());
  if (isFullyReplicated(inSharding) || !origInputType) {
    return success();
  }

  MeshAttr mesh = inSharding.getMesh(symbolTable);
  if (!mesh || mesh.isSingleDevice()) {
    return success();
  }

  DataExchangeInfo info =
      getSliceDataExchangeInfo(sliceOp, inSharding, mesh, origInputType);
  if (info.manualAxes.empty() || !info.needsComm()) {
    return success();
  }
  if (!canUseHalo(info)) {
    return failure();
  }

  Location loc = sliceOp.getLoc();
  rewriter.setInsertionPoint(sliceOp);
  // Align logical shard sizes via high-side padding.
  Value divisibleInput =
      padHighSideToShape(loc, origInput, info.divisibleInputShape, inSharding,
                         rewriter, /*paddingValue=*/nullptr);

  Value paddingValue =
      createZeroScalarConstant(loc, origInputType.getElementType(), rewriter);

  int64_t rank = origInputType.getRank();
  SmallVector<int64_t> edgePaddingLow;
  edgePaddingLow.reserve(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    // Unsharded dimensions are sliced in postProcess
    // (handleReplicatedSliceDims) and do not participate in HALO exchange
    // padding.
    if (inSharding.getDimShardings()[dim].getShardedSize(mesh) <= 1) {
      edgePaddingLow.push_back(0);
      continue;
    }
    int64_t start = sliceOp.getStartIndices()[dim];
    edgePaddingLow.push_back(-start);
  }

  auto postProcess = [&](Value exchangedLocal, Value /*localPaddingValue*/,
                         RankedTensorType /*unused paddedGlobalResultType*/,
                         ResolutionState& state) -> Value {
    return handleReplicatedSliceDims(loc, exchangedLocal, inSharding, mesh,
                                     info.manualAxes, sliceOp.getStartIndices(),
                                     sliceOp.getLimitIndices(),
                                     sliceOp.getStrides(), state);
  };

  Value result = haloDataExchange(
      loc, divisibleInput, info.divisibleInputShape, origInputType,
      sliceOp.getType(), inSharding, inSharding, mesh, info.manualAxes,
      paddingValue, edgePaddingLow, info.dimExchanges, /*windowStrides=*/{},
      /*baseDilations=*/{}, postProcess, state);

  // Trim final tensor shape to match expected output shape.
  result =
      sliceHighSideToType(loc, result, sliceOp.getType(), inSharding, rewriter);

  rewriter.replaceOp(sliceOp, result);
  return success();
}

// =============================================================================
// Replication fallback helpers.
// =============================================================================

// Returns true if the convolution spatial dimension `spatialDimIdx` is a pure
// reduction dimension without sliding-window shifts (i.e., output spatial size
// `numWindows[spatialDimIdx]` is 1, padding is 0, window stride is 1, and
// lhs/rhs dilations are 1). When false, sharding this spatial reduction
// dimension also requires permutation (halo exchange) across shards.
bool isPureSpatialReductionDim(stablehlo::ConvolutionOp convOp,
                               int64_t spatialDimIdx,
                               ArrayRef<int64_t> numWindows) {
  return isWindowPassthroughDim(
      convOp.getPadding(), /*windowDimensions=*/numWindows,
      convOp.getWindowStrides().value_or(ArrayRef<int64_t>()), spatialDimIdx,
      convOp.getLhsDilation().value_or(ArrayRef<int64_t>()),
      convOp.getRhsDilation().value_or(ArrayRef<int64_t>()));
}

// Returns true if factorIdx is a Reduction factor with dual kPermutation
// semantics.
bool isReductionFactorWithPermutationSemantics(stablehlo::ConvolutionOp convOp,
                                               OpShardingRuleAttr rule,
                                               int64_t factorIdx) {
  if (rule.getFactorType(factorIdx) != FactorType::kReduction) {
    return false;
  }
  stablehlo::ConvDimensionNumbersAttr dimNums = convOp.getDimensionNumbers();
  RankedTensorType outType = convOp.getType();
  SmallVector<int64_t> numWindows = llvm::map_to_vector(
      dimNums.getOutputSpatialDimensions(),
      [&](int64_t outDim) { return outType.getDimSize(outDim); });
  for (auto [i, dims] :
       llvm::enumerate(llvm::zip_equal(dimNums.getInputSpatialDimensions(),
                                       dimNums.getKernelSpatialDimensions(),
                                       dimNums.getOutputSpatialDimensions()))) {
    auto [lhsDim, rhsDim, outDim] = dims;
    auto lhsFactors =
        rule.getOperandMapping(0).getDimMappings()[lhsDim].getFactorIndices();
    auto rhsFactors =
        rule.getOperandMapping(1).getDimMappings()[rhsDim].getFactorIndices();
    if (llvm::is_contained(lhsFactors, factorIdx) &&
        llvm::is_contained(rhsFactors, factorIdx)) {
      // This is a spatial factor.
      if (!isPureSpatialReductionDim(convOp, i, numWindows)) {
        // If the spatial reduction dimension involves sliding windows, padding,
        // or dilation, it has dual kPermutation semantics.
        return true;
      }
    }
  }
  return false;
}

// Returns true if factorIdx belongs to a grouped convolution dimension whose
// per-group size is greater than 1 (i.e., decomposed into multiple factors in
// the sharding rule).
bool isUnsupportedGroupedConvFactor(stablehlo::ConvolutionOp convOp,
                                    OpShardingRuleAttr rule,
                                    int64_t factorIdx) {
  if (convOp.getBatchGroupCount() <= 1 && convOp.getFeatureGroupCount() <= 1) {
    return false;
  }
  stablehlo::ConvDimensionNumbersAttr dimNums = convOp.getDimensionNumbers();
  int64_t lhsGroupDim = convOp.getBatchGroupCount() > 1
                            ? dimNums.getInputBatchDimension()
                            : dimNums.getInputFeatureDimension();
  int64_t rhsGroupDim = dimNums.getKernelOutputFeatureDimension();
  ArrayRef<int64_t> lhsFactors = rule.getOperandMapping(0)
                                     .getDimMappings()[lhsGroupDim]
                                     .getFactorIndices();
  ArrayRef<int64_t> rhsFactors = rule.getOperandMapping(1)
                                     .getDimMappings()[rhsGroupDim]
                                     .getFactorIndices();
  if (lhsFactors.size() == 1 && rhsFactors.size() == 1) {
    return false;
  }
  return llvm::is_contained(lhsFactors, factorIdx) ||
         llvm::is_contained(rhsFactors, factorIdx);
}

// Returns the sliced dimension of `operand` mapped to `factorIndex`, or
// `std::nullopt` if `factorIndex` maps to `update` instead.
std::optional<int64_t> getDynamicUpdateSliceDim(OpShardingRuleAttr rule,
                                                int64_t factorIndex) {
  if (rule.getNumOperands() == 0) {
    return std::nullopt;
  }
  for (auto [dimIdx, dimMapping] :
       llvm::enumerate(rule.getOperandMapping(0).getDimMappings())) {
    if (llvm::is_contained(dimMapping.getFactorIndices(), factorIndex)) {
      return dimIdx;
    }
  }
  return std::nullopt;
}

void resolvePermutationFactorsViaReplication(Operation* op,
                                             OpShardingRuleAttr rule,
                                             ResolutionState& state) {
  SmallVector<TensorShardingAttr> inShardings = getShardings(op->getOperands());
  SmallVector<TensorShardingAttr> outShardings = getShardings(op->getResults());
  std::optional<StringRef> meshName =
      getCommonMeshName(inShardings, outShardings, state.symbolTable, true);
  if (!meshName) {
    return;
  }
  MeshOp meshOp = getMeshOp(state.symbolTable, *meshName);
  if (!meshOp || meshOp.getMesh().isSingleDevice()) {
    return;
  }

  ShardingProjection projection =
      ShardingProjection::build(inShardings, outShardings, rule,
                                meshOp.getMesh(), /*closedIfMissing=*/true);
  UpdateTensorShardings update(op->getNumOperands(), op->getNumResults());

  SmallVector<AxisRefAttr> removedReductionAxes;
  for (int64_t i = 0; i < rule.getNumFactors(); ++i) {
    // Replicate permutation factors when halo exchange is disabled, unless the
    // slice or dynamic-update-slice dimension is communication-free. For
    // concatenate, replicate its concatenation dimension factor when halo
    // exchange is disabled.
    //
    // For convolution, we also replicate unsupported parallel
    // group-conv-factors.
    bool isReplicatedFactor =
        rule.getFactorType(i) == FactorType::kPermutation ||
        (isa<stablehlo::ConcatenateOp>(op) &&
         rule.getFactorType(i) == FactorType::kNeedReplication);
    if (auto convOp = dyn_cast<stablehlo::ConvolutionOp>(op);
        convOp && (isReductionFactorWithPermutationSemantics(convOp, rule, i) ||
                   isUnsupportedGroupedConvFactor(convOp, rule, i))) {
      isReplicatedFactor = true;
      if (rule.getFactorType(i) == FactorType::kReduction) {
        if (std::optional<ArrayRef<AxisRefAttr>> axes =
                getCompatibleFactorSharding(projection, i)) {
          llvm::append_range(removedReductionAxes, *axes);
        }
      }
    }
    if (!isReplicatedFactor) {
      continue;
    }
    if (auto sliceOp = dyn_cast<stablehlo::SliceOp>(op)) {
      SDY_CHECK(inShardings[0] == outShardings[0]);
      if (isCommunicationFreeSliceDim(i, sliceOp, inShardings[0],
                                      meshOp.getMesh())) {
        continue;
      }
    }
    if (auto padOp = dyn_cast<stablehlo::PadOp>(op)) {
      if (isCommunicationFreePadDim(i, padOp, inShardings[0],
                                    meshOp.getMesh())) {
        continue;
      }
    }
    if (auto dusOp = dyn_cast<stablehlo::DynamicUpdateSliceOp>(op)) {
      std::optional<int64_t> dimIdx = getDynamicUpdateSliceDim(rule, i);
      // DynamicUpdateSliceOp must have a single use so the intermediate reshard
      // inserted during permutation factor resolution has a single use, which
      // is required for ReshardOfReshardPattern to fold it with the user
      // reshard.
      bool singleUserReplicatesDim = false;
      if (dimIdx && dusOp->hasOneUse()) {
        auto reshardOp = dyn_cast<ReshardOp>(*dusOp->user_begin());
        singleUserReplicatesDim =
            reshardOp &&
            reshardOp.getSharding().getDimSharding(*dimIdx).emptyAxes();
      }
      if (dimIdx && !singleUserReplicatesDim &&
          isCommunicationFreeDynamicUpdateSliceDim(
              *dimIdx, dusOp, inShardings[0], meshOp.getMesh())) {
        continue;
      }
    }

    update |= projection.updateSharding(i, /*axes=*/{}, /*overflowAxes=*/{});
  }

  sdy::AllReduceOp remainingAllReduceOp = nullptr;
  if (!removedReductionAxes.empty() && !outShardings.empty() &&
      outShardings[0]) {
    TensorShardingAttr oldOutSharding = outShardings[0];
    ArrayRef<AxisRefAttr> oldUnreduced = oldOutSharding.getUnreducedAxes();
    if (!oldUnreduced.empty()) {
      SmallVector<AxisRefAttr> newUnreduced;
      for (AxisRefAttr axis : oldUnreduced) {
        SmallVector<AxisRefAttr> curAxes = {axis};
        for (AxisRefAttr removed : removedReductionAxes) {
          SmallVector<AxisRefAttr> nextAxes;
          for (AxisRefAttr cur : curAxes) {
            if (!cur.overlaps(removed)) {
              nextAxes.push_back(cur);
              continue;
            }
            if (auto prefix = cur.getPrefixWithoutOverlap(removed)) {
              nextAxes.push_back(*prefix);
            }
            if (auto suffix =
                    cur.getSuffixWithoutOverlap(removed, meshOp.getMesh())) {
              nextAxes.push_back(*suffix);
            }
          }
          curAxes = std::move(nextAxes);
        }
        llvm::append_range(newUnreduced, curAxes);
      }
      if (newUnreduced != oldUnreduced) {
        // Remove the dual-semantics reduction axes from the op's result
        // unreduced sharding annotation and update/remove its downstream
        // sdy.all_reduce user before inserting explicit reshards.
        SDY_CHECK(oldOutSharding.getReductionOp() == ReductionOp::SUM);
        outShardings[0] = oldOutSharding.replaceUnreducedAxes(newUnreduced);
        sdy::setShardings(op, outShardings[0]);
        projection.getMutableResult(0).unreducedAxes = newUnreduced;
        if (op->getResult(0).hasOneUse()) {
          if (auto allReduceOp = dyn_cast<sdy::AllReduceOp>(
                  *op->getResult(0).getUsers().begin());
              allReduceOp && allReduceOp.getReductionAxes() == oldUnreduced) {
            if (newUnreduced.empty()) {
              allReduceOp.replaceAllUsesWith(allReduceOp.getTensor());
              state.opsToErase.push_back(allReduceOp);
            } else {
              allReduceOp.setReductionAxesAttr(
                  AxisRefListAttr::get(op->getContext(), newUnreduced));
              remainingAllReduceOp = allReduceOp;
            }
          }
        }
      }
    }
  }

  if (update.updateOperands.any() || update.updateResults.any()) {
    insertExplicitReshards(op, inShardings, outShardings, projection, update,
                           state.rewriter, rule, state.symbolTable, meshOp);
    if (remainingAllReduceOp && update.updateResults.test(0)) {
      if (auto reshardOp = remainingAllReduceOp.getTensor()
                               .getDefiningOp<sdy::ReshardOp>()) {
        // Move the result reshard after remainingAllReduceOp so that
        // sdy.reshard operates on the reduced tensor rather than the
        // intermediate unreduced tensor.
        remainingAllReduceOp.getTensorMutable().assign(reshardOp.getInput());
        remainingAllReduceOp.setOutShardingAttr(
            getSharding(op->getResult(0)).replaceUnreducedAxes({}));
        state.rewriter.setInsertionPointAfter(remainingAllReduceOp);
        auto newReshardOp = sdy::ReshardOp::create(
            state.rewriter, reshardOp.getLoc(),
            remainingAllReduceOp.getResult(),
            reshardOp.getSharding().replaceUnreducedAxes({}));
        state.rewriter.replaceAllUsesExcept(remainingAllReduceOp.getResult(),
                                            newReshardOp, newReshardOp);
        state.opsToErase.push_back(reshardOp);
      }
    }
  }
}

// -----------------------------------------------------------------------------
// The module pass.
// -----------------------------------------------------------------------------

// Returns the maximum channel ID in moduleOp plus one.
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
    ResolutionState state{rewriter,      symbolTable,  meshCache,
                          nextChannelId, replicaCount, partitionCount};

    // Walk the module to resolve permutation factors for each op.
    moduleOp.walk([&](Operation* op) {
      // Skip terminators and any operations not in the StableHLO dialect
      // unless they implement PartitioningOpInterface. This prevents "unknown
      // op" warnings for Shardy collectives or return ops.
      auto externalOp = dyn_cast<PartitioningOpInterface>(op);
      if (op->hasTrait<OpTrait::IsTerminator>() ||
          (!inDialect<stablehlo::StablehloDialect>(op) && !externalOp)) {
        return;
      }

      // Ops with special handlers.
      if (auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(op)) {
        SDY_CHECK(
            succeeded(handleReshapeOp(reshapeOp, enableHaloExchange, state)));
        return;
      }
      if (auto rngOp = dyn_cast<stablehlo::RngBitGeneratorOp>(op)) {
        handleRngBitGeneratorOp(rngOp, rngBitGeneratorUnsafe, state);
        return;
      }

      OpShardingRuleAttr rule = getOrCreateShardingRule(op, false, false);
      if (!rule || rule.isCustom()) {
        return;
      }

      // Identify if the op defines any factor that needs resolution:
      // permutation factors, dual-semantics convolution factors, unsupported
      // grouped convolution factors, or concatenating factors.
      auto shouldResolveFactor = [&](int64_t i) {
        if (rule.getFactorType(i) == FactorType::kPermutation) {
          return true;
        }
        if (isa<stablehlo::ConcatenateOp>(op)) {
          // We don't check whether the concatenate operands are slices, because
          // if they aren't the factor is true kNeedReplication and is already
          // replicated.
          return rule.getFactorType(i) == FactorType::kNeedReplication;
        }
        if (auto convOp = dyn_cast<stablehlo::ConvolutionOp>(op)) {
          return isReductionFactorWithPermutationSemantics(convOp, rule, i) ||
                 isUnsupportedGroupedConvFactor(convOp, rule, i);
        }
        return false;
      };
      if (llvm::none_of(llvm::seq<int64_t>(0, rule.getNumFactors()),
                        shouldResolveFactor)) {
        return;
      }

      if (externalOp) {
        rewriter.setInsertionPoint(op);
        if (std::optional<LogicalResult> res =
                externalOp.resolvePermutationFactors(rule, rewriter,
                                                     state.symbolTable)) {
          if (failed(*res)) {
            signalPassFailure();
          }
          return;
        }
      }

      // Dispatch to HALO exchange if enabled and implemented for the op.
      if (enableHaloExchange) {
        if (auto sliceOp = dyn_cast<stablehlo::SliceOp>(op);
            sliceOp && sliceOp->hasOneUse()) {
          if (auto userConcat =
                  dyn_cast<stablehlo::ConcatenateOp>(*sliceOp->user_begin());
              userConcat &&
              findRotateRightPattern(userConcat, state.symbolTable)) {
            return;
          }
        }
        bool resolved =
            llvm::TypeSwitch<Operation*, bool>(op)
                .Case([&](stablehlo::ConcatenateOp concatOp) {
                  return succeeded(handleConcatenateOp(concatOp, state));
                })
                .Case([&](stablehlo::ConvolutionOp convOp) {
                  return succeeded(handleConvolutionOp(convOp, state));
                })
                .Case([&](stablehlo::PadOp padOp) {
                  return succeeded(handlePadOp(padOp, state));
                })
                .Case([&](stablehlo::ReverseOp reverseOp) {
                  return succeeded(handleReverseOp(reverseOp, state));
                })
                .Case([&](stablehlo::SliceOp sliceOp) {
                  return succeeded(handleSliceOp(sliceOp, state));
                })
                .Default(false);
        if (resolved) {
          return;
        }
      }

      // Otherwise, use a generic resolution based on explicit reshards.
      resolvePermutationFactorsViaReplication(op, rule, state);
    });
    for (Operation* opToErase : state.opsToErase) {
      rewriter.eraseOp(opToErase);
    }
  }
};

}  // namespace
}  // namespace sdy
}  // namespace mlir
