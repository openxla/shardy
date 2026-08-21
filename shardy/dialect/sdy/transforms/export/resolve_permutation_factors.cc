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
#include <numeric>
#include <optional>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"  // IWYU pragma: keep
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
  int64_t replicaCount = 1;
  int64_t partitionCount = 1;
};

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
  llvm::SmallDenseSet<StringRef> manualAxes;
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
  return getDeviceId(state.replicaCount, state.partitionCount, loc,
                     state.rewriter);
}

// Returns the logical index of the shard that the given device (`deviceId`)
// resides in, along the given dimension sharding.
int64_t getShardIndex(int64_t deviceId, MeshAttr mesh,
                      DimensionShardingAttr dimSharding) {
  return getShardIndex(deviceId, mesh, dimSharding.getAxes());
}

// Projects the runtime globalPartitionId to a 0-based shard index for the
// provided targetDimSharding (which may include sub-axes or combined output
// dimensions in a reshape group).
//
// For meshes with explicit device_ids, precomputes a lookup table of shard
// indices at compile time and emits a stablehlo.dynamic_slice. For standard
// iota meshes, emits closed-form HLO arithmetic ops (div, rem, mul, add).
Value convertPartitionIdToIdInGroup(Location loc, Value globalPartitionId,
                                    MeshAttr mesh,
                                    DimensionShardingAttr targetDimSharding,
                                    IRRewriter& rewriter) {
  Type i64Ty = rewriter.getI64Type();
  Value globalPartitionIdI64 = globalPartitionId;
  if (cast<RankedTensorType>(globalPartitionId.getType()).getElementType() !=
      i64Ty) {
    globalPartitionIdI64 = stablehlo::ConvertOp::create(
        rewriter, loc, RankedTensorType::get({}, i64Ty), globalPartitionId);
  }

  // If mesh has explicit non-iota device_ids mapping, use lookup table constant.
  if (!mesh.getDeviceIds().empty()) {
    int64_t totalDevices = mesh.getTotalSize();
    SmallVector<int64_t> shardIndices;
    shardIndices.reserve(totalDevices);
    for (int64_t devId = 0; devId < totalDevices; ++devId) {
      shardIndices.push_back(getShardIndex(devId, mesh, targetDimSharding));
    }

    auto shardIndicesConst = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseIntElementsAttr::get(
            RankedTensorType::get({totalDevices}, i64Ty), shardIndices));

    auto sliceOp = stablehlo::DynamicSliceOp::create(
        rewriter, loc, RankedTensorType::get({1}, i64Ty), shardIndicesConst,
        globalPartitionIdI64, rewriter.getDenseI64ArrayAttr({1}));

    return stablehlo::ReshapeOp::create(
        rewriter, loc, RankedTensorType::get({}, i64Ty), sliceOp);
  }

  // Standard iota mesh: compute shard index via scalar HLO arithmetic ops.
  auto getConstVal = [&](int64_t val) -> Value {
    return stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseIntElementsAttr::get(RankedTensorType::get({}, i64Ty), {val}));
  };

  Value idInGroup = nullptr;
  ArrayRef<MeshAxisAttr> meshAxes = mesh.getAxes();
  int64_t totalMeshDevices = mesh.getTotalSize();

  for (AxisRefAttr axis : targetDimSharding.getAxes()) {
    int64_t axisSize = axis.getSize(mesh);
    int64_t suffixSize = 1;
    bool foundAxis = false;
    for (MeshAxisAttr meshAxis : meshAxes) {
      if (foundAxis) {
        suffixSize *= meshAxis.getSize();
      }
      if (meshAxis.getName() == axis.getName()) {
        foundAxis = true;
      }
    }

    int64_t fullSize = mesh.getAxisSize(axis.getName());
    int64_t subAxisStride = fullSize / (axis.getSubAxisPreSize() * axisSize);
    int64_t divisor = suffixSize * subAxisStride;

    Value coord = globalPartitionIdI64;
    if (divisor > 1) {
      coord = stablehlo::DivOp::create(rewriter, loc, coord.getType(), coord,
                                       getConstVal(divisor));
    }
    if (axisSize < totalMeshDevices) {
      coord = stablehlo::RemOp::create(rewriter, loc, coord.getType(), coord,
                                       getConstVal(axisSize));
    }

    if (!idInGroup) {
      idInGroup = coord;
    } else {
      idInGroup = stablehlo::MulOp::create(rewriter, loc, idInGroup,
                                           getConstVal(axisSize));
      idInGroup = stablehlo::AddOp::create(rewriter, loc, idInGroup, coord);
    }
  }

  if (!idInGroup) {
    idInGroup = getConstVal(0);
  }

  return idInGroup;
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
                                     bool needUndilated = true) {
  IRRewriter& rewriter = state.rewriter;
  Type i64Ty = rewriter.getI64Type();
  auto diffSizeVal = stablehlo::ConstantOp::create(
      rewriter, loc,
      DenseIntElementsAttr::get(RankedTensorType::get({}, i64Ty),
                                ArrayRef<int64_t>{diffSize}));
  auto baseOffsetVal = stablehlo::ConstantOp::create(
      rewriter, loc,
      DenseIntElementsAttr::get(RankedTensorType::get({}, i64Ty),
                                ArrayRef<int64_t>{baseOffset}));
  Value offsetInPartition =
      stablehlo::MulOp::create(rewriter, loc, partitionId, diffSizeVal);
  Value dilatedOffset =
      stablehlo::AddOp::create(rewriter, loc, offsetInPartition, baseOffsetVal);

  auto zeroConst = stablehlo::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(RankedTensorType::get({}, i64Ty)));

  dilatedOffset =
      stablehlo::MaxOp::create(rewriter, loc, dilatedOffset, zeroConst);

  Value undilatedOffset;
  if (needUndilated) {
    if (baseDilation == 1) {
      undilatedOffset = dilatedOffset;
    } else {
      auto baseDilationVal = stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseIntElementsAttr::get(RankedTensorType::get({}, i64Ty),
                                    ArrayRef<int64_t>{baseDilation}));
      undilatedOffset =
          stablehlo::DivOp::create(rewriter, loc, dilatedOffset.getType(),
                                   dilatedOffset, baseDilationVal);
    }
  }
  return {dilatedOffset, undilatedOffset, zeroConst};
}

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
                                 sharding.getUnreducedAxes());
}

// -----------------------------------------------------------------------------
// Pad and slice helpers for dimension alignment.
// -----------------------------------------------------------------------------

inline int64_t getPaddedDimSize(int64_t dimSize, int64_t shardCount) {
  return llvm::divideCeil(dimSize, shardCount) * shardCount;
}

// Creates a zero scalar constant of element type `elemType`.
inline Value createZeroScalarConstant(Location loc, IRRewriter& rewriter,
                                      Type elemType) {
  return stablehlo::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(RankedTensorType::get({}, elemType)));
}

// Pads the high side of `operand` with `paddingValue` (default zero if null)
// to match the target `paddedShape`. When used to align dimensions for the
// input sharding's own divisibility, the created pad op is a
// communication-free operation.
Value padHighSideToShape(Location loc, IRRewriter& rewriter, Value operand,
                         ArrayRef<int64_t> paddedShape,
                         TensorShardingAttr sharding,
                         Value paddingValue = nullptr) {
  auto type = cast<RankedTensorType>(operand.getType());
  if (paddedShape == type.getShape()) {
    return operand;
  }
  auto paddedType = RankedTensorType::get(paddedShape, type.getElementType());
  SmallVector<int64_t> edgePaddingHigh;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    edgePaddingHigh.push_back(paddedShape[i] - type.getDimSize(i));
  }
  Value padVal = paddingValue ? paddingValue
                              : createZeroScalarConstant(loc, rewriter,
                                                         type.getElementType());
  auto padHighOp = stablehlo::PadOp::create(
      rewriter, loc, paddedType, operand, padVal,
      rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(type.getRank(), 0)),
      rewriter.getDenseI64ArrayAttr(edgePaddingHigh),
      rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(type.getRank(), 0)));
  setSharding(padHighOp.getResult(), sharding);
  return padHighOp.getResult();
}

// Slices the high side of `operand` (e.g., to trim away high-side padding
// elements previously inserted by `padHighSideToShape`) to match `targetType`.
// When used to align dimensions after local operations complete, the created
// slice op is a communication-free operation.
Value sliceHighSideToShape(Location loc, IRRewriter& rewriter, Value operand,
                           Type targetType, TensorShardingAttr sharding) {
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
SmallVector<int64_t> getDataShiftSourceTargetPairs(
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

// Generates code to right shift elements logically in each local device across
// the partitioning space using HALO exchange and support multi-hop shifts.
Value rightShiftData(Location loc, Value input, int64_t dim, int64_t totalShift,
                     TensorShardingAttr sharding, MeshAttr mesh,
                     const llvm::SmallDenseSet<StringRef>& manualAxes,
                     ResolutionState& state) {
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t shardSize = inputType.getDimSize(dim);
  int64_t rank = inputType.getRank();
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
        getDataShiftSourceTargetPairs(mesh, manualAxesInDim, deviceOffset);
    if (pairs.empty()) {
      Value zeroConst = createZeroScalarConstant(loc, state.rewriter,
                                                 inputType.getElementType());
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

SmallVector<StringAttr> getManualAxesAttrs(
    MeshAttr mesh, const llvm::SmallDenseSet<StringRef>& manualAxes,
    Builder& builder) {
  SmallVector<StringAttr> manualAxesAttrs;
  for (auto axis : mesh.getAxes()) {
    if (manualAxes.contains(axis.getName())) {
      manualAxesAttrs.push_back(builder.getStringAttr(axis.getName()));
    }
  }
  return manualAxesAttrs;
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
  auto channelAttr = stablehlo::ChannelHandleAttr::get(
      state.rewriter.getContext(), state.nextChannelId++, 1);
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
// interior padding.
Value assembleHaloExchangeBuffer(
    Location loc, Value input, int64_t dim, int64_t edgePadSize,
    Value paddingValue, int64_t leftHops, int64_t rightHops,
    TensorShardingAttr sharding, MeshAttr mesh,
    const llvm::SmallDenseSet<StringRef>& manualAxes,
    TensorShardingAttr localSharding, ResolutionState& state,
    int64_t interiorPad = 0, const DimExchangeInfo* dimExchange = nullptr) {
  SmallVector<AxisRefAttr> manualAxesInDim;
  DimensionShardingAttr activeDimSharded =
      (dimExchange && dimExchange->reshapeInfo)
          ? dimExchange->reshapeInfo->getTargetDimSharding()
          : sharding.getDimShardings()[dim];
  for (auto axisRef : activeDimSharded.getAxes()) {
    if (manualAxes.contains(axisRef.getName())) {
      manualAxesInDim.push_back(axisRef);
    }
  }

  SmallVector<Value> concatSegments;
  concatSegments.reserve(leftHops + 1 + rightHops);

  auto fetchNeighbors = [&](int64_t start, int64_t end) {
    for (int64_t offset = start; offset >= end; --offset) {
      SmallVector<int64_t> pairs =
          getDataShiftSourceTargetPairs(mesh, manualAxesInDim, offset);
      concatSegments.push_back(
          mayCollectivePermute(loc, input, pairs, state));
    }
  };

  // Fetch Left Neighbors (hop count h = leftHops down to 1)
  fetchNeighbors(leftHops, 1);
  // Local Shard
  concatSegments.push_back(input);
  // Fetch Right Neighbors (hop count h = 1 up to rightHops, offset maps to -1
  // down to -rightHops)
  fetchNeighbors(-1, -rightHops);

  // Concatenate all segments.
  Value concat = stablehlo::ConcatenateOp::create(state.rewriter, loc,
                                                  concatSegments, dim);
  setSharding(concat, localSharding);

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
    MeshAttr mesh, const llvm::SmallDenseSet<StringRef>& manualAxes,
    TensorShardingAttr localSharding, int64_t padLow, int64_t stride,
    int64_t sOut, int64_t baseDilation, bool dilatedHaloBuffer,
    ResolutionState& state) {
  int64_t edgePadSize = dilatedHaloBuffer
                            ? dimExchange.sFootprint * baseDilation
                            : dimExchange.sFootprint;
  Value paddedConcat = assembleHaloExchangeBuffer(
      loc, operand, dim, edgePadSize, paddingValue, dimExchange.leftHops,
      dimExchange.rightHops, sharding, mesh, manualAxes, localSharding, state,
      dilatedHaloBuffer ? baseDilation - 1 : 0, &dimExchange);

  Value partitionId = createDeviceId(loc, state);

  auto inputType = cast<RankedTensorType>(operand.getType());
  int64_t sIn = inputType.getShape()[dim];
  int64_t sInDilated = sIn * baseDilation;
  int64_t activeSOut = sOut == 0 ? dimExchange.sFootprint : sOut;
  int64_t diffSize = activeSOut * stride - sInDilated;
  int64_t baseOffsetInPaddedConcat = -padLow +
                                     dimExchange.leftHops * sInDilated +
                                     dimExchange.sFootprint * baseDilation;

  llvm::SmallDenseSet<StringRef> manualAxesInDim;
  for (AxisRefAttr axis : sharding.getDimShardings()[dim].getAxes()) {
    manualAxesInDim.insert(axis.getName());
  }

  DimensionShardingAttr activeDimShardedAttr =
      dimExchange.reshapeInfo ? dimExchange.reshapeInfo->getTargetDimSharding()
                              : sharding.getDimShardings()[dim];
  Value idInPartitionGroup = convertPartitionIdToIdInGroup(
      loc, partitionId, mesh, activeDimShardedAttr, state.rewriter);

  Value offset;
  Value zeroConst;
  if (dimExchange.reshapeInfo) {
    auto i64Ty = state.rewriter.getI64Type();
    SDY_CHECK(dilatedHaloBuffer);
    SDY_CHECK_EQ(baseDilation, 1);
    SmallVector<int64_t> offsets(dimExchange.shardExchanges.size(), 0);
    for (int64_t t = 0; t < dimExchange.shardExchanges.size(); ++t) {
      const auto& ex = dimExchange.shardExchanges[t];
      int64_t edgePad = dimExchange.sFootprint;
      int64_t sourceId = ex.sourceId != -1 ? ex.sourceId : t;
      int64_t relativeHop = dimExchange.leftHops + sourceId - t;
      int64_t off = -padLow + edgePad + relativeHop * sIn + ex.localOffset;
      offsets[t] = off;
    }
    int64_t n = offsets.size();
    auto offsetsConst = stablehlo::ConstantOp::create(
        state.rewriter, loc,
        DenseIntElementsAttr::get(RankedTensorType::get({n}, i64Ty), offsets));
    auto sliceOp = stablehlo::DynamicSliceOp::create(
        state.rewriter, loc, RankedTensorType::get({1}, i64Ty), offsetsConst,
        idInPartitionGroup, state.rewriter.getDenseI64ArrayAttr({1}));
    offset = stablehlo::ReshapeOp::create(
        state.rewriter, loc, RankedTensorType::get({}, i64Ty), sliceOp);
    zeroConst = stablehlo::ConstantOp::create(
        state.rewriter, loc,
        state.rewriter.getZeroAttr(RankedTensorType::get({}, i64Ty)));
  } else {
    DeviceOffsetInfo offsetInfo =
        getDeviceOffsetInfo(loc, idInPartitionGroup, diffSize,
                            baseOffsetInPaddedConcat, baseDilation, state,
                            /*needUndilated=*/!dilatedHaloBuffer);
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
    const llvm::SmallDenseSet<StringRef>& manualAxes,
    ArrayRef<DimExchangeInfo> dimExchanges, ArrayRef<int64_t> edgePaddingLow,
    ArrayRef<int64_t> windowStrides, bool dilatedHaloBuffer,
    ResolutionState& state, ArrayRef<int64_t> baseDilations = {}) {
  TensorShardingAttr localSharding =
      removeAxesFromSharding(sharding, manualAxes);
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

int64_t getManualShardCount(DimensionShardingAttr dimSharding, MeshAttr mesh,
                            const llvm::SmallDenseSet<StringRef>& manualAxes) {
  int64_t count = 1;
  for (AxisRefAttr axis : dimSharding.getAxes()) {
    if (manualAxes.contains(axis.getName())) {
      count *= axis.getSize(mesh);
    }
  }
  return count;
}

// Performs HALO exchange over the sharded input inside a newly created
// `sdy.manual_computation` block.
//
// Inside the block, it shifts local shards along the sharded dimensions
// specified in `dimExchanges` to align boundary requirements. It then runs
// the user-provided `postProcess` callback to process the local result
// before returning the output of the manual computation.
Value haloDataExchange(
    Location loc, Value divisibleInput, ArrayRef<int64_t> divisibleInputShape,
    RankedTensorType origInputType, RankedTensorType origOutputType,
    TensorShardingAttr inSharding, TensorShardingAttr outSharding,
    MeshAttr mesh, const llvm::SmallDenseSet<StringRef>& manualAxes,
    Value paddingValue, ArrayRef<int64_t> edgePaddingLow,
    ArrayRef<DimExchangeInfo> dimExchanges, const PostProcessFn& postProcess,
    ResolutionState& state, ArrayRef<int64_t> windowStrides = {},
    ArrayRef<int64_t> baseDilations = {}) {
  // Extract manual axes while preserving their original order in the mesh.
  SmallVector<StringAttr> manualAxesAttrs =
      getManualAxesAttrs(mesh, manualAxes, state.rewriter);

  // Compute the local shape for input in the manual block based on
  // divisibleInputShape.
  SmallVector<int64_t> localShape;
  auto inputType = cast<RankedTensorType>(divisibleInput.getType());
  for (auto [dim, dimSharding] :
       llvm::enumerate(inSharding.getDimShardings())) {
    int64_t shardCount = getManualShardCount(dimSharding, mesh, manualAxes);
    // Target shape is the padded shape of the input.
    SDY_CHECK(divisibleInputShape[dim] % shardCount == 0);
    localShape.push_back(divisibleInputShape[dim] / shardCount);
  }

  // Build the manual computation block with required operands.
  SmallVector<Value> operands = {divisibleInput, paddingValue};
  SmallVector<TensorShardingAttr> inShardings = {
      inSharding, TensorShardingAttr::getFullyReplicated(
                      state.rewriter.getContext(), 0, inSharding.getMeshOrRef(),
                      /*isClosed=*/false)};

  SmallVector<int64_t> paddedGlobalResultShape;
  paddedGlobalResultShape.reserve(origOutputType.getRank());
  for (auto [i, dimSize] : llvm::enumerate(origOutputType.getShape())) {
    paddedGlobalResultShape.push_back(getPaddedDimSize(
        dimSize, outSharding.getDimShardings()[i].getShardedSize(mesh)));
  }
  RankedTensorType paddedGlobalResultType = RankedTensorType::get(
      paddedGlobalResultShape, origOutputType.getElementType());

  SmallVector<Type> resultTypes = {paddedGlobalResultType};
  auto manualComp = ManualComputationOp::create(
      state.rewriter, loc, resultTypes, operands,
      TensorShardingPerValueAttr::get(state.rewriter.getContext(), inShardings),
      TensorShardingPerValueAttr::get(state.rewriter.getContext(),
                                      {outSharding}),
      manualAxesAttrs);

  Region& body = manualComp.getBody();
  body.emplaceBlock();
  Value localInput = body.addArgument(
      RankedTensorType::get(localShape, inputType.getElementType()), loc);
  Value localPaddingValue = body.addArgument(paddingValue.getType(), loc);

  OpBuilder::InsertionGuard guard(state.rewriter);
  state.rewriter.setInsertionPointToStart(&body.front());

  Value exchangedLocal = localInput;
  exchangedLocal = exchangeDataWithDynamicOffset(
      loc, exchangedLocal, localPaddingValue, inSharding, mesh, manualAxes,
      dimExchanges, edgePaddingLow, windowStrides,
      /*dilatedHaloBuffer=*/windowStrides.empty(), state, baseDilations);

  exchangedLocal = postProcess(exchangedLocal, localPaddingValue,
                               paddedGlobalResultType, state);

  ReturnOp::create(state.rewriter, loc, exchangedLocal);
  return manualComp.getResult(0);
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
    int64_t start = (t * sOut * stride - padLow) / baseDilation;
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
    info.manualAxes.insert(axis.getName());
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
        return getDimExchangeInfo(shardCount, sIn, sOut,
                                  /*sFootprint=*/sOut, pLow,
                                  /*stride=*/1, baseDilation,
                                  inputType.getDimSize(dim));
      });
}

// Implements the non-trivial padding operation on replicated dimensions.
Value handleReplicatedPadDims(
    Location loc, Value exchangedLocal, Value localPaddingValue,
    RankedTensorType paddedGlobalType, TensorShardingAttr sharding,
    MeshAttr mesh, const llvm::SmallDenseSet<StringRef>& manualAxes,
    ArrayRef<int64_t> edgePaddingLow, ArrayRef<int64_t> edgePaddingHigh,
    ArrayRef<int64_t> interiorPadding, ResolutionState& state) {
  auto exchangedLocalType = cast<RankedTensorType>(exchangedLocal.getType());
  int64_t rank = exchangedLocalType.getRank();
  TensorShardingAttr localSharding =
      removeAxesFromSharding(sharding, manualAxes);

  SmallVector<int64_t> localResultShape(rank, 0), localLow(rank, 0),
      localHigh(rank, 0), localInterior(rank, 0);
  bool needsLocalPad = false;
  for (int64_t i = 0; i < rank; ++i) {
    int64_t manualFactor = 1;
    for (auto axis : sharding.getDimShardings()[i].getAxes()) {
      if (manualAxes.contains(axis.getName())) {
        manualFactor *= axis.getSize(mesh);
      }
    }
    int64_t sOutLocal = paddedGlobalType.getDimSize(i) / manualFactor;
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
  if (!mesh || mesh.isMaximal()) {
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
      padHighSideToShape(loc, rewriter, origInput, info.divisibleInputShape,
                         inSharding, padOp.getPaddingValue());

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
      postProcess, state,
      /*windowStrides=*/{}, baseDilations);

  // Trim final tensor shape to match expected output shape.
  result =
      sliceHighSideToShape(loc, rewriter, result, padOp.getType(), inSharding);

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
  llvm::SmallDenseSet<StringRef> nonDivisibleAxes;
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
      auto collectAxes = [&](int64_t startDim, int64_t lastDim,
                             TensorShardingAttr sharding) {
        for (int64_t d = startDim; d < lastDim; ++d) {
          for (AxisRefAttr axis : sharding.getDimShardings()[d].getAxes()) {
            a.nonDivisibleAxes.insert(axis.getName());
          }
        }
      };
      collectAxes(g.inStartDim, g.inLastDim, inSharding);
      collectAxes(g.outStartDim, g.outLastDim, outSharding);
    }
  }

  if (a.nonDivisibleAxes.empty() || !enableHaloExchange) {
    return a;
  }

  a.haloInfo = getReshapeHaloInfo(a.reshapeGroups, origInputType, resultType,
                                  a.paddedInputShape, a.paddedOutputShape);
  return a;
}

// Computes DimExchangeInfo for a sharded reshape op dimension group
// represented by ReshapeGroupInfo.
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
    ArrayRef<int64_t> divisibleInputShape, ArrayRef<int64_t> paddedOutputShape,
    const ReshapeHaloInfo& haloInfo,
    const llvm::SmallDenseSet<StringRef>& nonDivisibleAxes) {
  DataExchangeInfo info;
  info.manualAxes = nonDivisibleAxes;
  auto origInputType = reshapeOp.getOperand().getType();
  int64_t activeRank = origInputType.getRank();
  TensorShardingAttr activeSharding = inSharding;

  const auto *activeGroupIt = haloInfo.activeGroups.begin();
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
    ReshapeHaloInfo& haloInfo,
    const llvm::SmallDenseSet<StringRef>& nonDivisibleAxes,
    ResolutionState& state) {
  SDY_CHECK(!haloInfo.activeGroups.empty());

  DataExchangeInfo info = getReshapeDataExchangeInfo(
      reshapeOp, inSharding, outSharding, mesh, divisibleInputShape,
      paddedOutputShape, haloInfo, nonDivisibleAxes);

  RankedTensorType resultType = reshapeOp.getResult().getType();
  if (!canUseHalo(info)) {
    return handleReshapeOpReplicate(reshapeOp, inSharding, outSharding,
                                    resultType, reshapeGroups, state);
  }

  Location loc = reshapeOp.getLoc();
  IRRewriter& rewriter = state.rewriter;
  rewriter.setInsertionPoint(reshapeOp);
  Value paddingValue =
      createZeroScalarConstant(loc, rewriter, resultType.getElementType());
  Value operand = reshapeOp.getOperand();
  auto origInputType = cast<RankedTensorType>(operand.getType());
  SmallVector<int64_t> edgePaddingLow(origInputType.getRank(), 0);
  Value paddedInput = padHighSideToShape(loc, rewriter, operand,
                                         divisibleInputShape, inSharding);

  auto postProcess = [&](Value exchangedLocal, Value localPaddingValue,
                         RankedTensorType paddedGlobalResultType,
                         ResolutionState& state) -> Value {
    // Reshape the local value after data exchange to the output shape.
    SmallVector<int64_t> localOutputShape;
    for (auto [dim, dimSharding] :
         llvm::enumerate(outSharding.getDimShardings())) {
      localOutputShape.push_back(
          paddedOutputShape[dim] /
          getManualShardCount(dimSharding, mesh, info.manualAxes));
    }
    auto localOutputType =
        RankedTensorType::get(localOutputShape, resultType.getElementType());
    TensorShardingAttr localOutSharding =
        removeAxesFromSharding(outSharding, info.manualAxes);
    auto localReshape = stablehlo::ReshapeOp::create(
        state.rewriter, loc, localOutputType, exchangedLocal);
    setSharding(localReshape.getResult(), localOutSharding);
    return localReshape;
  };

  Value result = haloDataExchange(
      loc, paddedInput, divisibleInputShape, origInputType, resultType,
      inSharding, outSharding, mesh, info.manualAxes, paddingValue,
      edgePaddingLow, info.dimExchanges, postProcess, state);
  result = sliceHighSideToShape(loc, rewriter, result, resultType, outSharding);
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
  SDY_CHECK((inSharding != nullptr) == (outSharding != nullptr));
  if (!inSharding) {
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
        int64_t sliceSize = sliceOp.getType().getDimSize(dim);
        int64_t sOut = llvm::divideCeil(sliceSize, shardCount);
        int64_t start = sliceOp.getStartIndices()[dim];
        return getDimExchangeInfo(shardCount, sIn, sOut,
                                  /*sFootprint=*/sOut, /*padLow=*/-start,
                                  /*stride=*/1, /*baseDilation=*/1,
                                  inputType.getDimSize(dim));
      });
}

// Implements the slicing operation on replicated dimensions.
Value handleReplicatedSliceDims(
    Location loc, Value exchangedLocal, TensorShardingAttr sharding,
    MeshAttr mesh, const llvm::SmallDenseSet<StringRef>& manualAxes,
    ArrayRef<int64_t> startIndices, ArrayRef<int64_t> limitIndices,
    ResolutionState& state) {
  auto exchangedLocalType = cast<RankedTensorType>(exchangedLocal.getType());
  int64_t rank = exchangedLocalType.getRank();
  TensorShardingAttr localSharding =
      removeAxesFromSharding(sharding, manualAxes);

  SmallVector<int64_t> localSliceStarts(rank, 0);
  SmallVector<int64_t> localSliceLimits =
      llvm::to_vector(exchangedLocalType.getShape());
  bool needsLocalSlice = false;
  for (int64_t i = 0; i < rank; ++i) {
    if (sharding.getDimShardings()[i].getShardedSize(mesh) > 1) {
      continue;
    }
    int64_t start = startIndices[i];
    int64_t limit = limitIndices[i];
    if (start != 0 || limit != exchangedLocalType.getDimSize(i)) {
      localSliceStarts[i] = start;
      localSliceLimits[i] = limit;
      needsLocalSlice = true;
    }
  }

  if (!needsLocalSlice) {
    return exchangedLocal;
  }

  SmallVector<int64_t> localResultShape =
      llvm::to_vector(exchangedLocalType.getShape());
  for (int64_t i = 0; i < rank; ++i) {
    localResultShape[i] = localSliceLimits[i] - localSliceStarts[i];
  }
  auto localResultType = RankedTensorType::get(
      localResultShape, exchangedLocalType.getElementType());
  exchangedLocal = stablehlo::SliceOp::create(
      state.rewriter, loc, localResultType, exchangedLocal,
      state.rewriter.getDenseI64ArrayAttr(localSliceStarts),
      state.rewriter.getDenseI64ArrayAttr(localSliceLimits),
      state.rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(rank, 1)));
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
  if (!mesh || mesh.isMaximal()) {
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
      padHighSideToShape(loc, rewriter, origInput, info.divisibleInputShape,
                         inSharding, /*paddingValue=*/nullptr);

  Value paddingValue =
      createZeroScalarConstant(loc, rewriter, origInputType.getElementType());

  int64_t rank = origInputType.getRank();
  SmallVector<int64_t> edgePaddingLow, edgePaddingHigh, interiorPadding;
  edgePaddingLow.reserve(rank);
  edgePaddingHigh.reserve(rank);
  interiorPadding.reserve(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    int64_t start = sliceOp.getStartIndices()[dim];
    int64_t limit = sliceOp.getLimitIndices()[dim];
    int64_t size = origInputType.getDimSize(dim);
    edgePaddingLow.push_back(-start);
    edgePaddingHigh.push_back(limit - size);
    interiorPadding.push_back(0);
  }

  // Define step 3 as post processing inside haloDataExchange.
  auto postProcess = [&](Value exchangedLocal, Value localPaddingValue,
                         RankedTensorType /*unused padddedGlobalResultType*/,
                         ResolutionState& state) -> Value {
    return handleReplicatedSliceDims(loc, exchangedLocal, inSharding, mesh,
                                     info.manualAxes, sliceOp.getStartIndices(),
                                     sliceOp.getLimitIndices(), state);
  };

  Value result = haloDataExchange(
      loc, divisibleInput, info.divisibleInputShape, origInputType,
      sliceOp.getType(), inSharding, inSharding, mesh, info.manualAxes,
      paddingValue, edgePaddingLow, info.dimExchanges, postProcess, state);

  // Trim final tensor shape to match expected output shape.
  result = sliceHighSideToShape(loc, rewriter, result, sliceOp.getType(),
                                inSharding);

  rewriter.replaceOp(sliceOp, result);
  return success();
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
  if (!meshOp || meshOp.getMesh().isMaximal()) {
    return;
  }

  ShardingProjection projection =
      ShardingProjection::build(inShardings, outShardings, rule,
                                meshOp.getMesh(), /*closedIfMissing=*/true);
  UpdateTensorShardings update(op->getNumOperands(), op->getNumResults());

  for (int64_t i = 0; i < rule.getNumFactors(); ++i) {
    // When HALO exchange is disabled, we replication-reshard the
    // permutation factors.
    bool isReplicatedFactor = rule.getFactorType(i) == FactorType::kPermutation;
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

    update |= projection.updateSharding(i, /*axes=*/{}, /*overflowAxes=*/{});
  }

  if (update.updateOperands.any() || update.updateResults.any()) {
    insertExplicitReshards(op, inShardings, outShardings, projection, update,
                           state.rewriter, rule, state.symbolTable, meshOp);
  }
}

// -----------------------------------------------------------------------------
// The module pass.
// -----------------------------------------------------------------------------

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
      // Skip terminators and any operations not in the StableHLO dialect.
      // This prevents "unknown op" warnings for Shardy collectives or return
      // ops.
      if (op->hasTrait<OpTrait::IsTerminator>() ||
          !inDialect<stablehlo::StablehloDialect>(op)) {
        return;
      }
      // Reshape op is the only op that doesn't have kPermutation factor but
      // needs HALO exchange.
      if (auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(op)) {
        SDY_CHECK(
            succeeded(handleReshapeOp(reshapeOp, enableHaloExchange, state)));
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
        bool resolved =
            llvm::TypeSwitch<Operation*, bool>(op)
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
  }
};

}  // namespace
}  // namespace sdy
}  // namespace mlir
