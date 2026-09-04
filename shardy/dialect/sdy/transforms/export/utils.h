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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_UTILS_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_UTILS_H_

#include <cstdint>
#include <list>
#include <optional>

#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

constexpr int64_t kCrossPartitionChannelHandleType = 1;

using OptionalAxisRef = std::optional<AxisRefAttr>;

// We use an std::list so we can pop from the front, back, and with a specific
// iterator at constant time.
using AxisList = std::list<AxisRefAttr>;

// Returns a vector of `InnerAxisList` per dimension from the given `sharding`.
template <class InnerAxisList>
SmallVector<InnerAxisList> getAxesPerDim(TensorShardingAttr sharding) {
  SmallVector<InnerAxisList> axesPerDim;
  axesPerDim.reserve(sharding.getRank());
  for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
    axesPerDim.emplace_back(dimSharding.axis_begin(), dimSharding.axis_end());
  }
  return axesPerDim;
}

// In case an axis A in `axes` overlaps but isn't equal to an axis B in
// `orderedOtherAxes`, decomposes A into 1-3 sub-axes (overlap and
// non-overlapping prefix and suffix), and replaces A with the decomposed
// sub-axes that form it.
void alignSubAxesByDecomposition(AxisList& axes,
                                 ArrayRef<AxisRefAttr> orderedOtherAxes,
                                 MeshAttr mesh);

// For every dimension d, calls
// `alignSubAxesByDecomposition(axesPerDim[d], orderedOtherAxes, mesh)`.
void alignSubAxesByDecomposition(SmallVector<AxisList>& axesPerDim,
                                 ArrayRef<AxisRefAttr> orderedOtherAxes,
                                 MeshAttr mesh);

// In case two `AxisRefAttr` in `inAxesPerDim` and `outAxesPerDim` respectively
// overlap but aren't equal, decomposes them into up to three sub-axes (overlap
// and non-overlapping prefix and suffix), and replaces each original axis with
// the decomposed sub-axes that form it.
void alignSubAxesByDecomposition(SmallVector<AxisList>& inAxesPerDim,
                                 SmallVector<AxisList>& outAxesPerDim,
                                 MeshAttr mesh);

// Returns a copy of `sharding` with its dimension axes updated to `axesPerDim`.
// Preserves all other properties of the sharding (replicated axes, unreduced
// axes, reduction op, and per-dimension priorities/closedness).
TensorShardingAttr updateSharding(TensorShardingAttr sharding,
                                  ArrayRef<AxisList> axesPerDim);

// Returns true if the slice operation on the given dimension is
// "communication-free". A slice is communication-free if it is not sharded, or
// if it is sharded but the slice starts at index 0, has a stride of 1, and the
// reduction in size does not cross any sharding boundaries.
//
// This routine assume both the operand and result have the same sharding. In
// such cases, even the shard is not divisible, we can simply pad the operand
// then perform a slice op on each device.
bool isCommunicationFreeSliceDim(int64_t dimIdx, stablehlo::SliceOp sliceOp,
                                 TensorShardingAttr sharding, MeshAttr mesh);

// Returns true if the pad operation on the given dimension is
// "communication-free". A pad is communication-free if it is not sharded, or if
// it is sharded but has no interior padding, and edge padding low and high are
// multiples of the shard size.
bool isCommunicationFreePadDim(int64_t dimIdx, stablehlo::PadOp padOp,
                               TensorShardingAttr sharding, MeshAttr mesh);

// Converts an SDY MeshAttr to a StableHLO MeshAttr.
mlir::stablehlo::MeshAttr convertMeshAttr(MeshAttr sdyMesh);

// Returns true if the device ID should be derived from partition ID (i.e.,
// partitionCount > 1 or replicaCount == 1).
//
// TODO(b/545097355): support replica_count > 1 && partition_count > 1.
bool usePartitionId(int64_t replicaCount, int64_t partitionCount);

// Returns a scalar i64 tensor containing the device ID. Currently, the device
// ID is either replica ID or partition ID, depending on the replica count and
// partition count.
//
// TODO(b/545097355): support replica_count > 1 && partition_count > 1.
Value getDeviceId(int64_t replicaCount, int64_t partitionCount, Location loc,
                  OpBuilder& rewriter);

// Returns a channel handle with handle = nextChannelId++ and type = 1
// (CROSS_PARTITION) if usePartitionId is true. Otherwise returns nullptr (to
// perform cross-replica communication without channel handle).
mlir::stablehlo::ChannelHandleAttr getChannelHandle(MLIRContext* ctx,
                                                    int64_t replicaCount,
                                                    int64_t partitionCount,
                                                    int64_t& nextChannelId);

// Returns the first non-maximal MeshOp found in moduleOp, or nullptr if none
// exists.
MeshOp getGlobalMeshOp(ModuleOp moduleOp);

// Returns the logical index of the shard that the given device (`deviceId`)
// resides in, along a dimension sharded by the provided `axes`.
// This "shard index" ranges is [0, (TotalShardCount - 1)] and identifies
// the device's position in the logical grid formed by the sharding axes.
int64_t getShardIndex(int64_t deviceId, MeshAttr mesh,
                      ArrayRef<AxisRefAttr> axes);

// Returns the padded RankedTensorType where each sharded dimension of `type`
// is padded to be divisible by its shard count along the specified axes (or all
// axes in `sharding` if `allowedAxes` is null). If the type is already
// divisible, not a RankedTensorType, or un-sharded, returns `type`.
Type getDivisiblePaddedType(
    Type type, TensorShardingAttr sharding, const SymbolTable& symbolTable,
    const llvm::DenseSet<StringRef>* allowedAxes = nullptr);

// Returns the FlatSymbolRefAttr for meshOrRef if it is already a symbol
// reference, finds a matching existing MeshOp for an inlined MeshAttr, or
// creates a new MeshOp in the module.
FlatSymbolRefAttr getOrCreateMeshSymbol(Location loc, ModuleOp module,
                                        Attribute meshOrRef,
                                        SymbolTable& symbolTable);

// Same as above, retrieving location and parent ModuleOp from `op`.
FlatSymbolRefAttr getOrCreateMeshSymbol(Operation* op, Attribute meshOrRef,
                                        SymbolTable& symbolTable);

// Converts an SDY AxisRefAttr to a StableHLO AxisRefAttr.
mlir::stablehlo::AxisRefAttr convertAxisRefAttr(AxisRefAttr sdyAxisRef);
// Information about an independent contiguous group of dimensions in a reshape
// whose input and output volumes match. Classifies the sub-transform as
// pass-through, split, or combine to determine communication and padding needs.
struct ReshapeGroupInfo {
  int64_t inStartDim = 0;
  int64_t inLastDim = 0;
  int64_t outStartDim = 0;
  int64_t outLastDim = 0;
  int64_t numInNontrivialDims = 0;
  int64_t numOutNontrivialDims = 0;

  int64_t getInIndivisibleDim() const { return inLastDim - 1; }
  int64_t getOutIndivisibleDim() const { return outLastDim - 1; }

  bool isSplit() const {
    return numInNontrivialDims <= 1 && numOutNontrivialDims > 1;
  }
  bool isCombine() const {
    return numInNontrivialDims > 1 && numOutNontrivialDims <= 1;
  }
  bool isPassthrough() const {
    return numInNontrivialDims <= 1 && numOutNontrivialDims <= 1;
  }
  bool isNeither() const {
    return numInNontrivialDims > 1 && numOutNontrivialDims > 1;
  }

  // Returns whether any dimension in this group is indivisible based on padded
  // shapes.
  bool hasIndivisibility(ArrayRef<int64_t> paddedInputShape,
                         ArrayRef<int64_t> paddedOutputShape,
                         RankedTensorType inType,
                         RankedTensorType outType) const;

  // Returns whether any dimension in this group is indivisible based on
  // sharding.
  bool hasIndivisibility(TensorShardingAttr inSharding,
                         TensorShardingAttr outSharding, MeshAttr mesh,
                         RankedTensorType inType,
                         RankedTensorType outType) const;

  // Returns all axis references across input and output dimensions in this
  // group.
  SmallVector<AxisRefAttr> getAllAxisRefs(TensorShardingAttr inSharding,
                                          TensorShardingAttr outSharding) const;

  // Returns the volume (product of dimension sizes) for input dimensions in
  // this group.
  int64_t getInVolume(ArrayRef<int64_t> shape) const;

  // Returns the volume (product of dimension sizes) for output dimensions in
  // this group.
  int64_t getOutVolume(ArrayRef<int64_t> shape) const;
};

// Builds ReshapeGroupInfo instances based on cumulative shape prefix products.
SmallVector<ReshapeGroupInfo> buildReshapeGroupInfos(RankedTensorType inType,
                                                     RankedTensorType outType);

// Checks whether a reshape operation is communication-free, meaning we can
// just apply high pad to the input and the output is correctly padded without
// any halo exchange or data redistribution.
bool isCommunicationFreeReshape(stablehlo::ReshapeOp reshapeOp,
                                TensorShardingAttr inSharding,
                                TensorShardingAttr outSharding, MeshAttr mesh,
                                RankedTensorType inputType,
                                RankedTensorType outputType,
                                ArrayRef<ReshapeGroupInfo> reshapeGroups);

bool isCommunicationFreeReshape(stablehlo::ReshapeOp reshapeOp,
                                TensorShardingAttr inSharding,
                                TensorShardingAttr outSharding, MeshAttr mesh,
                                RankedTensorType inputType,
                                RankedTensorType outputType);

// Slices the high side of `operand` starting at zero with unit strides to match
// `targetType`. If `operand` already matches `targetType`, returns `operand`.
// Attaches `sharding` to the created slice op if non-null.
Value sliceHighSideToType(OpBuilder& builder, Location loc, Value operand,
                          Type targetType,
                          TensorShardingAttr sharding = nullptr);

// Returns a zero attribute for any ShapedType or scalar Type, including real
// integer, real float, and all ComplexType elements (complex<f32>,
// complex<bf16>, etc.).
Attribute getZeroAttr(OpBuilder& builder, Type type);

// Creates a stablehlo.constant zero matching `type` (scalar or tensor, real or
// complex).
Value createZeroConstant(OpBuilder& builder, Location loc, Type type);

// Zero-pads the high side of `operand` to match `targetType`. If `operand`
// already matches `targetType`, returns `operand`. Attaches `sharding` to the
// created pad op if non-null.
Value padHighSideToType(OpBuilder& builder, Location loc, Value operand,
                        Type targetType, TensorShardingAttr sharding = nullptr,
                        Value paddingValue = nullptr,
                        bool allowSlicePeephole = false);

// Returns the reduction type for a region (e.g., reduce body or scatter update
// computation), or std::nullopt if the region cannot be matched to a supported
// reduction type (SUM, MAX, MIN).
std::optional<ReductionOp> getReductionType(Region& region);

// Returns the reduction type for an operation, or std::nullopt if not
// supported. Defaults to SUM for ops without a region (e.g., Dot, Convolution,
// Gather).
std::optional<ReductionOp> getReductionType(Operation* op);

// Zero-pads an ElementsAttr from `origType` to `paddedType`. Handles splat
// attributes, dense attributes, and all element types (float, int, complex).
ElementsAttr padElementsAttr(ElementsAttr elementsAttr,
                             RankedTensorType origType,
                             RankedTensorType paddedType);
}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_UTILS_H_
