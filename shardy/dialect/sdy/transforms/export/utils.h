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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

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

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_UTILS_H_
