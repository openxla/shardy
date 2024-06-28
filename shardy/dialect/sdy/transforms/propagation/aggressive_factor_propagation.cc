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

#include "shardy/dialect/sdy/transforms/propagation/aggressive_factor_propagation.h"

#include <cassert>
#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

AxesPerFactor
AggressiveFactorPropagation::getCompatibleMajorShardingAxesForAllFactors(
    const ShardingProjection& projection, PropagationDirection direction,
    ArrayRef<int64_t> factorSizes, MeshAttr mesh, Operation* op,
    bool conservativePropagation) const {
  if (direction == PropagationDirection::NONE) {
    return AxesPerFactor(factorSizes.size());
  }

  // Finds the compatible major axes ignoring conflicts.
  AxesPerFactor result;
  result.reserve(factorSizes.size());
  for (int64_t i = 0; i < factorSizes.size(); ++i) {
    result.push_back(getCompatibleMajorAxes(projection, i, direction, op));
  }

  // Removes the conflicts within every single factor. This strategy and
  // `BasicFactorPropagation` handles conflicts within a factor in the same way.
  for (const TensorFactorShardings& tensorFactorShardings :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorShardings.factorIndexToSharding) {
      truncateAxesByRemovingConflicts(
          result[factorIndex],
          [&](AxisRefAttr axisRef, int64_t shardedSize) {
            return compatiblePrefixNoConflictsWithinFactor(
                axisRef, tensorFactorShardings.replicatedAxes, factorSharding,
                shardedSize, factorSizes[factorIndex]);
          },
          mesh);
    }
  }

  // Removes the conflicts across factors, where this strategy and
  // `BasicFactorPropagation` diverge.
  //
  // With `BasicFactorPropagation`, the compatible axes of a factor Fi cannot
  // overlap with the existing sharding axes or the overflow axes related to all
  // other factors. This criterion is considered for all tensors, no matter if
  // Fi is mapped to the tensor or not. The table below shows the criterion:
  //
  //                  existing sharding axes & overflow axes   new sharding axes
  // factor in tensor              remove overlap                      -
  // factor not in tensor          remove overlap                      -
  //
  // On the contrary, `AggressiveFactorPropagation` has the following criterion:
  //
  //                  existing sharding axes & overflow axes   new sharding axes
  // factor in tensor              remove overlap               remove overlap
  // factor not in tensor                 -                            -
  //
  // There are two differences:
  //
  // 1. `BasicFactorPropagation` removes the overlap between the compatible axes
  // of a factor Fi with the existing sharding axes and overflow axes in a
  // tensor Tj even if Fi is not in Tj. `AggressiveFactorPropagation` does not
  // remove this overlap if Fi is not in Tj. `BasicFactorPropagation` is too
  // strict, since we cannot propagate sharding axes to Tj along Fi.
  //
  // `AggressiveFactorPropagation` cannot handle the following case if we only
  // have difference #1. `-` means that the factor is not mapped to the tensor.
  // After removing conflicts within factors, we will propagate "x" to T2 along
  // F0 and F1 at the same time, which induces a conflict. To resolve this
  // conflict, we have difference #2.
  //
  //     F0   F1
  // T0  "x"   -
  // T1   -   "x"
  // T2   ?    ?
  //
  // 2. `AggressiveFactorPropagation` removes the overlap between compatible
  // axes of a factor Fi with the potential new sharding axes of other factors
  // in Tj if Fi is in Tj. Thus, it is safe to propagate the axes to Tj along Fi
  // without conflicts with other factors. In the example, we will not propagate
  // "x" along F0 or F1 since their potential new sharding axes overlap.
  //
  // The potential new sharding axes are saved in `resultSnapshot`. It is a hard
  // copy since we need to handle the following case.
  //
  //     F0   F1   F2
  // T0  "x"   -    -
  // T1   -   "x"   -
  // T2   -    -   "x"
  // T3   ?    ?    ?
  //
  // The `result` and `resultSnapshot` is [["x"], ["x"], ["x"]] before removing
  // conflicts across factors. After removing conflicts between F0/F1 and other
  // factors, `result` is [[], [], ["x"]]. When we remove conflicts between F2
  // and other factors, if we use `result` as the potential new sharding axes,
  // we will not remove "x" for F2 because it is no longer present in 'result'
  // for F0 and F1. We have to use `resultSnapshot` to save the potential new
  // sharding axes and remove "x" for F2.
  const AxesPerFactor resultSnapshot = result;
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      truncateAxesByRemovingConflicts(
          result[factorIndex],
          [&](AxisRefAttr axisRef, int64_t) {
            return compatiblePrefixNoConflictsAcrossFactors(
                axisRef, tensorFactorSharding.factorIndexToSharding,
                factorIndex, resultSnapshot);
          },
          mesh);
    }
  }

  return result;
}

}  // namespace sdy
}  // namespace mlir
