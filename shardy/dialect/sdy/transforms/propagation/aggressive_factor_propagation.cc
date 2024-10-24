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
#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"

namespace mlir {
namespace sdy {

namespace {

bool updateTensorSharding(ShardingProjection& projection, int64_t tensorIndex,
                          int64_t factorIndex, ArrayRef<AxisRefAttr> newAxes) {
  if (tensorIndex < projection.getNumOperands()) {
    return projection.updateOperandSharding(tensorIndex, factorIndex, newAxes);
  }
  return projection.updateResultSharding(
      tensorIndex - projection.getNumOperands(), factorIndex, newAxes);
}

}  // namespace

UpdateTensorShardings AggressiveFactorPropagation::propagateFactorShardings(
    ShardingProjection& projection, PropagationDirection direction,
    ArrayRef<int64_t> factorSizes, MeshAttr mesh, Operation* op,
    bool conservativePropagation) const {
  UpdateTensorShardings result(projection.getNumOperands(),
                               projection.getNumResults());
  if (direction == PropagationDirection::NONE) {
    return result;
  }

  // Find the compatible major axes ignoring conflicts.
  SmallVector<SmallVector<AxisRefAttr>> axesPerFactor;
  axesPerFactor.reserve(factorSizes.size());
  bool allElementsAreEmpty = true;
  for (int64_t i = 0; i < factorSizes.size(); ++i) {
    SmallVector<AxisRefAttr>& axes = axesPerFactor.emplace_back(
        getCompatibleMajorAxes(projection, i, direction, op));
    if (!axes.empty()) {
      allElementsAreEmpty = false;
    }
  }
  if (allElementsAreEmpty) {
    return result;
  }

  // Given a factor with non-empty new axes, if we cannot propagate the new axes
  // to a tensor that contains this factor, this tensor must be a source of this
  // new axes. We collect a map from factor index to the size of the larget
  // source tensor.
  SmallVector<int64_t> factorToTensorSize(factorSizes.size(), 1);
  for (const auto& tensorFactorShardings :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    int64_t tensorSize = 1;
    for (const auto& [factorIndex, _] :
         tensorFactorShardings.factorIndexToSharding) {
      tensorSize *= factorSizes[factorIndex];
    }

    for (const auto& [factorIndex, sharding] :
         tensorFactorShardings.factorIndexToSharding) {
      if (!axesPerFactor[factorIndex].empty() &&
          !shouldUpdate(sharding.axisRefs, axesPerFactor[factorIndex]) &&
          factorToTensorSize[factorIndex] < tensorSize) {
        factorToTensorSize[factorIndex] = tensorSize;
      }
    }
  }

  // The propagation on each tensor is independent. This strategy can propagate
  // different shardings to different tensors along the same factor. Examples
  // are provided in the docstring of this class.
  for (const auto& [tensorIndex, tensorFactorShardings] :
       llvm::enumerate(llvm::concat<const TensorFactorShardings>(
           projection.getOperands(), projection.getResults()))) {
    // Propagate the axes got in Step 1, and resolve conflicts within a factor.
    FactorIndexToSharding newSharding =
        tensorFactorShardings.factorIndexToSharding;
    BitVector factorUpdated(factorSizes.size());
    for (auto& [factorIndex, factorSharding] : newSharding) {
      SmallVector<AxisRefAttr> newAxes = axesPerFactor[factorIndex];
      truncateAxesByRemovingConflicts(
          newAxes,
          [&, factorIndex = factorIndex, &factorSharding = factorSharding,
           &tensorFactorShardings = tensorFactorShardings](
              AxisRefAttr axisRef, int64_t prevShardedSize) {
            return compatiblePrefixNoConflictsWithinFactor(
                axisRef, tensorFactorShardings.replicatedAxes, factorSharding,
                prevShardedSize, factorSizes[factorIndex], mesh);
          },
          mesh, conservativePropagation);
      if (shouldUpdate(factorSharding.axisRefs, newAxes)) {
        factorSharding.axisRefs = newAxes;
        factorUpdated.set(factorIndex);
      }
    }

    SmallVector<int> factorIndices = toSetBitsVector(factorUpdated);
    // Unstable sort is acceptable since there is no equality in the tuple.
    llvm::sort(factorIndices, [&](int64_t i, int64_t j) {
      return std::forward_as_tuple(factorToTensorSize[i], -i) >
             std::forward_as_tuple(factorToTensorSize[j], -j);
    });

    // Resolve conflicts (overlapping sharding axes) between factors.
    bool tensorUpdated = false;
    for (const int64_t factorIndex : factorIndices) {
      SmallVector<AxisRefAttr> newAxes = newSharding[factorIndex].axisRefs;
      truncateAxesByRemovingConflicts(
          newAxes,
          [&, factorIndex = factorIndex](AxisRefAttr axisRef, int64_t) {
            return compatiblePrefixNoConflictsAcrossFactors(
                axisRef, newSharding, factorIndex);
          },
          mesh, conservativePropagation);
      tensorUpdated |=
          updateTensorSharding(projection, tensorIndex, factorIndex, newAxes);
    }

    if (tensorIndex < projection.getNumOperands()) {
      result.updateOperands[tensorIndex] = tensorUpdated;
    } else {
      result.updateResults[tensorIndex - projection.getNumOperands()] =
          tensorUpdated;
    }
  }

  return result;
}

}  // namespace sdy
}  // namespace mlir
