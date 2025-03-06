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
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/op_properties.h"
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

namespace {

bool expandTensorSharding(ShardingProjection& projection, int64_t tensorIndex,
                          int64_t factorIndex, ArrayRef<AxisRefAttr> newAxes) {
  if (tensorIndex < projection.getNumOperands()) {
    return projection.expandOperandSharding(tensorIndex, factorIndex, newAxes);
  }
  return projection.expandResultSharding(
      tensorIndex - projection.getNumOperands(), factorIndex, newAxes);
}

struct TensorIndexSize {
  int64_t index;
  int64_t size;
};

// Given a factor Fi with non-empty new axes, if tensor Tj contains this factor
// and Tj/Fi is a prefix of the new axes, Tj is a source of this new axes.
// Return a vector of source tensor per factor.
SmallVector<TensorIndexSize> getFactorToSourceTensor(
    const ShardingProjection& projection, ArrayRef<int64_t> factorSizes,
    AxesPerFactorRef axesPerFactor) {
  SmallVector<TensorIndexSize> factorToSourceTensor(
      factorSizes.size(), {/*index=*/-1, /*size=*/-1});
  for (const auto& [tensorIndex, tensorFactorShardings] :
       llvm::enumerate(llvm::concat<const TensorFactorShardings>(
           projection.getOperands(), projection.getResults()))) {
    int64_t tensorSize = 1;
    for (const auto& [factorIndex, _] :
         tensorFactorShardings.factorIndexToSharding) {
      tensorSize *= factorSizes[factorIndex];
    }

    for (const auto& [factorIndex, sharding] :
         tensorFactorShardings.factorIndexToSharding) {
      const bool isSource =
          !axesPerFactor[factorIndex].empty() &&
          isAxisListPrefixOf(axesPerFactor[factorIndex], sharding.axisRefs) !=
              PrefixStatus::NOT_A_PREFIX;
      // There may be multiple sources for the same factor. We take the one with
      // largest tensor size.
      TensorIndexSize& sourceTensor = factorToSourceTensor[factorIndex];
      if (isSource && tensorSize > sourceTensor.size) {
        sourceTensor.size = tensorSize;
        sourceTensor.index = tensorIndex;
      }
    }
  }
  return factorToSourceTensor;
}

// Returns if `factorSharding` has a factor at `factorIndex` which is the
// strict prefix of `shardingAxes`.
bool isStrictPrefixOfFactorSharding(
    const TensorFactorShardings& factorSharding, int64_t factorIndex,
    ArrayRef<AxisRefAttr> shardingAxes) {
  if (auto it = factorSharding.factorIndexToSharding.find(factorIndex);
      it != factorSharding.factorIndexToSharding.end()) {
    return isAxisListPrefixOf(it->getSecond().axisRefs, shardingAxes) ==
        PrefixStatus::STRICT_PREFIX;
  }
  return false;
}

// Only propagate axes to operands that are also present in at least one result.
//
// We want to avoid the following situation which can happen when a
// `sharding_constraint` is added onto the operand during Shardy import:
// ```
// %arg0: [{"a", ?}]
// %arg1: [{?}]
// %0 = add %arg0, %arg1 : [{}]
// ```
// We don't want to do an all-gather on both %arg0 and %arg1 due to "a"
// propagating sideways. Instead with the code below, since "a" can't
// propagate to `%0`, we will only do an all-gather on %arg0.
//
// TODO(b/396642774): Long term we should undo this and allow sideways
// propagation, but have our explicit reshard pass make sure the result is
// all-gathered instead of both operands.
void cancelSidewaysPropagationForElementwise(ShardingProjection& projection,
                                             int64_t factorIndex,
                                             SmallVector<AxisRefAttr>& newAxes,
                                             Operation* op) {
  if (!op || !isElementwise(op)) {
    return;
  }
  for (const TensorFactorShardings& result : projection.getResults()) {
    if (isStrictPrefixOfFactorSharding(result, factorIndex, newAxes)) {
      newAxes = result.factorIndexToSharding.at(factorIndex).axisRefs;
    }
  }
}

}  // namespace

SmallVector<AxisRefAttr>
AggressiveFactorPropagation::getPropagatedFactorSharding(
    int64_t factorIndex, const TensorFactorShardings& tensorFactorShardings,
    const FactorIndexToSharding& factorIndexToSharding,
    AxesPerFactorRef axesPerFactor, MeshAttr mesh, bool conservativePropagation,
    ArrayRef<int64_t> factorSizes) const {
  auto factorShardingIt = factorIndexToSharding.find(factorIndex);
  if (factorShardingIt == factorIndexToSharding.end()) {
    return {};
  }
  const FactorSharding& factorSharding = factorShardingIt->second;
  SmallVector<AxisRefAttr> newAxes = axesPerFactor[factorIndex];

  // Resolve conflicts within a factor.
  truncateAxesByRemovingConflicts(
      newAxes,
      [&](AxisRefAttr axisRef, int64_t prevShardedSize) {
        return compatiblePrefixNoConflictsWithinFactor(
            axisRef, tensorFactorShardings.replicatedAxes, factorSharding,
            prevShardedSize, factorSizes[factorIndex], mesh);
      },
      mesh, conservativePropagation);
  if (!isStrictPrefix(factorSharding.axisRefs, newAxes)) {
    return {};
  }

  // Resolve conflicts (overlapping sharding axes) between factors.
  //
  // Note that we pass `factorIndexToSharding`, which might have been
  // updated for a previous factor (previous iteration), thus we are
  // checking for conflicts w.r.t. the updated state of this tensor.
  truncateAxesByRemovingConflicts(
      newAxes,
      [&](AxisRefAttr axisRef, int64_t) {
        return compatiblePrefixNoConflictsAcrossFactors(
            axisRef, factorIndexToSharding, factorIndex);
      },
      mesh, conservativePropagation);

  return newAxes;
}

UpdateTensorShardings AggressiveFactorPropagation::propagateFactorShardings(
    ShardingProjection& projection,
    PropagationDirectionAlongFactor directionAlongFactor,
    ArrayRef<int64_t> factorSizes, MeshAttr mesh, Operation* op,
    bool conservativePropagation) const {
  UpdateTensorShardings result(projection.getNumOperands(),
                               projection.getNumResults());

  // Find the compatible major axes ignoring conflicts.
  AxesPerFactor axesPerFactor;
  axesPerFactor.reserve(factorSizes.size());
  bool allElementsAreEmpty = true;
  for (int64_t i = 0; i < factorSizes.size(); ++i) {
    SmallVector<AxisRefAttr>& axes = axesPerFactor.emplace_back(
        getCompatibleMajorAxes(projection, i, directionAlongFactor(i), op));
    if (!axes.empty()) {
      allElementsAreEmpty = false;
    }
  }
  if (allElementsAreEmpty) {
    return result;
  }

  // We sort the factors based on:
  // 1. larger source tensor size first
  // 2. smaller source tensor index first
  // 3. smaller factor index first
  // Unstable sort is fine because there is no equality in the candidates.
  // TODO(b/376233527): reevaluate this conflict resolution heuristic.
  SmallVector<int64_t> sortedFactorIndices =
      llvm::to_vector(llvm::seq<int64_t>(0, factorSizes.size()));
  SmallVector<TensorIndexSize> factorToSourceTensor =
      getFactorToSourceTensor(projection, factorSizes, axesPerFactor);
  llvm::sort(sortedFactorIndices, [&](int64_t i, int64_t j) {
    return std::forward_as_tuple(-factorToSourceTensor[i].size,
                                 factorToSourceTensor[i].index, i) <
           std::forward_as_tuple(-factorToSourceTensor[j].size,
                                 factorToSourceTensor[j].index, j);
  });

  // Propagate the axes got in Step 1, resolving conflicts between factors by
  // following the order of preference in  `sortedFactorIndices`.
  for (int64_t factorIndex : sortedFactorIndices) {
    PropagationDirection direction = directionAlongFactor(factorIndex);
    // 1. Propagate to results.
    //
    // We don't propagate sideways between results in backwards propagation,
    // so the sharding of the results along this factor shouldn't change.
    if (direction != PropagationDirection::BACKWARD) {
      for (const auto& [tensorIndex, tensorFactorShardings] :
           llvm::enumerate(projection.getResults())) {
        SmallVector<AxisRefAttr> newAxes = getPropagatedFactorSharding(
            factorIndex, tensorFactorShardings,
            tensorFactorShardings.factorIndexToSharding, axesPerFactor, mesh,
            conservativePropagation, factorSizes);
        if (newAxes.empty()) {
          continue;
        }
        if (expandTensorSharding(projection,
                                 tensorIndex + projection.getNumOperands(),
                                 factorIndex, newAxes)) {
          result.updateResults.set(tensorIndex);
        }
      }
    }

    // 2. Propagate to operands.
    //
    // We don't propagate sideways between operands in forward propagation,
    // so the sharding of the operands along this factor shouldn't change.
    if (direction != PropagationDirection::FORWARD) {
      for (const auto& [tensorIndex, tensorFactorShardings] :
           llvm::enumerate(projection.getOperands())) {
        SmallVector<AxisRefAttr> newAxes = getPropagatedFactorSharding(
            factorIndex, tensorFactorShardings,
            tensorFactorShardings.factorIndexToSharding, axesPerFactor, mesh,
            conservativePropagation, factorSizes);
        if (newAxes.empty()) {
          continue;
        }
        cancelSidewaysPropagationForElementwise(projection, factorIndex,
                                                newAxes, op);
        if (expandTensorSharding(projection, tensorIndex, factorIndex,
                                 newAxes)) {
          result.updateOperands.set(tensorIndex);
        }
      }
    }
  }
  return result;
}

}  // namespace sdy
}  // namespace mlir
