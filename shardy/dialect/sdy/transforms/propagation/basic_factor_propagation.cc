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

#include "shardy/dialect/sdy/transforms/propagation/basic_factor_propagation.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <tuple>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/common/macros.h"
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

namespace {

// Returns the largest prefix of `axisRef` that does not overlap with any axes
// in `otherAxisRefs`.
std::optional<AxisRefAttr> getPrefixWithoutOverlap(
    AxisRefAttr axisRef, ArrayRef<AxisRefAttr> otherAxisRefs) {
  AxisRefAttr result = axisRef;
  for (AxisRefAttr otherAxisRef : otherAxisRefs) {
    ASSIGN_OR_RETURN_IF_NULLOPT(result,
                                result.getPrefixWithoutOverlap(otherAxisRef));
  }
  return result;
}

}  // namespace

std::optional<AxisRefAttr>
BasicFactorPropagation::compatiblePrefixNoConflictsAcrossFactors(
    AxisRefAttr axisRef, const FactorIndexToSharding& factorIndexToSharding,
    int64_t factorIndex) const {
  AxisRefAttr result = axisRef;
  for (const auto& [otherFactorIndex, shardings] : factorIndexToSharding) {
    if (otherFactorIndex != factorIndex) {
      ASSIGN_OR_RETURN_IF_NULLOPT(
          result, getPrefixWithoutOverlap(result, shardings.overflowAxes));
      ASSIGN_OR_RETURN_IF_NULLOPT(
          result, getPrefixWithoutOverlap(result, shardings.axisRefs));
    }
  }
  return result;
}

std::optional<AxisRefAttr>
BasicFactorPropagation::compatiblePrefixNoConflictsWithinFactor(
    AxisRefAttr axisRef, ArrayRef<AxisRefAttr> replicatedAxes,
    const FactorSharding& factorSharding, int64_t shardedSize,
    int64_t factorSize) const {
  AxisRefAttr result = axisRef;

  ASSIGN_OR_RETURN_IF_NULLOPT(result,
                              getPrefixWithoutOverlap(result, replicatedAxes));

  ArrayRef<AxisRefAttr> factorAxes = factorSharding.axisRefs;
  if (llvm::any_of(factorAxes, [&](AxisRefAttr shardingAxis) {
        return shardingAxis.contains(result);
      })) {
    // `result` is already contained in the corresponding factor shardings.
    return result;
  }

  // `factorAxes` does not contain `result`, indicating that we will expand
  // `factorAxes` with `result`. It is feasible only if the factor:
  // - is open.
  // - does not have overflow axes.
  // - is minor-most or `factorSize` is divisible by `shardedSize`.
  if (!factorSharding.isClosed && factorSharding.overflowAxes.empty() &&
      (factorSharding.isMinorMost || factorSize % shardedSize == 0)) {
    return result;
  }

  // We cannot propagate the full `result` to `factorAxes`. We can still keep
  // `factorAxes.back()` if it is a prefix of `result`. Since `factorAxes` and
  // compatible axes share the same prefix, we only need to handle the last
  // element of `factorAxes`.
  if (!factorAxes.empty() && factorAxes.back().prefixOf(result)) {
    return factorAxes.back();
  }

  return std::nullopt;
}

void BasicFactorPropagation::truncateAxesByRemovingConflicts(
    SmallVector<AxisRefAttr>& axes,
    std::function<std::optional<AxisRefAttr>(AxisRefAttr curAxis,
                                             int64_t shardedSize)>
        removeConflicts,
    MeshAttr mesh, bool conservativePropagation) const {
  int64_t shardedSize = 1;
  for (const auto [axisIndex, curAxis] : llvm::enumerate(axes)) {
    // This check is only for tests. For convenience we can pass a `MeshAttr()`
    // to avoid the divisibility constraint.
    if (mesh) {
      shardedSize *= curAxis.getSize(mesh);
    }

    std::optional<AxisRefAttr> newAxis = removeConflicts(curAxis, shardedSize);
    if (!newAxis || (conservativePropagation && newAxis->getSubAxisInfo())) {
      axes.truncate(axisIndex);
      return;
    }
    if (axes[axisIndex] != *newAxis) {
      axes[axisIndex] = *newAxis;
      axes.truncate(axisIndex + 1);
      return;
    }
  }
}

namespace {

using DirectionBasedTensorShardings =
    std::pair<ArrayRef<TensorFactorShardings>, ArrayRef<TensorFactorShardings>>;

// Gets the tensor shardings that should be processed first and then second.
//
// If we're propagating in one direction, we want to only allow a subset of
// the factor shardings to expand the compatible sharding axes.
//
// For the forwards direction, we only want to allow the operand factor
// shardings to further expand the compatible sharding axes, but not allow the
// result factor shardings to expand the compatible sharding axes. We still want
// to consider the result factor shardings in case they have conflicting
// axes - we just don't want to add more axes if there is a compatible one.
//
// The same holds for backwards propagation, except we allow expansion based
// on the result factor shardings but not the operands.
std::optional<DirectionBasedTensorShardings> getDirectionBasedTensorShardings(
    PropagationDirection direction, Operation* op,
    ArrayRef<TensorFactorShardings> operands,
    ArrayRef<TensorFactorShardings> results) {
  static const char* errMsg =
      "since Shardy is propagating {0} for this op, Shardy may not "
      "fully propagate to each of the multiple {1}s; {0} "
      "propagation was designed with single {1} ops in mind. Let the "
      "Shardy team know the operation that you'd like to be fully "
      "supported.";
  static llvm::once_flag flag;
  switch (direction) {
    case PropagationDirection::BOTH:
      return std::make_pair(operands, results);
    case PropagationDirection::FORWARD: {
      if (op && results.size() > 1) {
        emitOpWarningOnce(flag, op,
                          llvm::formatv(errMsg, "forward", "result").str());
      }
      return std::make_pair(operands, results);
    }
    case PropagationDirection::BACKWARD: {
      if (op && operands.size() > 1) {
        emitOpWarningOnce(flag, op,
                          llvm::formatv(errMsg, "backward", "operand").str());
      }
      return std::make_pair(results, operands);
    }
    case PropagationDirection::NONE:
      return std::nullopt;
  }
}

// Returns all compatible major axes between `oldAxes` and `newAxes`,
// and whether the new list can be expanded (i.e., there is no conflict and
// `canExpand` is true).
//
// If `canExpand` is false, the result would be at most equal to `oldAxes`.
//
// For example:
// * If `canExpand` is true:
//   - Given axes ["a"] and ["a","b"] returns ["a","b"] and true.
//   - Given axes ["a", "b"] and ["a", "c"] returns ["a"] and false.
//   - Given axes ["a":(1)2] and ["a":(1)4] returns ["a":(1)4] and true.
//   - Given axes ["a":(1)2, "b"] and ["a":(1)4, "b"] returns ["a":(1)2] and
//     false.
// * If `canExpand` is false:
//   - Given axes ["a"] and ["a"] returns ["a"] and false.
//   - Given axes ["a"] and ["a", "b"] returns ["a"] and false.
//   - Given axes ["a", "b"] and ["a"] returns ["a", "b"] and false.
//   - Given axes ["a":(1)2] and ["a":(1)4] returns ["a":(1)2] and false.
std::pair<SmallVector<AxisRefAttr>, bool> getCompatibleMajorAxesInternal(
    ArrayRef<AxisRefAttr> oldAxes, ArrayRef<AxisRefAttr> newAxes,
    bool canExpand) {
  SmallVector<AxisRefAttr> result;
  result.reserve(std::max(oldAxes.size(), newAxes.size()));

  while (!oldAxes.empty() && !newAxes.empty()) {
    AxisRefAttr oldAxisRef = oldAxes.front();
    AxisRefAttr newAxisRef = newAxes.front();
    oldAxes = oldAxes.drop_front();
    newAxes = newAxes.drop_front();

    if (newAxisRef.getName() != oldAxisRef.getName()) {
      // Axis names don't match, stop.
      return {result, false};
    }

    if (newAxisRef.getSubAxisInfo() == oldAxisRef.getSubAxisInfo()) {
      // Same axis or sub-axis, add the axis and continue.
      result.push_back(newAxisRef);
      continue;
    }

    if (newAxisRef.prefixOf(oldAxisRef)) {
      // The new axis is a sub axis which is a prefix of the old axis, e.g. axes
      // "a":(2)4 and "a":(2)2.
      if (!newAxes.empty()) {
        // We assume that if the new list has another axis, it would be
        // conflicting. It is not conflicting iff it's the next consecutive
        // sub-axis. However, this shouldn't happen as that would mean those
        // two sub-axes could be merged. Therefore, we add the new axis, stop,
        // and block the new result from being expanded.
        // For example: ["a":(2)4, ...] and ["a":(2)2, "b"].
        result.push_back(newAxisRef);
        return {result, false};
      }
      // Otherwise, we can expand the new list of axes, therefore add the old
      // axis and continue.
      result.push_back(oldAxisRef);
    } else if (oldAxisRef.prefixOf(newAxisRef)) {
      // The old axis is a sub axis which is a prefix of the new axis e.g. axes
      // "a":(2)2 and "a":(2)4.
      if (!canExpand || !oldAxes.empty()) {
        // If `canExpand` is false, we can't expand the old list of axes with
        // the new axis, otherwise if the old list of axes has another axis, we
        // assume it would be conflicting (similar to above). Therefore, we add
        // the old axis, stop, and block the new result from being expanded.
        // For example: ["a":(2)2, "b"] and ["a":(2)4, ...].
        result.push_back(oldAxisRef);
        return {result, false};
      }
      // Otherwise, we can expand the old list of axes, add the new tensor and
      // continue;
      result.push_back(newAxisRef);
    } else {
      // The axes are conflicting, we will stop and block the new result from
      // being expanded.
      // For example: axes "a":(1)2 and "a":(2)2.
      return {result, false};
    }
  }

  // If we got here it means the two lists are not conflicting, so we need to
  // add the remaining old axes.
  llvm::copy(oldAxes, std::back_inserter(result));

  // If we can expand the old list of axes and there are additional new axes,
  // append all of them to the result.
  if (canExpand) {
    llvm::copy(newAxes, std::back_inserter(result));
  }

  return {result, canExpand};
}

}  // namespace

SmallVector<AxisRefAttr> BasicFactorPropagation::getCompatibleMajorAxes(
    const ShardingProjection& projection, int64_t factorIndex,
    PropagationDirection direction, Operation* op) const {
  std::optional<DirectionBasedTensorShardings> tensorShardings =
      getDirectionBasedTensorShardings(direction, op, projection.getOperands(),
                                       projection.getResults());
  assert(tensorShardings.has_value());

  SmallVector<AxisRefAttr> resultAxes;
  bool canExpand = true;

  auto updateCompatibleMajorAxesWithTensors =
      [&](ArrayRef<TensorFactorShardings> tensors) {
        for (const TensorFactorShardings& tensor : tensors) {
          if (auto factorShardingIt =
                  tensor.factorIndexToSharding.find(factorIndex);
              factorShardingIt != tensor.factorIndexToSharding.end()) {
            std::tie(resultAxes, canExpand) = getCompatibleMajorAxesInternal(
                resultAxes, factorShardingIt->second.axisRefs, canExpand);
          }
        }
      };

  updateCompatibleMajorAxesWithTensors(tensorShardings->first);
  if (direction != PropagationDirection::BOTH) {
    // For `PropagationDirection::FORWARD/BACKWARD`, `tensorShardings.first` is
    // allowed to expand the sharding axes, while `tensorShardings.second` is
    // not. `tensorShardings.second` is only used to account for conflicts.
    canExpand = false;
  }
  updateCompatibleMajorAxesWithTensors(tensorShardings->second);
  return resultAxes;
}

std::optional<AxisRefAttr> BasicFactorPropagation::compatiblePrefix(
    AxisRefAttr axisRef, const TensorFactorShardings& tensorFactorSharding,
    int64_t factorIndex, int64_t shardedSize, int64_t factorSize) const {
  const FactorIndexToSharding& factorIndexToSharding =
      tensorFactorSharding.factorIndexToSharding;

  ASSIGN_OR_RETURN_IF_NULLOPT(AxisRefAttr result,
                              compatiblePrefixNoConflictsAcrossFactors(
                                  axisRef, factorIndexToSharding, factorIndex));

  auto factorShardingIt = factorIndexToSharding.find(factorIndex);
  if (factorShardingIt == factorIndexToSharding.end()) {
    // This tensor does not contain the factor at `factorIndex`. We can not
    // propagate `axisRef` to this tensor at `factorIndex`. Hence, the only
    // conflict is the overlap between `axisRef` and other factor shardings. The
    // overlap between `axisRef` and (implicitly or explicitly) replicated axes
    // does not trigger conflicts.
    return result;
  }

  // This tensor contains the factor at `factorIndex`. We remove conflicts
  // within the factor.
  return compatiblePrefixNoConflictsWithinFactor(
      result, tensorFactorSharding.replicatedAxes, factorShardingIt->second,
      shardedSize, factorSize);
}

std::optional<AxisRefAttr> BasicFactorPropagation::compatiblePrefix(
    AxisRefAttr axisRef, const ShardingProjection& projection,
    int64_t factorIndex, int64_t shardedSize, int64_t factorSize) const {
  AxisRefAttr result = axisRef;
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    ASSIGN_OR_RETURN_IF_NULLOPT(
        result, compatiblePrefix(result, tensorFactorSharding, factorIndex,
                                 shardedSize, factorSize));
  }
  return result;
}

SmallVector<AxisRefAttr> BasicFactorPropagation::getCompatibleMajorShardingAxes(
    const ShardingProjection& projection, int64_t factorIndex,
    PropagationDirection direction, int64_t factorSize, MeshAttr mesh,
    Operation* op, bool conservativePropagation) const {
  if (direction == PropagationDirection::NONE) {
    return SmallVector<AxisRefAttr>();
  }

  // Finds the compatible major axes ignoring conflicts.
  SmallVector<AxisRefAttr> resultAxes =
      getCompatibleMajorAxes(projection, factorIndex, direction, op);

  // Removes the major-most axis that isn't compatible w.r.t. other factors or
  // the replicated axes, and all axes that are minor to it.
  truncateAxesByRemovingConflicts(
      resultAxes,
      [&](AxisRefAttr axisRef, int64_t shardedSize) {
        return compatiblePrefix(axisRef, projection, factorIndex, shardedSize,
                                factorSize);
      },
      mesh, conservativePropagation);

  return resultAxes;
}

UpdateTensorShardings BasicFactorPropagation::propagateFactorShardings(
    ShardingProjection& projection, PropagationDirection direction,
    ArrayRef<int64_t> factorSizes, MeshAttr mesh, Operation* op,
    bool conservativePropagation) const {
  UpdateTensorShardings result_2{
      .updateOperands = BitVector(projection.getNumOperands()),
      .updateResults = BitVector(projection.getNumResults())};

  // We propagate each factor separately.
  for (auto [factorIndex, factorSize] : llvm::enumerate(factorSizes)) {
    // For each factor, find the compatible major sharding axes that can shard
    // that factor for all tensors, those are the axes we will propagate to
    // tensors that aren't already sharded.
    SmallVector<AxisRefAttr> axesToPropagate = getCompatibleMajorShardingAxes(
        projection, factorIndex, direction, factorSize, mesh, op,
        conservativePropagation);

    // Update all shardings along this factor if possible.
    auto [updateOperandForFactor, updateResultForFactor] =
        projection.updateSharding(factorIndex, axesToPropagate);

    result_2.updateOperands |= updateOperandForFactor;
    result_2.updateResults |= updateResultForFactor;
  }

  return result_2;
}

}  // namespace sdy
}  // namespace mlir
