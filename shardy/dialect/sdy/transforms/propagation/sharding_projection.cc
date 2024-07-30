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

#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

#include <cassert>
#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

namespace {

// Returns if `oldAxes` should be updated by `newAxes`.
bool shouldUpdate(ArrayRef<AxisRefAttr> oldAxes,
                  ArrayRef<AxisRefAttr> newAxes) {
  if (newAxes.empty()) {
    return false;
  }
  if (newAxes.size() < oldAxes.size()) {
    return false;
  }
  if (newAxes.size() > oldAxes.size()) {
    return true;
  }
  return newAxes.back().strictlyContains(oldAxes.back());
}

}  // namespace

bool TensorFactorShardings::updateShardingAxes(int64_t factorIndex,
                                               ArrayRef<AxisRefAttr> newAxes) {
  auto factorShardingIt = factorIndexToSharding.find(factorIndex);
  if (factorShardingIt == factorIndexToSharding.end()) {
    return false;
  }

  SmallVector<AxisRefAttr>& oldAxes = factorShardingIt->second.axisRefs;
  if (shouldUpdate(oldAxes, newAxes)) {
    oldAxes = llvm::to_vector(newAxes);
    return true;
  }

  return false;
}

namespace {

// Adds all axes in `axes` to `dimSharding`.
//
// If the last axis in `dimSharding` can be merged with the first axis in
// `axes`, adds the merged axis instead of both.
//
// Returns the product of all axes sizes.
int64_t addAxesToDimSharding(SmallVector<AxisRefAttr>& dimSharding,
                             ArrayRef<AxisRefAttr> axes, MeshAttr mesh) {
  int64_t totalSize = 1;
  for (auto [axisIndex, axisRef] : llvm::enumerate(axes)) {
    totalSize *= axisRef.getSize(mesh);
    if (axisIndex == 0 && !dimSharding.empty() &&
        dimSharding.back().canMerge(axisRef)) {
      // Merge consecutive sub-axes at the boundary between two factors
      dimSharding.back() = dimSharding.back().merge(axisRef, mesh);
    } else {
      dimSharding.push_back(axisRef);
    }
  }
  return totalSize;
}

}  // namespace

TensorShardingAttr TensorFactorShardings::createTensorShardingAttr(
    MLIRContext* ctx, TensorMappingAttr tensorMapping,
    ArrayRef<int64_t> factorSizes, StringRef meshName, MeshAttr mesh) const {
  SmallVector<DimensionShardingAttr> newDimShardings;
  newDimShardings.reserve(tensorMapping.getRank());

  for (DimMappingAttr dimMapping : tensorMapping.getDimMappings()) {
    bool isClosed = false;
    SmallVector<AxisRefAttr> dimSharding;
    for (int64_t factorIndex : dimMapping.getFactorIndices()) {
      int64_t factorSize = factorSizes[factorIndex];
      const FactorSharding& factorSharding =
          factorIndexToSharding.at(factorIndex);
      isClosed |= factorSharding.isClosed;

      int64_t shardedSize =
          addAxesToDimSharding(dimSharding, factorSharding.axisRefs, mesh);

      if (!factorSharding.overflowAxes.empty()) {
        // If this factor has overflow axes, that means any subsequent factor
        // should be ignored, so we add the overflow axes to the dimension and
        // move to the next dimension.
        (void)addAxesToDimSharding(dimSharding, factorSharding.overflowAxes,
                                   mesh);
        break;
      }

      // The following assertion holds because we wouldn't have propagated the
      // non-divisible axis otherwise.
      assert(dimMapping.isMinorMost(factorIndex) ||
             factorSize % shardedSize == 0 &&
                 "non-minor-most factor must be divisible by axis sizes");
      if (shardedSize < factorSize) {
        // Any subsequent factor will require strided view, add the axes up to
        // this factor (including) to this dimension sharding and move to the
        // next dimension.
        break;
      }
    }
    // If this dimension is fully sharded, we mark it as closed since it can't
    // be further sharded.
    newDimShardings.push_back(
        DimensionShardingAttr::get(ctx, dimSharding, isClosed));
  }

  return TensorShardingAttr::get(ctx, meshName, newDimShardings,
                                 replicatedAxes);
}

UpdateShardings ShardingProjection::updateSharding(
    int64_t factorIndex, ArrayRef<AxisRefAttr> newAxes) {
  UpdateShardings result{/* .updateOperands = */ BitVector(getNumOperands()),
                         /* .updateResults = */ BitVector(getNumResults())};
  for (auto [i, tensor] : llvm::enumerate(operands)) {
    result.updateOperands[i] = tensor.updateShardingAxes(factorIndex, newAxes);
  }
  for (auto [i, tensor] : llvm::enumerate(results)) {
    result.updateResults[i] = tensor.updateShardingAxes(factorIndex, newAxes);
  }
  return result;
}

namespace {

// Holds the size of an axis ref, and its pre-size if it's a sub-axis.
struct AxisRefInfo {
  int64_t size;
  std::optional<int64_t> splitPreSize = std::nullopt;

  SubAxisInfoAttr getSubAxisInfo(MLIRContext* ctx) const {
    return splitPreSize ? SubAxisInfoAttr::get(ctx, *splitPreSize, size)
                        : SubAxisInfoAttr();
  }
};

// Returns the size of the axis with the specified `axisIndex` and its pre-size
// if it's a sub-axis, or `std::nullopt` if `axisIndex` is out of bounds.
std::optional<AxisRefInfo> getAxisRefInfo(ArrayRef<AxisRefAttr> axes,
                                          int64_t axisIndex, MeshAttr mesh) {
  if (axisIndex >= axes.size()) {
    return std::nullopt;
  }
  AxisRefAttr axisRef = axes[axisIndex];
  SubAxisInfoAttr splitInfo = axisRef.getSubAxisInfo();
  return AxisRefInfo{
      /* .size = */ axisRef.getSize(mesh),
      /* .splitPreSize = */ splitInfo ? std::make_optional(splitInfo.getPreSize())
                                : std::nullopt};
}

// Adds all remaining axes in `allAxes`, starting from
// {`axisIndex`, `remainingAxisInfo`}, to `currentAxes`.
void addRemainingAxes(SmallVector<AxisRefAttr>& currentAxes,
                      std::optional<AxisRefInfo> remainingAxisInfo,
                      ArrayRef<AxisRefAttr> allAxes, int64_t axisIndex,
                      MeshAttr mesh) {
  MLIRContext* ctx = mesh.getContext();
  while (remainingAxisInfo) {
    currentAxes.push_back(
        AxisRefAttr::get(ctx, allAxes[axisIndex].getName(),
                         remainingAxisInfo->getSubAxisInfo(ctx)));
    remainingAxisInfo = getAxisRefInfo(allAxes, ++axisIndex, mesh);
  }
}

// Builds a `TensorFactorShardings` for a tensor with the specified
// `optionalSharding` and `tensorMapping`.
//
// The high level algorithm for projecting a dimension sharding into factor
// shardings is to add axes (or sub-axes) from the dimension sharding to the
// current factor sharding (starting from the major-most factor and axis) until
// the factor is fully sharded, which might require further splitting an axis,
// or this is the minor-most factor, then moving to the next factor.
TensorFactorShardings buildTensorFactorShardings(
    TensorMappingAttr tensorMapping, TensorShardingAttr optionalSharding,
    ArrayRef<int64_t> factorSizes, MeshAttr mesh) {
  TensorFactorShardings result;
  auto& [factorIndexToSharding, replicatedAxes] = result;
  factorIndexToSharding.reserve(factorSizes.size());

  // 1. Populate factor shardings
  for (const auto [dim, dimMapping] :
       llvm::enumerate(tensorMapping.getDimMappings())) {
    ArrayRef<AxisRefAttr> axes =
        optionalSharding ? optionalSharding.getDimSharding(dim).getAxes()
                         : ArrayRef<AxisRefAttr>();
    bool isDimClosed = optionalSharding && optionalSharding.isClosed(dim);

    int64_t axisIndex = 0;
    std::optional<AxisRefInfo> remainingAxisInfo =
        getAxisRefInfo(axes, axisIndex, mesh);

    MLIRContext* ctx = mesh.getContext();

    bool hasOverflowAxes = false;
    for (int64_t factorIndex : dimMapping.getFactorIndices()) {
      FactorSharding& factorSharding = factorIndexToSharding[factorIndex];
      factorSharding.isMinorMost = dimMapping.isMinorMost(factorIndex);

      if (hasOverflowAxes) {
        // If a previous factor had overflow axes, all subsequent factors should
        // be empty and closed, so that they won't be sharded along any axis.
        factorSharding.isClosed = true;
        break;
      }

      factorSharding.isClosed = isDimClosed;
      int64_t remainingFactorSize = factorSizes[factorIndex];

      if (factorSharding.isMinorMost) {
        // This is the minor-most factor
        //
        // NOTE: we allow the minor-most factor to be sharded by axes whose
        // product of sizes doesn't divide the factor's size (which requires
        // padding).
        addRemainingAxes(factorSharding.axisRefs, remainingAxisInfo, axes,
                         axisIndex, mesh);
        break;
      }

      // This is a non-minor-most factor
      int64_t remainingSizeGcd = 1;
      while (remainingAxisInfo &&
             (remainingAxisInfo->size == 1 ||
              (remainingSizeGcd = std::gcd(remainingFactorSize,
                                           remainingAxisInfo->size)) > 1)) {
        // Extract the current axis name here, as `axisIndex` might change
        // before the name is used.
        StringRef axisName = axes[axisIndex].getName();
        SubAxisInfoAttr subAxisInfo = nullptr;
        if (remainingAxisInfo->size > remainingSizeGcd) {
          // We need to further split the axis
          subAxisInfo = SubAxisInfoAttr::get(
              ctx, remainingAxisInfo->splitPreSize.value_or(1),
              remainingSizeGcd);
          remainingAxisInfo->splitPreSize = subAxisInfo.getNextPreSize();
          remainingAxisInfo->size /= remainingSizeGcd;
          remainingFactorSize /= remainingSizeGcd;
        } else {
          // We can advance to the next axis or sub-axis
          subAxisInfo = remainingAxisInfo->getSubAxisInfo(ctx);
          remainingFactorSize /= remainingAxisInfo->size;
          remainingAxisInfo = getAxisRefInfo(axes, ++axisIndex, mesh);
        }
        factorSharding.axisRefs.push_back(
            AxisRefAttr::get(ctx, axisName, subAxisInfo));
      }

      if (remainingFactorSize > 1 && remainingAxisInfo) {
        // This means the remaining size of the factor and current axis are
        // relatively prime, therefore we add all remaining axes to the list of
        // overflow axes.
        hasOverflowAxes = true;
        addRemainingAxes(factorSharding.overflowAxes, remainingAxisInfo, axes,
                         axisIndex, mesh);
      }
    }
  }

  // 2. Populate replicated axes
  if (optionalSharding) {
    replicatedAxes.assign(optionalSharding.getReplicatedAxes().begin(),
                          optionalSharding.getReplicatedAxes().end());
  }

  return result;
}

}  // namespace

ShardingProjection::ShardingProjection(
    SmallVector<TensorFactorShardings> operands,
    SmallVector<TensorFactorShardings> results)
    : operands(std::move(operands)), results(std::move(results)) {}

ShardingProjection ShardingProjection::build(
    ArrayRef<TensorShardingAttr> operandShardings,
    ArrayRef<TensorShardingAttr> resultShardings,
    OpShardingRuleAttr shardingRule, MeshAttr mesh) {
  ShardingProjection projection;

  for (const auto& [operandSharding, operandMapping] :
       llvm::zip_equal(operandShardings, shardingRule.getOperandMappings())) {
    projection.operands.push_back(buildTensorFactorShardings(
        operandMapping, operandSharding, shardingRule.getFactorSizes(), mesh));
  }

  for (const auto& [resultSharding, resultMapping] :
       llvm::zip_equal(resultShardings, shardingRule.getResultMappings())) {
    projection.results.push_back(buildTensorFactorShardings(
        resultMapping, resultSharding, shardingRule.getFactorSizes(), mesh));
  }

  return projection;
}

ShardingProjection ShardingProjection::build(Operation* op,
                                             OpShardingRuleAttr shardingRule,
                                             MeshAttr mesh) {
  return build(getShardings(op->getOperands()), getShardings(op->getResults()),
               shardingRule, mesh);
}

}  // namespace sdy
}  // namespace mlir
