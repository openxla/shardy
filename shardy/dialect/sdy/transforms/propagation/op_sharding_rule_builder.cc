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

#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

namespace {

// Creates a vector of `TensorMappingAttr` corresponding to the given vector of
// `TensorMapping`.
//
// In addition, adds a factor of size 1 to all dimensions that don't have a
// factor.
SmallVector<TensorMappingAttr> buildTensorMappingAttrList(
    ArrayRef<TensorMapping> tensorMappings, SmallVector<int64_t>& factorSizes,
    MLIRContext* context) {
  SmallVector<TensorMappingAttr> tensorMappingAttrs;
  tensorMappingAttrs.reserve(tensorMappings.size());
  for (const TensorMapping& tensorMapping : tensorMappings) {
    SmallVector<DimMappingAttr> dimMappings;
    dimMappings.reserve(tensorMapping.size());
    for (const DimMapping& dimMapping : tensorMapping) {
      if (dimMapping.factorIndices.empty()) {
        dimMappings.push_back(DimMappingAttr::get(context, factorSizes.size()));
        factorSizes.push_back(1);
      } else {
        dimMappings.push_back(
            DimMappingAttr::get(context, dimMapping.factorIndices));
      }
    }
    tensorMappingAttrs.push_back(TensorMappingAttr::get(context, dimMappings));
  }
  return tensorMappingAttrs;
}

// Maps the given `tensorDims` that are not equal to `kNullDim` to
// `factorIndex`.
void mapDimsToFactor(SmallVector<TensorMapping>& tensorMappings,
                        ArrayRef<int64_t> tensorDims, int64_t factorIndex) {
  for (auto [tensorMapping, tensorDim] :
       llvm::zip_equal(tensorMappings, tensorDims)) {
    if (tensorDim == kNullDim) {
      continue;
    }
    assert(tensorDim >= 0);
    tensorMapping[tensorDim].factorIndices.push_back(factorIndex);
  }
}

}  // namespace

OpShardingRuleBuilder::OpShardingRuleBuilder(
    TypeRange operandTypes, TypeRange resultTypes, MLIRContext* context,
    std::optional<int64_t> reserveNumFactors)
    : context(context) {
  operandMappings.reserve(operandTypes.size());
  resultMappings.reserve(resultTypes.size());
  int64_t maxRank = 0;
  for (Type operandType : operandTypes) {
    int64_t rank = cast<RankedTensorType>(operandType).getRank();
    maxRank = std::max(maxRank, rank);
    operandMappings.push_back(TensorMapping(rank));
  }
  for (Type resultType : resultTypes) {
    int64_t rank = cast<RankedTensorType>(resultType).getRank();
    maxRank = std::max(maxRank, rank);
    resultMappings.push_back(TensorMapping(rank));
  }
  factorSizes.reserve(reserveNumFactors.value_or(maxRank));
}

OpShardingRuleBuilder::OpShardingRuleBuilder(
    Operation* op, std::optional<int64_t> reserveNumFactors)
    : OpShardingRuleBuilder(op->getOperandTypes(), op->getResultTypes(),
                            op->getContext(), reserveNumFactors) {}

OpShardingRuleAttr OpShardingRuleBuilder::build() {
  // NOTE: `factorSizes` might be modified by `buildTensorMappingAttrList`,
  // therefore we can't inline these variables.
  int64_t originalNumFactors = factorSizes.size();
  SmallVector<TensorMappingAttr> operandMappingAttrs =
      buildTensorMappingAttrList(operandMappings, factorSizes, context);
  SmallVector<TensorMappingAttr> resultMappingAttrs =
      buildTensorMappingAttrList(resultMappings, factorSizes, context);

  auto result = OpShardingRuleAttr::get(
      context, factorSizes, operandMappingAttrs, resultMappingAttrs);

  // Erase all added factors, to return the builder to its original state before
  // calling this method.
  factorSizes.resize(originalNumFactors);
  return result;
}

OpShardingRuleAttr OpShardingRuleBuilder::buildPointwise(Operation* op) {
  // All results should have the same shape, so we look at the first.
  ArrayRef<int64_t> shape =
      cast<RankedTensorType>(op->getResultTypes().front()).getShape();

  OpShardingRuleBuilder builder(op);

  builder.factorSizes.assign(shape.begin(), shape.end());

  for (TensorMapping& tensorMapping : llvm::concat<TensorMapping>(
           builder.operandMappings, builder.resultMappings)) {
    for (auto [i, dimMapping] : llvm::enumerate(tensorMapping)) {
      dimMapping.factorIndices.push_back(i);
    }
  }

  return builder.build();
}

OpShardingRuleBuilder& OpShardingRuleBuilder::addFactor(
    ArrayRef<int64_t> operandDims, ArrayRef<int64_t> resultDims,
    int64_t factorSize) {
  int64_t factorIndex = factorSizes.size();
  mapDimsToFactor(operandMappings, operandDims, factorIndex);
  mapDimsToFactor(resultMappings, resultDims, factorIndex);
  factorSizes.push_back(factorSize);
  return *this;
}

OpShardingRuleBuilder& OpShardingRuleBuilder::addFactor(int64_t dim,
                                                        int64_t factorSize) {
  int64_t factorIndex = factorSizes.size();
  for (TensorMapping& tensorMapping :
       llvm::concat<TensorMapping>(operandMappings, resultMappings)) {
    if (tensorMapping.empty()) {
      // Rank is 0
      continue;
    }
    tensorMapping[dim].factorIndices.push_back(factorIndex);
  }
  factorSizes.push_back(factorSize);
  return *this;
}

OpShardingRuleBuilder& OpShardingRuleBuilder::addPointwise(
    ArrayRef<int64_t> shape) {
  for (auto [dim, dimSize] : llvm::enumerate(shape)) {
    addFactor(dim, dimSize);
  }
  return *this;
}

OpShardingRuleBuilder& OpShardingRuleBuilder::addPointwiseIf(
    ArrayRef<int64_t> shape, std::function<bool(int64_t)> pred) {
  for (auto [dim, dimSize] : llvm::enumerate(shape)) {
    if (pred(dim)) {
      addFactor(dim, dimSize);
    }
  }
  return *this;
}

OpShardingRuleAttr createIdentityShardingRule(RankedTensorType type,
                                              size_t numOperands,
                                              size_t numResults) {
  return OpShardingRuleBuilder(SmallVector<Type>(numOperands, type),
                               SmallVector<Type>(numResults, type),
                               type.getContext())
      .addPointwise(type.getShape())
      .build();
}

}  // namespace sdy
}  // namespace mlir
