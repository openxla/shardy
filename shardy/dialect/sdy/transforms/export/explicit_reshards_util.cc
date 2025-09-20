/* Copyright 2025 The Shardy Authors.

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

#include "shardy/dialect/sdy/transforms/export/explicit_reshards_util.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/axis_list_ref.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"

namespace mlir {
namespace sdy {

namespace {

// Returns true iff any tensor factor sharding has non-empty overflow axes.
bool hasOverflowAxes(const ShardingProjection& shardingProjection) {
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(
           shardingProjection.getOperands(), shardingProjection.getResults())) {
    for (const auto& [_, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      if (!factorSharding.overflowAxes.empty()) {
        return true;
      }
    }
  }
  return false;
}

bool hasShardedPermutationFactors(
    const TensorFactorShardings& tensorFactorSharding,
    OpShardingRuleAttr shardingRule) {
  return llvm::any_of(tensorFactorSharding.factorIndexToSharding,
                      [&](const auto& factorIndexAndSharding) {
                        const auto& [factorIndex, factorSharding] =
                            factorIndexAndSharding;
                        return shardingRule.isPermutationFactor(factorIndex) &&
                               !factorSharding.axisRefs.empty();
                      });
}

// Checks if factor sharding is compatible, that is, it satisfies:
// 1. Factors are sharded the same way across operands and results.
// 2. Factors that need replication are unsharded.
//
// Returns the common axes per factor if the factor sharding is compatible.
// Otherwise, returns std::nullopt.
//
// Assumes factor shardings do not have overflow axes.
// TODO(enver): Handle the case when some factor shardings have overflow axes.
std::optional<AxesPerFactor> getCompatibleFactorShardings(
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule) {
  AxesPerFactor commonAxesPerFactor(shardingRule.getNumFactors());
  BitVector seenFactors(shardingRule.getNumFactors());
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(
           shardingProjection.getOperands(), shardingProjection.getResults())) {
    // Detects conflicts within the same factor.
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      // Factors that need replication should be unsharded across all operands
      // and results in order for it to have a compatible sharding.
      if (shardingRule.isNeedReplicationFactor(factorIndex)) {
        if (!factorSharding.axisRefs.empty()) {
          return std::nullopt;
        }
        continue;
      }
      if (!seenFactors.test(factorIndex)) {
        commonAxesPerFactor[factorIndex] = factorSharding.axisRefs;
        seenFactors.set(factorIndex);
      } else if (factorSharding.axisRefs != commonAxesPerFactor[factorIndex]) {
        return std::nullopt;
      }
    }
  }

  return commonAxesPerFactor;
}

void insertExplicitReshardsOnOperand(
    Operation* op, const int64_t operandIndex,
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule, const Mesh& mesh, IRRewriter& rewriter) {
  auto operand = op->getOperand(operandIndex);
  auto newTensorSharding =
      shardingProjection.getOperand(operandIndex)
          .createTensorShardingAttr(
              mesh.getContext(), shardingRule.getOperandMapping(operandIndex),
              shardingRule.getFactorSizes(), mesh.name(), mesh.attr());
  auto reshardOp =
      ReshardOp::create(rewriter, operand.getLoc(), operand, newTensorSharding);
  op->setOperand(operandIndex, reshardOp);
}

void insertExplicitReshardsOnResult(
    Operation* op, const int64_t resultIndex,
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule, const Mesh& mesh, IRRewriter& rewriter) {
  auto result = op->getResult(resultIndex);
  auto newTensorSharding =
      shardingProjection.getResult(resultIndex)
          .createTensorShardingAttr(
              mesh.getContext(), shardingRule.getResultMapping(resultIndex),
              shardingRule.getFactorSizes(), mesh.name(), mesh.attr());
  auto reshardOp = ReshardOp::create(
      rewriter, result.getLoc(), result,
      getOrCreateSharding(result, mesh.name(), /*closedIfMissing=*/true));
  rewriter.replaceAllUsesExcept(result, reshardOp, reshardOp);
  setSharding(result, newTensorSharding);
}

// TODO(enver): Detect cases where two meshes are indeed equivalent when device
// orders are different only on replicated axes, for example:
// mesh1 = [x:2, y:2]
// mesh2 = [x:2, y:2], device_ids=[1,0,3,2].
// If we replicate along y and only use the x in sharding.
bool shouldReshardToCommonMesh(TensorShardingAttr sharding, const Mesh& mesh,
                               const SymbolTable& symbolTable) {
  return !isFullyReplicated(sharding) &&
         sharding.getMesh(symbolTable).getDeviceIds() !=
             mesh.attr().getDeviceIds();
}

// Insert explicit reshards for operands and results that change by
// the given `shardingProjection` for a given `op`. The reshards are inserted
// only to make the given operation compatible.
//
// For example,
//
// ```mlir
//   %arg0: tensor<8x32xf32> { sdy.sharding = @mesh, [{}, {"y"}]>}
//   %arg1: tensor<32x16xf32> { sdy.sharding = <@mesh, [{"y"}, {"x"}]>}
//   %0 = stablehlo.dot %arg0, %arg1 { sdy.sharding = <@mesh, [{"x"}, {}]>,
//     sdy.sharding_rule = <([i, k], [k, j])->([i, j])> }
//   %1 = stablehlo.negate %0 {sdy.sharding = <@mesh, [{"x"}, {}]>
//   return %1
// ```
//
// after a call on the stablehlo.dot operation, by the sharding projection,
// i: {}, j: {}, k: {"y"}, the module becomes:
//
// ```mlir
//   %arg0: tensor<8x32xf32> { sdy.sharding = @mesh, [{}, {"y"}]>}
//   %arg1: tensor<32x16xf32> { sdy.sharding = <@mesh, [{"y"}, {"x"}]>}
//   %0 = stablehlo.reshard %arg1 {sdy.sharding = <@mesh, [{"y"}, {}]>}
//   %1 = stablehlo.dot %arg0, %0 { sdy.sharding = <@mesh, [{}, {}]>,
//     sdy.sharding_rule = <([i, k], [k, j])->([i, j])> }
//   %2 = stablehlo.reshard %1 {sdy.sharding = <@mesh, [{"x"}, {}]>}
//   %3 = stablehlo.negate %2 {sdy.sharding = <@mesh, [{"x"}, {}]>
//   return %3
// ```
//
// In the above example, note that the operand and result shardings for
// stablehlo.negate op remained unchanged.
//
// Assumes factor shardings do not have overflow axes.
// TODO(enver): Handle the case when some factor shardings have overflow axes.
//
// Assumes all tensor shardings have the same mesh as `mesh` on axes but may be
// different on device order.
void insertExplicitReshards(Operation* op,
                            ArrayRef<TensorShardingAttr> inShardings,
                            ArrayRef<TensorShardingAttr> outShardings,
                            const ShardingProjection& shardingProjection,
                            UpdateTensorShardings updateTensorShardings,
                            IRRewriter& rewriter,
                            OpShardingRuleAttr shardingRule,
                            const SymbolTable& symbolTable, const Mesh& mesh) {
  rewriter.setInsertionPoint(op);
  for (const auto& [operandIndex, operandSharding] :
       llvm::enumerate(inShardings)) {
    if (updateTensorShardings.updateOperands.test(operandIndex) ||
        shouldReshardToCommonMesh(operandSharding, mesh, symbolTable)) {
      insertExplicitReshardsOnOperand(op, operandIndex, shardingProjection,
                                      shardingRule, mesh, rewriter);
    }
  }
  rewriter.setInsertionPointAfter(op);
  for (const auto& [resultIndex, resultSharding] :
       llvm::enumerate(outShardings)) {
    if (updateTensorShardings.updateResults.test(resultIndex) ||
        shouldReshardToCommonMesh(resultSharding, mesh, symbolTable)) {
      insertExplicitReshardsOnResult(op, resultIndex, shardingProjection,
                                     shardingRule, mesh, rewriter);
    }
  }
}

struct FactorAxesPair {
  constexpr static int64_t kEmptyFactorIndex = -1;
  constexpr static int64_t kTombstoneFactorIndex = -2;

  int64_t factorIndex = kEmptyFactorIndex;
  AxisListRef axes;

  FactorAxesPair(int64_t factorIndex, AxisListRef axes)
      : factorIndex(factorIndex), axes(axes) {}

  // TODO(enver): Define EmptyFactorAxesPair class with overloaded methods and
  // use it when the axes is empty.
  FactorAxesPair(int64_t factorIndex) : factorIndex(factorIndex) {}
  FactorAxesPair() = default;

  bool operator<(const FactorAxesPair& rhs) const {
    if (factorIndex != rhs.factorIndex) {
      return factorIndex < rhs.factorIndex;
    }
    return axes < rhs.axes;
  }

  bool operator==(const FactorAxesPair& rhs) const {
    return factorIndex == rhs.factorIndex && axes == rhs.axes;
  }

  bool empty() const { return factorIndex == kEmptyFactorIndex; }

  bool isFullySharded(OpShardingRuleAttr shardingRule, MeshAttr mesh) {
    return axes.getShardingSize(mesh) ==
           shardingRule.getFactorSizes()[factorIndex];
  }
};

struct FactorAxesPairInfo : public llvm::DenseMapInfo<FactorAxesPair> {
  static unsigned getHashValue(const FactorAxesPair& m) {
    return llvm::hash_combine(m.factorIndex, m.axes.toPair());
  }
  static bool isEqual(const FactorAxesPair& lhs, const FactorAxesPair& rhs) {
    return lhs == rhs;
  }

  static inline FactorAxesPair getEmptyKey() { return FactorAxesPair(); }

  static inline FactorAxesPair getTombstoneKey() {
    return FactorAxesPair(FactorAxesPair::kTombstoneFactorIndex);
  }
};

struct FactorAxesCandidate {
  FactorAxesPair factorAxes;
  // The total size of the source tensors.
  int64_t totalSourceTensorSize = 0;
  // The size of the source tensor. In case the factor-axes pair has multiple
  // source tensors, the size of the largest one. A tensor is a source for a
  // factor-axes pair if the axes is a prefix of the factor sharding on the
  // tensor.
  int64_t largestSourceTensorSize = 0;
  // The size of axes to shard further. Hence, if the factor is already assigned
  // to axes A, and this factor-axes pair has axes B, the size of further
  // sharding is size(B)/size(A), and where A is a strict prefix of B.
  int64_t shardingSize = 0;
  int64_t factorTypePrecedence = 0;

  FactorAxesCandidate(FactorAxesPair factorAxes, int64_t sourceTensorSize,
                      int64_t shardingSize, FactorType factorType)
      : factorAxes(factorAxes),
        totalSourceTensorSize(sourceTensorSize),
        largestSourceTensorSize(sourceTensorSize),
        shardingSize(shardingSize),
        factorTypePrecedence(precedence(factorType)) {}

  FactorAxesCandidate() = default;

  bool operator<(const FactorAxesCandidate& rhs) const {
    if (totalSourceTensorSize != rhs.totalSourceTensorSize) {
      return totalSourceTensorSize < rhs.totalSourceTensorSize;
    }
    if (factorTypePrecedence != rhs.factorTypePrecedence) {
      return factorTypePrecedence < rhs.factorTypePrecedence;
    }
    if (largestSourceTensorSize != rhs.largestSourceTensorSize) {
      return largestSourceTensorSize < rhs.largestSourceTensorSize;
    }
    if (shardingSize != rhs.shardingSize) {
      return shardingSize < rhs.shardingSize;
    }
    // The following also ensures that, for two axes, if one is a strict prefix
    // of another, the strict prefix one is smaller.
    return factorAxes < rhs.factorAxes;
  }

  // A candidate with a higher precedence will always be preferable to a
  // candidate with a lower precedence when finding the best candidate to extend
  // the factor sharding assignment during the majority vote heuristic.
  int64_t precedence(FactorType factorType) {
    switch (factorType) {
      case FactorType::kPassThrough:
      case FactorType::kReduction:
        return 2;
      case FactorType::kPermutation:
        return 1;
      case FactorType::kNeedReplication:
        return 0;
    }
  }

  bool empty() const { return factorAxes.empty(); }
};

using FactorAxesCandidatesMap =
    DenseMap<FactorAxesPair, FactorAxesCandidate, FactorAxesPairInfo>;

// Increment the count for the factor-axes pair, also modify source tensor size
// to keep the largest.
void updateFactorAxesCandidate(FactorAxesCandidatesMap& factorAxesCounts,
                               const FactorAxesPair& factorAxes,
                               int64_t sourceTensorSize, const Mesh& mesh,
                               const FactorType factorType) {
  if (auto factorAxesCountIt = factorAxesCounts.find(factorAxes);
      factorAxesCountIt != factorAxesCounts.end()) {
    FactorAxesCandidate& candidate = factorAxesCountIt->second;
    candidate.totalSourceTensorSize += sourceTensorSize;
    candidate.largestSourceTensorSize =
        std::max(candidate.largestSourceTensorSize, sourceTensorSize);
    return;
  }
  factorAxesCounts.try_emplace(factorAxes, factorAxes, sourceTensorSize,
                               factorAxes.axes.getShardingSize(mesh.attr()),
                               factorType);
}

// A container for FactorAxesCandidates where the order of iteration does not
// matter, and provides methods to insert and remove candidates in constant-time
// while maintaining the best through explicit calls on its touchAt method.
class FactorAxesCandidateBag {
 public:
  FactorAxesCandidateBag(MeshAttr mesh, OpShardingRuleAttr shardingRule)
      : mesh(mesh) {
    initFactorDependencies(shardingRule);
  }

  // Returns whether the bag is empty.
  bool empty() const { return candidates.empty(); }

  // Inserts a new candidate to the bag. Performs in constant-time.
  void insert(const FactorAxesCandidate& candidate) {
    candidates.push_back(candidate);
    updateBestCandidateIfValid(candidate);
  }

  // Updates the sharding size of the one at index as the  product of the
  // sharding sizes of all individual axes excluding the `prefix`, also update
  // the best.
  //
  // Assumes `prefix` is a prefix of the axes of the candidate at index.
  void updateShardingSizeAt(const int64_t index,
                            const AxisListRef& prefix = AxisListRef()) {
    FactorAxesCandidate& candidate = candidates[index];
    candidate.shardingSize =
        candidate.factorAxes.axes.getExpandedShardingSize(mesh, prefix);
    updateBestCandidateIfValid(candidate);
  }

  // Updates the source tensor sizes of all candidates.
  // TODO(enver): Optimize updating source tensor sizes.
  void updateSourceTensorSizes(const ShardingProjection& shardingProjection,
                               ArrayRef<int64_t> tensorSizes,
                               const SmallVector<AxisListRef>& factorAxisRefs) {
    // Since the (local) source tensor sizes get smaller at each iteration on
    // which we extend sharding of a factor, in order to recompute largest
    // source tensor sizes, we first need to reset them to zero.
    resetLargestSourceTensorSizes();
    for (const auto& [tensorIndex, tensorFactorSharding] :
         llvm::enumerate(llvm::concat<const TensorFactorShardings>(
             shardingProjection.getOperands(),
             shardingProjection.getResults()))) {
      int64_t localTensorSize = tensorSizes[tensorIndex];
      for (const auto& [factorIndex, _] :
           tensorFactorSharding.factorIndexToSharding) {
        // TODO(enver): Consider cases tensor size may not be divisible.
        localTensorSize /= factorAxisRefs[factorIndex].getShardingSize(mesh);
      }
      for (const auto& [factorIndex, _] :
           tensorFactorSharding.factorIndexToSharding) {
        for (int64_t candidateIndex = 0; candidateIndex < size();
             ++candidateIndex) {
          updateSourceTensorSizeAt(factorIndex, candidateIndex,
                                   localTensorSize);
        }
      }
    }
  }

  void dropFactorDependencies(const int64_t factorIndex) {
    for (auto& [_, factorDependencies] : factorDependenciesMap) {
      factorDependencies.reset(factorIndex);
    }
  }

  // Resets best. Performs in constant-time.
  void resetBest() { bestCandidate = FactorAxesCandidate(); }

  // Removes candidate at index. Performs in constant-time. After the
  // operation, the candidates before the index keep being before the index, and
  // the candidates after the index (except the removed one) keep being after
  // the index. Assumes that the index is within the bounds and the removed one
  // is not the best one.
  //
  // Since the order of iteration does not matter, it simply swaps the candidate
  // at index with the last one, hence in the constant time.
  void removeAt(const int64_t index) {
    candidates[index] = candidates.back();
    candidates.pop_back();
  }

  // Returns the best. Performs in constant-time.
  FactorAxesCandidate best() const { return bestCandidate; }
  // Returns the candidate at index. Performs in constant-time.
  FactorAxesCandidate& at(const int64_t index) { return candidates[index]; }
  // Returns the number of candidates in the bag.
  int64_t size() const { return candidates.size(); }

 private:
  void resetLargestSourceTensorSizes() {
    for (FactorAxesCandidate& candidate : candidates) {
      candidate.largestSourceTensorSize = 0;
    }
  }

  void initFactorDependencies(OpShardingRuleAttr shardingRule) {
    for (const TensorMappingAttr& tensorMapping :
         llvm::concat<const TensorMappingAttr>(
             shardingRule.getOperandMappings(),
             shardingRule.getResultMappings())) {
      for (DimMappingAttr dimMapping : tensorMapping.getDimMappings()) {
        ArrayRef<int64_t> factorIndices = dimMapping.getFactorIndices();
        for (int64_t index = 1; index < factorIndices.size(); index++) {
          int64_t factorIndex = factorIndices[index];
          int64_t dependsOn = factorIndices[index - 1];
          if (!factorDependenciesMap.contains(factorIndex)) {
            factorDependenciesMap.try_emplace(factorIndex,
                                              shardingRule.getNumFactors());
          }
          factorDependenciesMap[factorIndex].set(dependsOn);
        }
      }
    }
  }

  void updateBestCandidateIfValid(const FactorAxesCandidate& candidate) {
    if (isValid(candidate)) {
      bestCandidate = std::max(bestCandidate, candidate);
    }
  }

  bool isValid(const FactorAxesCandidate& candidate) {
    auto it = factorDependenciesMap.find(candidate.factorAxes.factorIndex);
    return it == factorDependenciesMap.end() || it->second.none();
  }

  void updateSourceTensorSizeAt(const int64_t factorIndex, const int64_t index,
                                const int64_t sourceTensorSize) {
    FactorAxesCandidate& candidate = candidates[index];
    if (candidate.factorAxes.factorIndex == factorIndex) {
      candidate.largestSourceTensorSize =
          std::max(candidate.largestSourceTensorSize, sourceTensorSize);
      updateBestCandidateIfValid(candidate);
    }
  }

  // A factor is non-full if its sharding size is smaller than the size of the
  // factor. `factorDependenciesMap` is a map from factor indices to bitvectors,
  // each bitvector is associated with a factor f, and represents the set of
  // non-full factor indices that factor f depends on. A factor f depends on
  // factor g if two factors appear together in any tensor dimension, and g
  // appears immediately before f. This list is used to determine if a
  // factor-axes candidate is valid yet since a factor should not be sharded
  // until it does have zero dependencies, that is, all factors that appear, in
  // any tensor dimension, before the factor needs to be fully-sharded,
  // otherwise it would introduce a strided view which is not supported yet.
  // Note that a factor may depend on separate factors on separate dimensions,
  // hence it may depend on multiple factors.
  llvm::SmallDenseMap<int64_t, BitVector> factorDependenciesMap;
  SmallVector<FactorAxesCandidate> candidates;
  FactorAxesCandidate bestCandidate;
  // Used for recalculating sharding size of a candidate.
  MeshAttr mesh;
};

FactorAxesCandidateBag findFactorAxesCandidates(
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule, ArrayRef<int64_t> tensorSizes,
    const Mesh& mesh) {
  // Find sets of candidate axes per factor.
  SmallVector<DenseSet<AxisListRef, AxisListRefInfo>> axesSets(
      shardingRule.getNumFactors());
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(
           shardingProjection.getOperands(), shardingProjection.getResults())) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      if (shardingRule.isNeedReplicationFactor(factorIndex)) {
        continue;
      }
      ArrayRef<AxisRefAttr> axisRefs = factorSharding.axisRefs;
      while (!axisRefs.empty()) {
        axesSets[factorIndex].insert(AxisListRef(axisRefs));
        axisRefs = axisRefs.drop_back();
      }
    }
  }

  // TODO(enver): For two factor-axes pairs, if both have the same factor and
  // the same count, and one is the prefix of the other, drop the prefix one.

  // Count factor-axes pairs.
  FactorAxesCandidatesMap factorAxesCandidatesMap;
  for (const auto& [tensorIndex, tensorFactorSharding] :
       llvm::enumerate(llvm::concat<const TensorFactorShardings>(
           shardingProjection.getOperands(),
           shardingProjection.getResults()))) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      if (factorSharding.axisRefs.empty() ||
          shardingRule.isNeedReplicationFactor(factorIndex)) {
        continue;
      }
      FactorAxesPair factorAxes(factorIndex,
                                AxisListRef(factorSharding.axisRefs));
      updateFactorAxesCandidate(factorAxesCandidatesMap, factorAxes,
                                tensorSizes[tensorIndex], mesh,
                                shardingRule.getFactorType(factorIndex));
      // Increment counts for all its strict prefixes.
      for (const AxisListRef& axes : axesSets[factorIndex]) {
        if (axes.strictPrefixOf(factorAxes.axes)) {
          updateFactorAxesCandidate(factorAxesCandidatesMap,
                                    FactorAxesPair(factorIndex, axes),
                                    tensorSizes[tensorIndex], mesh,
                                    shardingRule.getFactorType(factorIndex));
        }
      }
    }
  }

  FactorAxesCandidateBag factorAxesCandidates(mesh.attr(), shardingRule);
  for (const auto& [_, candidate] : factorAxesCandidatesMap) {
    factorAxesCandidates.insert(candidate);
  }
  return factorAxesCandidates;
}

AxesPerFactor toAxesPerFactor(const SmallVector<AxisListRef>& factorAxisRefs) {
  AxesPerFactor factorCommonAxes(factorAxisRefs.size());
  for (const auto& [factorCommonAxesOfFactor, factorAxisRef] :
       llvm::zip_equal(factorCommonAxes, factorAxisRefs)) {
    factorCommonAxesOfFactor = factorAxisRef.toVector();
  }
  return factorCommonAxes;
}

// Broadly the algorithm is, at each iteration, to pick a {factor,axis} pair
// with the largest count from a list that is initialized with all the
// pairs with non-zero count, assign the picked axis to the picked factor, and
// delete all the pairs from the list that is either with the picked factor,
// or with an axis that overlaps with the picked axis. Continue iterating
// until the list is empty.
AxesPerFactor findCommonAxesUsingMajorityVoteHeuristic(
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule, ArrayRef<int64_t> tensorSizes,
    const Mesh& mesh) {
  SmallVector<AxisListRef> factorAxisRefs(shardingRule.getNumFactors());
  FactorAxesCandidateBag factorAxesCandidates = findFactorAxesCandidates(
      shardingProjection, shardingRule, tensorSizes, mesh);
  // TODO(enver): Assign an axis to a factor immediately if the count is more
  // than floor(n/2) where n is the number of tensors.
  while (!factorAxesCandidates.best().empty()) {
    FactorAxesPair bestFactorAxes = factorAxesCandidates.best().factorAxes;
    factorAxesCandidates.resetBest();
    factorAxisRefs[bestFactorAxes.factorIndex] = bestFactorAxes.axes;
    if (bestFactorAxes.isFullySharded(shardingRule, mesh.attr())) {
      factorAxesCandidates.dropFactorDependencies(bestFactorAxes.factorIndex);
    }
    // Invalidate axes that overlaps with the picked one across all unseen
    // factors. During the iteration, also find the new best.
    int64_t candidateIndex = 0;
    while (candidateIndex < factorAxesCandidates.size()) {
      FactorAxesCandidate& candidate = factorAxesCandidates.at(candidateIndex);
      // TODO(enver): Relax the overlap check. We need to erase in case of an
      // overlap only if the factor indices appear together in any of the
      // operands or results.
      if (candidate.factorAxes.factorIndex == bestFactorAxes.factorIndex) {
        // Drop any factor-axis pair that can not extend on the best one, for
        // the best factor, which is a (not necessarily strict) prefix of an
        // existing sharding of the factor.
        // Drops when the iterated axes is the same as the best one, as a
        // result the best factor-axis pair removed from the map.
        if (!bestFactorAxes.axes.strictPrefixOf(candidate.factorAxes.axes)) {
          factorAxesCandidates.removeAt(candidateIndex);
        } else {
          // At each iteration, we pick a factor-axes pair that expands
          // on the existing assignment on `factorAxisRefs`. In order to
          // use the part that we expand, we exclude the existing
          // assignment when taking the sharding size. Note, for a
          // factor-axes pair in the map, it is guaranteed that the
          // existing assignment is always a prefix of the axes of the
          // factor-axes pair, as we remove all factor-axes pair who can
          // not expand from the picked axes for the picked factor from
          // map at each iteration.
          factorAxesCandidates.updateShardingSizeAt(
              candidateIndex++, /*prefix=*/bestFactorAxes.axes);
        }
        continue;
      }
      if (candidate.factorAxes.axes.truncateWithoutOverlap(
              bestFactorAxes.axes)) {
        // In case the axes is trimmed up to the current assignment then we
        // can drop it from the list as it would not be expanding on the current
        // assignment.
        // Note that it is guaranteed that the current assignment of candidate's
        // factor is a prefix of `prefixAxes` as
        //   1. We only keep the candidates with axes that can expand on the
        //   current assignment of its factor, and
        //   2. We trim the axes of candidates that overlaps with any of the
        //   current assignment (and hence the picked axes do not overlap with
        //   the current assignment of candidate's factor).
        if (candidate.factorAxes.axes ==
            factorAxisRefs[candidate.factorAxes.factorIndex]) {
          factorAxesCandidates.removeAt(candidateIndex);
        } else {
          // Trim the axes to use the largest prefix that does not overlap
          // with the picked one.
          factorAxesCandidates.updateShardingSizeAt(
              candidateIndex++,
              /*prefix=*/factorAxisRefs[candidate.factorAxes.factorIndex]);
        }
        continue;
      }
      factorAxesCandidates.updateShardingSizeAt(candidateIndex++);
    }

    // TODO(enver): Optimize updating source tensor sizes.
    factorAxesCandidates.resetBest();
    factorAxesCandidates.updateSourceTensorSizes(shardingProjection,
                                                 tensorSizes, factorAxisRefs);
  }

  // TODO(enver): Consider to keep factorAxisRefs for longer until actual
  // needed to call toVector.
  return toAxesPerFactor(factorAxisRefs);
}

std::optional<int64_t> findTensorIndexToPreferOnUnaryOperation(
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule, ArrayRef<int64_t> tensorSizes,
    const Mesh& mesh) {
  // Find common axes on the larger tensor, hence reshard the smaller tensor.
  SmallVector<int64_t> tensorIndices = shardingRule.getNonScalarTensorIndices();
  const int64_t lhs = tensorIndices[0];
  const int64_t rhs = tensorIndices[1];

  const bool lhsHasShardedPermutationFactor = hasShardedPermutationFactors(
      shardingProjection.getTensor(lhs), shardingRule);
  const bool rhsHasShardedPermutationFactor = hasShardedPermutationFactors(
      shardingProjection.getTensor(rhs), shardingRule);
  if (lhsHasShardedPermutationFactor != rhsHasShardedPermutationFactor) {
    return lhsHasShardedPermutationFactor ? rhs : lhs;
  }

  if (tensorSizes[lhs] != tensorSizes[rhs]) {
    return tensorSizes[lhs] < tensorSizes[rhs] ? rhs : lhs;
  }

  // Find common axes on the tensor that is more sharded, hence perform the
  // operation on smaller tensor per device.
  return shardingProjection.getTensor(lhs).getShardingSize(mesh.attr()) <
                 shardingProjection.getTensor(rhs).getShardingSize(mesh.attr())
             ? rhs
             : lhs;
}

// Assumes that:
// 1. Either tensor does not have factors that need replication.
// 2. Both tensors have the same mesh but may have different device orders.
// 3. The factor shardings are not compatible.
AxesPerFactor findCommonAxesOnUnaryOperation(
    ArrayRef<TensorShardingAttr> inShardings,
    ArrayRef<TensorShardingAttr> outShardings,
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule, ArrayRef<int64_t> tensorSizes,
    const SymbolTable& symbolTable, const Mesh& mesh) {
  std::optional<int64_t> tensorIndexToPrefer =
      findTensorIndexToPreferOnUnaryOperation(shardingProjection, shardingRule,
                                              tensorSizes, mesh);

  // If one tensor can not be chosen to be common axes, return empty so it skips
  // inserting explicit reshards for the operation.
  if (tensorIndexToPrefer == std::nullopt) {
    return AxesPerFactor();
  }

  // Set factor shardings to make sure factors that do not appear in the
  // preferred tensor are sharded on the other tensor.
  AxesPerFactor factorAxisRefs(shardingRule.getNumFactors());
  // TODO(enver): Add and use forEachFactorSharding helper method.
  for (const auto& [tensorIndex, tensorFactorSharding] :
       llvm::enumerate(llvm::concat<const TensorFactorShardings>(
           shardingProjection.getOperands(),
           shardingProjection.getResults()))) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      if (!factorSharding.axisRefs.empty()) {
        // TODO(enver): Drop the need for explicit AxisListRef casting.
        factorAxisRefs[factorIndex] = factorSharding.axisRefs;
      }
    }
  }

  // Override with the factor shardings on the preferred tensor.
  for (const auto& [factorIndex, factorSharding] :
       shardingProjection.getTensor(*tensorIndexToPrefer)
           .factorIndexToSharding) {
    factorAxisRefs[factorIndex] = factorSharding.axisRefs;
  }
  return factorAxisRefs;
}

void distributeAxisRefsToBatchingFactors(
    ArrayRef<AxisRefAttr> axisRefsToDistribute, OpShardingRuleAttr shardingRule,
    const Mesh& mesh, AxesPerFactor& factorCommonAxes) {
  // TODO(enver): Instead iterate batching factors in the order of their
  // available capacity, the one with largest available capacity being the
  // first one to distribute `axisRefsToDistribute`. Sort first, and then
  // iterate over sorted list.
  for (const int64_t factorIndex : shardingRule.getBatchingFactors()) {
    const int64_t factorSize = shardingRule.getFactorSizes()[factorIndex];
    // Skip if a factor has size zero which could happen if the corresponding
    // dimension has zero size.
    if (factorSize == 0) {
      continue;
    }
    SmallVector<AxisRefAttr>& factorSharding = factorCommonAxes[factorIndex];
    const int64_t factorShardingSize =
        AxisListRef(factorSharding).getShardingSize(mesh.attr());
    // NOTE: Here, `factorSize` can be smaller than `factorShardingSize` as in
    // some cases it is allowed to have shardings larger than its size.
    if ((factorSize % factorShardingSize) != 0) {
      // Skip the factor if factor size does not divide its sharding size,
      // hence a new axis can not be appended properly.
      continue;
    }
    int64_t factorCapacity = factorSize / factorShardingSize;
    while (!axisRefsToDistribute.empty()) {
      AxisRefAttr axisRef = axisRefsToDistribute.front();
      const int64_t axisSize = axisRef.getSize(mesh.attr());
      // TODO(enver): Split `axisRef` to fit it into the batching factor's
      // available space. The gcd of `axisSize` and `factorCapacity` should fit
      // to the batching factor.
      // The following check also guarantees that `factorCapacity` is not
      // smaller than `axisSize`.
      if (factorCapacity % axisSize != 0) {
        break;
      }
      addAxisOrMerge(factorSharding, axisRef, mesh.attr());

      factorCapacity /= axisSize;
      axisRefsToDistribute = axisRefsToDistribute.drop_front();
    }
    if (axisRefsToDistribute.empty()) {
      break;
    }
  }
}

AxesPerFactor findCommonAxes(ArrayRef<TensorShardingAttr> inShardings,
                             ArrayRef<TensorShardingAttr> outShardings,
                             const ShardingProjection& shardingProjection,
                             OpShardingRuleAttr shardingRule,
                             ArrayRef<int64_t> tensorSizes,
                             const SymbolTable& symbolTable, const Mesh& mesh) {
  // Return without inserting reshards if any factor sharding has overflow
  // axes. This case is not handled yet.
  // TODO(enver): Handle the case when factor shardings have overflow axes.
  if (hasOverflowAxes(shardingProjection)) {
    return AxesPerFactor();
  }

  // Checks if factors are sharded the same way across operands and results.
  if (std::optional<AxesPerFactor> commonAxesPerFactor =
          getCompatibleFactorShardings(shardingProjection, shardingRule)) {
    return std::move(commonAxesPerFactor.value());
  }

  // Handle the special case of unary operations without factors that need
  // replication. Reshard only one of the tensors.
  if (shardingRule.getNonScalarTensorIndices().size() == 2 &&
      shardingRule.getNeedReplicationFactors().empty() &&
      !shardingRule.hasDimensionsWithMultipleFactors()) {
    return findCommonAxesOnUnaryOperation(inShardings, outShardings,
                                          shardingProjection, shardingRule,
                                          tensorSizes, symbolTable, mesh);
  }

  AxesPerFactor factorCommonAxes = findCommonAxesUsingMajorityVoteHeuristic(
      shardingProjection, shardingRule, tensorSizes, mesh);

  // Distribute the greatest common prefix of shardings of factors that need
  // replication to batching factors.
  AxesPerFactor greatestCommonPrefixShardings =
      shardingProjection.getGreatestCommonPrefixAxes(
          shardingRule.getNumFactors());
  for (const int64_t factorIndex : shardingRule.getNeedReplicationFactors()) {
    SmallVector<AxisRefAttr> axisRefsToDistribute =
        greatestCommonPrefixShardings[factorIndex];
    if (shardingRule.isFactorInAllNonScalarTensors(factorIndex) &&
        !axisRefsToDistribute.empty()) {
      // TODO(enver): Instead of the greatest common prefix, explore options
      // to distribute more.
      distributeAxisRefsToBatchingFactors(axisRefsToDistribute, shardingRule,
                                          mesh, factorCommonAxes);
    }
  }

  return factorCommonAxes;
}

SmallVector<int64_t> getTensorSizes(Operation* op) {
  SmallVector<int64_t> tensorSizes;
  tensorSizes.reserve(op->getNumOperands() + op->getNumResults());
  for (Type type :
       llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
    ShapedType shapedType = dynCastStaticShapedType(type);
    // Assign zero as the tensor size for dynamically shaped types.
    tensorSizes.push_back(shapedType ? shapedType.getNumElements() : 0);
  }
  return tensorSizes;
}
}  // namespace

bool shouldReshard(TensorShardingAttr sourceSharding,
                   TensorShardingAttr targetSharding) {
  if (isFullyReplicated(sourceSharding) && isFullyReplicated(targetSharding)) {
    return false;
  }
  // At this point we assume a difference between source and target unreduced
  // axes has been resolved by inserting an all-reduce.
  return sourceSharding != targetSharding;
}

TensorShardingAttr insertAllReduceIfUnreducedToReplicated(
    OpOperand& use, TensorShardingAttr sourceSharding,
    TensorShardingAttr userSharding, const SymbolTable& symbolTable,
    IRRewriter& rewriter) {
  if (!sourceSharding || sourceSharding.getUnreducedAxes().empty()) {
    return sourceSharding;
  }
  MeshAttr mesh = sourceSharding.getMesh(symbolTable);
  ArrayRef<AxisRefAttr> sourceUnreducedAxes = sourceSharding.getUnreducedAxes();
  ArrayRef<AxisRefAttr> targetUnreducedAxes;
  if (userSharding) {
    targetUnreducedAxes = userSharding.getUnreducedAxes();
    // TODO(enver): Support the case the meshes differ only on device orders.
    // NOTE: At this point, it is guaranteed that source unreduced axes is
    // non-empty.
    SDY_CHECK(mesh.equals(userSharding.getMesh(symbolTable)))
        << "source and user shardings have different meshes for unreduced "
           "axes.";
  }
  SmallVector<AxisRefAttr> allReduceAxes =
      getAxisSetDiff(sourceUnreducedAxes, targetUnreducedAxes, mesh);
  if (allReduceAxes.empty()) {
    return sourceSharding;
  }
  // TODO(enver): Get this info from a single `getAxisSetDiff` call.
  SDY_CHECK(
      getAxisSetDiff(targetUnreducedAxes, sourceUnreducedAxes, mesh).empty())
      << "Both source and target have axes that does not appear in the other.";
  TensorShardingAttr allReduceSharding =
      sourceSharding.replaceUnreducedAxes(targetUnreducedAxes);
  auto allReduceOp =
      AllReduceOp::create(rewriter, use.get().getLoc(), use.get(),
                          allReduceAxes, allReduceSharding);
  use.set(allReduceOp);
  return allReduceSharding;
}

std::optional<ArrayRef<AxisRefAttr>> getFactorSharding(
    const TensorFactorShardings& factorShardings, int64_t factorIndex) {
  if (auto it = factorShardings.factorIndexToSharding.find(factorIndex);
      it != factorShardings.factorIndexToSharding.end()) {
    return it->second.axisRefs;
  }
  return std::nullopt;
}

ArrayRef<AxisRefAttr> getUnreducedAxes(TensorShardingAttr sharding) {
  return sharding ? sharding.getUnreducedAxes() : ArrayRef<AxisRefAttr>();
}

ArrayRef<AxisRefAttr> getUnreducedAxes(Value value) {
  return getUnreducedAxes(getSharding(value));
}

void insertAllReducesForRedcutionFactors(
    Operation* op, const AxesPerFactor& commonAxesPerFactor, const Mesh& mesh,
    OpShardingRuleAttr shardingRule, IRRewriter& rewriter) {
  if (op->getResults().empty()) {
    return;
  }

  rewriter.setInsertionPointAfter(op);
  SmallVector<AxisRefAttr> commonReductionAxes;
  for (int64_t reductionFactor : shardingRule.getReductionFactors()) {
    commonReductionAxes.append(commonAxesPerFactor[reductionFactor]);
  }

  // The first result unreduced axes is also the common one.
  SmallVector<AxisRefAttr> allReduceAxes = getAxisSetDiff(
      commonReductionAxes, getUnreducedAxes(op->getResult(0)), mesh.attr());
  if (allReduceAxes.empty()) {
    return;
  }

  // TODO(tomnatan): consider supporting multi-input all-reduce op.
  for (Value result : op->getResults()) {
    TensorShardingAttr resultSharding =
        getOrCreateSharding(result, mesh.name(),
                            /*closedIfMissing=*/true);
    auto allReduceOp = AllReduceOp::create(rewriter, result.getLoc(), result,
                                           allReduceAxes, resultSharding);
    rewriter.replaceAllUsesExcept(result, allReduceOp, allReduceOp);
  }
}

void insertExplicitReshardsOnOp(Operation* op,
                                ArrayRef<TensorShardingAttr> inShardings,
                                ArrayRef<TensorShardingAttr> outShardings,
                                IRRewriter& rewriter,
                                const SymbolTable& symbolTable,
                                OpShardingRuleAttr shardingRule,
                                const Mesh& mesh) {
  ShardingProjection shardingProjection = ShardingProjection::build(
      inShardings, outShardings, shardingRule, mesh.attr(),
      /*closedIfMissing=*/true);

  UpdateTensorShardings updateTensorShardings(shardingRule.getNumOperands(),
                                              shardingRule.getNumResults());
  AxesPerFactor commonAxesPerFactor =
      findCommonAxes(inShardings, outShardings, shardingProjection,
                     shardingRule, getTensorSizes(op), symbolTable, mesh);
  if (commonAxesPerFactor.empty()) {
    return;
  }
  for (const auto& [index, axes] : llvm::enumerate(commonAxesPerFactor)) {
    // TODO(enver): Add unit tests to test overflow axes are cleared after
    // handling the case that some factors have overflow axes.
    updateTensorShardings |=
        shardingProjection.updateSharding(index, axes, /*overflowAxes=*/{});
  }
  insertExplicitReshards(op, inShardings, outShardings, shardingProjection,
                         updateTensorShardings, rewriter, shardingRule,
                         symbolTable, mesh);

  // TODO(b/440055868): Insert a reshard from unreduced to replicated axes.
  insertAllReducesForRedcutionFactors(op, commonAxesPerFactor, mesh,
                                      shardingRule, rewriter);
}

}  // namespace sdy
}  // namespace mlir
