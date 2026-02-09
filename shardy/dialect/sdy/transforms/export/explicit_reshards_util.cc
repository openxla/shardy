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
#include <tuple>
#include <utility>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
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

namespace {
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

// Returns the common axes per factor if the factor sharding is compatible.
// Otherwise, returns empty AxesPerFactor.
//
// The factor sharding is compatible if it satisfies:
// 1. Factors are sharded the same way across operands and results.
// 2. Factors that need replication are unsharded.
// 3. There is no overlap between the sharding axes across different factors.
//
// Assumes factor shardings do not have overflow axes.
// TODO(enver): Handle the case when some factor shardings have overflow axes.
AxesPerFactor getCompatibleFactorShardings(
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule) {
  AxesPerFactor commonAxesPerFactor(shardingRule.getNumFactors());
  BitVector seenFactors(shardingRule.getNumFactors());
  SmallVector<AxisRefAttr> seenAxisRefs;
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(
           shardingProjection.getOperands(), shardingProjection.getResults())) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      // Factors that need replication should be unsharded to be compatible.
      if (shardingRule.isNeedReplicationFactor(factorIndex)) {
        if (!factorSharding.axisRefs.empty()) {
          return {};
        }
        continue;
      }
      if (!seenFactors.test(factorIndex)) {
        if (overlaps(factorSharding.axisRefs, seenAxisRefs)) {
          return {};
        }
        commonAxesPerFactor[factorIndex] = factorSharding.axisRefs;
        seenAxisRefs.append(factorSharding.axisRefs);
        seenFactors.set(factorIndex);
      } else if (factorSharding.axisRefs != commonAxesPerFactor[factorIndex]) {
        return {};
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
}  // namespace

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

namespace {
struct FactorAxesPair {
  constexpr static int64_t kEmptyFactorIndex = -1;
  constexpr static int64_t kTombstoneFactorIndex = -2;

  int64_t factorIndex = kEmptyFactorIndex;
  AxisListRef axes;

  FactorAxesPair(int64_t factorIndex, ArrayRef<AxisRefAttr> axisRefs)
      : factorIndex(factorIndex), axes(AxisListRef(axisRefs)) {}

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

  bool isFullySharded(OpShardingRuleAttr shardingRule, MeshAttr mesh) const {
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
  // The total global size of the source tensors.
  int64_t totalGlobalSourceTensorSize = 0;
  // The size of axes to shard further. Hence, if the factor is already assigned
  // to axes A, and this factor-axes pair has axes B, the size of further
  // sharding is size(B)/size(A), and where A is a strict prefix of B.
  int64_t shardingSize = 0;
  int64_t factorTypePrecedence = 0;

  FactorAxesCandidate(FactorAxesPair factorAxes, int64_t shardingSize,
                      FactorType factorType)
      : factorAxes(factorAxes),
        shardingSize(shardingSize),
        factorTypePrecedence(precedence(factorType)) {}

  FactorAxesCandidate() = default;

  // Multi-level comparison.
  // 1. totalGlobalSourceTensorSize
  // 2. factorTypePrecedence
  // 3. shardingSize
  // 4. factorAxes: If A is a strict prefix of B, then A is smaller than B.
  bool operator<(const FactorAxesCandidate& rhs) const {
    auto makeComparisonTuple = [](const FactorAxesCandidate& candidate) {
      return std::make_tuple(candidate.totalGlobalSourceTensorSize,
                             candidate.factorTypePrecedence,
                             candidate.shardingSize, candidate.factorAxes);
    };
    return makeComparisonTuple(*this) < makeComparisonTuple(rhs);
  }

  // A candidate with a higher precedence will be preferred (given their source
  // tensor sizes are the same) to a candidate with a lower precedence when
  // finding the best candidate to extend the factor sharding assignment.
  int64_t precedence(FactorType factorType) const {
    switch (factorType) {
      case FactorType::kPassThrough:
        return 3;
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

// A container for FactorAxesCandidates where the order of iteration does not
// matter, and provides methods to insert and remove candidates in constant-time
// while maintaining the best candidate.
class FactorAxesCandidateBag {
 public:
  FactorAxesCandidateBag(MeshAttr mesh, OpShardingRuleAttr shardingRule)
      : mesh(mesh) {
    initFactorDependencies(shardingRule);
  }

  // Returns whether the bag is empty.
  bool empty() const { return candidates.empty(); }

  // Inserts a new candidate to the bag. Performs in constant-time.
  void insert(const FactorAxesPair& factorAxes,
              OpShardingRuleAttr shardingRule) {
    candidates.emplace_back(factorAxes, factorAxes.axes.getShardingSize(mesh),
                            shardingRule.getFactorType(factorAxes.factorIndex));
  }

  // Updates the sharding size of the one at index as the  product of the
  // sharding sizes of all individual axes excluding the `prefix`.
  //
  // Assumes `prefix` is a prefix of the axes of the candidate at index.
  void updateShardingSizeAt(const int64_t index,
                            const AxisListRef& prefix = AxisListRef()) {
    FactorAxesCandidate& candidate = candidates[index];
    candidate.shardingSize =
        candidate.factorAxes.axes.getExpandedShardingSize(mesh, prefix);
  }

  // TODO(enver): Optimize by grouping candidates on the same factors.
  void updateTotalGlobalSourceTensorSizes(
      const int64_t sourceFactorIndex,
      ArrayRef<AxisRefAttr> sourceFactorAxisRefs,
      const int64_t sourceTensorSize) {
    AxisListRef sourceFactorAxes(sourceFactorAxisRefs);
    for (FactorAxesCandidate& candidate : candidates) {
      FactorAxesPair& factorAxesPair = candidate.factorAxes;
      if (factorAxesPair.factorIndex == sourceFactorIndex &&
          (sourceFactorAxes == factorAxesPair.axes ||
           factorAxesPair.axes.strictPrefixOf(sourceFactorAxes))) {
        candidate.totalGlobalSourceTensorSize += sourceTensorSize;
      }
    }
  }

  FactorAxesCandidate findBestCandidate() {
    FactorAxesCandidate bestCandidate;
    for (FactorAxesCandidate& candidate : candidates) {
      // The axes on replication factors are distributed to batching dimensions
      // after the common axes are found for all non-replication factors.
      if (isValid(candidate)) {
        bestCandidate = std::max(bestCandidate, candidate);
      }
    }
    return bestCandidate;
  }

  void dropFactorDependencies(const int64_t factorIndex) {
    for (auto& [_, factorDependencies] : factorDependenciesMap) {
      factorDependencies.reset(factorIndex);
    }
  }

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

  // Returns the candidate at index. Performs in constant-time.
  FactorAxesCandidate& at(const int64_t index) { return candidates[index]; }
  // Returns the number of candidates in the bag.
  int64_t size() const { return candidates.size(); }
  bool isValid(const FactorAxesCandidate& candidate) {
    auto it = factorDependenciesMap.find(candidate.factorAxes.factorIndex);
    return it == factorDependenciesMap.end() || it->second.none();
  }

 private:
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
          factorDependenciesMap
              .try_emplace(factorIndex, shardingRule.getNumFactors())
              .first->second.set(dependsOn);
        }
      }
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
  // Used for recalculating sharding size of a candidate.
  MeshAttr mesh;
};

FactorAxesCandidateBag findFactorAxesCandidates(
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule, ArrayRef<int64_t> tensorSizes,
    MeshAttr mesh) {
  // TODO(enver): For two factor-axes pairs, if both have the same factor and
  // the same count, and one is the prefix of the other, drop the prefix one.

  // Count factor-axes pairs by iterating through each sharding, and for each
  // sharding, update candidate for the sharding and all its prefixes.
  DenseSet<FactorAxesPair, FactorAxesPairInfo> factorAxesPairs;
  for (const auto& [tensorSize, tensorFactorSharding] :
       llvm::zip_equal(tensorSizes, llvm::concat<const TensorFactorShardings>(
                                        shardingProjection.getOperands(),
                                        shardingProjection.getResults()))) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      ArrayRef<AxisRefAttr> axisRefs = factorSharding.axisRefs;
      while (!axisRefs.empty()) {
        factorAxesPairs.insert(FactorAxesPair(factorIndex, axisRefs));
        axisRefs = axisRefs.drop_back();
      }
    }
  }

  FactorAxesCandidateBag factorAxesCandidates(mesh, shardingRule);
  for (const FactorAxesPair& factorAxes : factorAxesPairs) {
    factorAxesCandidates.insert(factorAxes, shardingRule);
  }

  // Set total global source tensor sizes of candidates.
  for (const auto& [tensorSize, tensorFactorSharding] :
       llvm::zip_equal(tensorSizes, llvm::concat<const TensorFactorShardings>(
                                        shardingProjection.getOperands(),
                                        shardingProjection.getResults()))) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      factorAxesCandidates.updateTotalGlobalSourceTensorSizes(
          factorIndex, factorSharding.axisRefs, tensorSize);
    }
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
//
// Guarantees to return a non-empty AxesPerFactor.
AxesPerFactor findCommonAxesHeuristic(
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule, ArrayRef<int64_t> tensorSizes,
    const Mesh& mesh) {
  SmallVector<AxisListRef> factorAxisRefs(shardingRule.getNumFactors());
  FactorAxesCandidateBag factorAxesCandidates = findFactorAxesCandidates(
      shardingProjection, shardingRule, tensorSizes, mesh.attr());
  while (!factorAxesCandidates.empty()) {
    FactorAxesCandidate bestCandidate =
        factorAxesCandidates.findBestCandidate();
    // TODO(enver): If there is no best canditate at this point, it means the
    // candidate set is nonempty but all invalid. Investigate how this happens.
    if (bestCandidate.empty()) {
      return toAxesPerFactor(factorAxisRefs);
    }
    // TODO(enver): Instead of finding the best candidate by a linear search at
    // the beginning of each iteration, keep the best on `factorAxesCandidates`
    // bag, through an internal priority queue, when a candidate becomes valid
    // and/or gets modified.
    FactorAxesPair bestFactorAxes = bestCandidate.factorAxes;
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
  }

  // TODO(enver): Consider to keep factorAxisRefs for longer until actual
  // needed to call toVector.
  return toAxesPerFactor(factorAxisRefs);
}

// Returns a pair of tensor indices sorted by preference for a unary operation.
// The first element is the preferred tensor index. The preference is determined
// by the following criteria in order:
// 1. The tensor without sharded permutation factors.
// 2. The tensor with a larger size.
// 3. The tensor with a larger sharding size.
std::pair<int64_t, int64_t> sortedTensorIndicesOnUnaryOperation(
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule, ArrayRef<int64_t> tensorSizes,
    const Mesh& mesh) {
  SmallVector<int64_t> tensorIndices = shardingRule.getNonScalarTensorIndices();
  const int64_t lhs = tensorIndices[0];
  const int64_t rhs = tensorIndices[1];

  auto getPreferenceTuple =
      [&](int64_t tensorIndex) -> std::tuple<bool, int64_t, int64_t> {
    const TensorFactorShardings& tensor =
        shardingProjection.getTensor(tensorIndex);
    return {!hasShardedPermutationFactors(tensor, shardingRule),
            tensorSizes[tensorIndex], tensor.getShardingSize(mesh.attr())};
  };

  if (getPreferenceTuple(lhs) >= getPreferenceTuple(rhs)) {
    return {lhs, rhs};
  }
  return {rhs, lhs};
}

// Assumes that:
// 1. Either tensor does not have factors that need replication.
// 2. Both tensors have the same mesh but may have different device orders.
// 3. The factor shardings are not compatible.
//
// Guarantees to return a non-empty AxesPerFactor.
AxesPerFactor findCommonAxesOnUnaryOperation(
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule, ArrayRef<int64_t> tensorSizes,
    const Mesh& mesh) {
  const auto [preferredTensor, otherTensor] =
      sortedTensorIndicesOnUnaryOperation(shardingProjection, shardingRule,
                                          tensorSizes, mesh);
  const FactorIndexToSharding& preferredFactorSharding =
      shardingProjection.getTensor(preferredTensor).factorIndexToSharding;
  const FactorIndexToSharding& otherFactorSharding =
      shardingProjection.getTensor(otherTensor).factorIndexToSharding;

  AxesPerFactor factorAxisRefs(shardingRule.getNumFactors());
  SmallVector<AxisRefAttr> axesInPreferredTensor;
  for (const auto& [factorIndex, factorSharding] : preferredFactorSharding) {
    factorAxisRefs[factorIndex] = factorSharding.axisRefs;
    axesInPreferredTensor.append(factorSharding.axisRefs.begin(),
                                 factorSharding.axisRefs.end());
  }

  for (const auto& [factorIndex, factorSharding] : otherFactorSharding) {
    if (!preferredFactorSharding.contains(factorIndex)) {
      factorAxisRefs[factorIndex] = factorSharding.axisRefs;
      truncateAxesByRemovingOverlaps(factorAxisRefs[factorIndex],
                                     axesInPreferredTensor);
    }
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

// Distribute the greatest common prefix of shardings of factors that need
// replication to batching factors.
void distributeAxisRefsToBatchingFactors(
    OpShardingRuleAttr shardingRule, const Mesh& mesh,
    AxesPerFactor& factorCommonAxes) {
  for (const int64_t factorIndex : shardingRule.getNeedReplicationFactors()) {
    SmallVector<AxisRefAttr> axisRefsToDistribute =
        factorCommonAxes[factorIndex];
    factorCommonAxes[factorIndex].clear();
    if (shardingRule.isFactorInAllNonScalarTensors(factorIndex) &&
        !axisRefsToDistribute.empty()) {
      distributeAxisRefsToBatchingFactors(axisRefsToDistribute, shardingRule,
                                          mesh, factorCommonAxes);
    }
  }
}
}  // namespace

AxesPerFactor findCommonAxes(const ShardingProjection& shardingProjection,
                             OpShardingRuleAttr shardingRule,
                             ArrayRef<int64_t> tensorSizes, const Mesh& mesh) {
  if (AxesPerFactor compatibleFactorShardings =
          getCompatibleFactorShardings(shardingProjection, shardingRule);
      !compatibleFactorShardings.empty()) {
    return compatibleFactorShardings;
  }

  // Handle the special case of unary operations without factors that need
  // replication. Reshard only one of the tensors.
  if (shardingRule.getNonScalarTensorIndices().size() == 2 &&
      shardingRule.getNeedReplicationFactors().empty() &&
      !shardingRule.hasDimensionsWithMultipleFactors()) {
    return findCommonAxesOnUnaryOperation(shardingProjection, shardingRule,
                                          tensorSizes, mesh);
  }

  AxesPerFactor factorCommonAxes = findCommonAxesHeuristic(
      shardingProjection, shardingRule, tensorSizes, mesh);

  if (!shardingRule.getNeedReplicationFactors().empty()) {
    distributeAxisRefsToBatchingFactors(shardingRule, mesh, factorCommonAxes);
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

namespace {

// Returns reduction axes that are the union of all axes on reduction factors.
// The result axes are not necessarily canonicalized.
//
// Returns empty axes if not `onFullVersion` and op results do not have
// unreduced axes.
//
// Assumes `commonAxesPerFactor` is non-empty if `onFullVersion` is true.
//
// Hard fails if some reduction factors do not have compatible shardings.
SmallVector<AxisRefAttr> getReductionAxes(
    Operation* op, const ShardingProjection& shardingProjection,
    const AxesPerFactor& commonAxesPerFactor, OpShardingRuleAttr shardingRule,
    const bool onFullVersion) {
  if (onFullVersion) {
    SmallVector<AxisRefAttr> reductionAxes;
    for (int64_t reductionFactor : shardingRule.getReductionFactors()) {
      reductionAxes.append(commonAxesPerFactor[reductionFactor]);
    }
    return reductionAxes;
  }

  if (getUnreducedAxes(op->getResult(0)).empty()) {
    return {};
  }

  // TODO(enver): Repurpose getCompatibleFactorShardings to return compatible
  // factors, and simplify the following logic.
  SmallVector<AxisRefAttr> axesAlongAllReductionFactors;
  for (int64_t reductionFactor : shardingRule.getReductionFactors()) {
    // We only iterate operands since reduction factors are not in results.
    bool seen = false;
    SmallVector<AxisRefAttr> axesAlongCurrentReductionFactor;
    for (const TensorFactorShardings& tensorFactorSharding :
         shardingProjection.getOperands()) {
      if (std::optional<ArrayRef<AxisRefAttr>> factorSharding =
              getFactorSharding(tensorFactorSharding, reductionFactor)) {
        if (seen) {
          SDY_CHECK(axesAlongCurrentReductionFactor == *factorSharding)
              << "For the operation " << op
              << ", the result has unreduced axes while the operand has "
                 "incompatible sharding along reduction factors.";
        } else {
          axesAlongCurrentReductionFactor = llvm::to_vector(*factorSharding);
          seen = true;
        }
      }
    }
    axesAlongAllReductionFactors.append(axesAlongCurrentReductionFactor);
  }
  return axesAlongAllReductionFactors;
}
}  // namespace

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
  SDY_CHECK(
      llvm::is_sorted(allReduceAxes, AxisRefAttr::getMeshComparator(mesh)));
  TensorShardingAttr allReduceSharding = sourceSharding.replaceUnreducedAxes(
      getAxisSetDiff(sourceUnreducedAxes, allReduceAxes, mesh));
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

void insertAllReducesForReductionFactors(
    Operation* op, const ShardingProjection& shardingProjection,
    const AxesPerFactor& commonAxesPerFactor, OpShardingRuleAttr shardingRule,
    const Mesh& mesh, IRRewriter& rewriter, const bool onFullVersion) {
  if (op->getResults().empty()) {
    return;
  }
  SmallVector<AxisRefAttr> reductionAxes = getReductionAxes(
      op, shardingProjection, commonAxesPerFactor, shardingRule, onFullVersion);
  if (reductionAxes.empty()) {
    return;
  }

  // The first result unreduced axes is also the common one.
  SmallVector<AxisRefAttr> allReduceAxes = getAxisSetDiff(
      reductionAxes, getUnreducedAxes(op->getResult(0)), mesh.attr());
  if (allReduceAxes.empty()) {
    return;
  }

  sortAndMergeAxes(allReduceAxes, mesh.attr());

  // TODO(tomnatan): consider supporting multi-input all-reduce op.
  rewriter.setInsertionPointAfter(op);
  for (Value result : op->getResults()) {
    TensorShardingAttr resultSharding =
        getOrCreateSharding(result, mesh.name(),
                            /*closedIfMissing=*/true);
    auto allReduceOp = AllReduceOp::create(rewriter, result.getLoc(), result,
                                           allReduceAxes, resultSharding);
    rewriter.replaceAllUsesExcept(result, allReduceOp, allReduceOp);
  }
}

bool convertReshardToUnreducedCollectives(Operation* op, IRRewriter& rewriter,
                                          const SymbolTable& symbolTable) {
  ReshardOp reshardOp = dyn_cast<ReshardOp>(op);
  if (!reshardOp) {
    return false;
  }

  Value input = reshardOp.getInput();
  TensorShardingAttr inSharding = getSharding(input);
  TensorShardingAttr outSharding = reshardOp.getSharding();
  if (!inSharding || !outSharding) {
    return false;
  }

  ArrayRef<AxisRefAttr> inUnreducedAxes = inSharding.getUnreducedAxes();
  ArrayRef<AxisRefAttr> outUnreducedAxes = outSharding.getUnreducedAxes();
  if (outUnreducedAxes.empty()) {
    return false;
  }

  MeshAttr inMesh = inSharding.getMesh(symbolTable);
  MeshAttr outMesh = outSharding.getMesh(symbolTable);
  SDY_CHECK(inMesh.equals(outMesh))
      << "Reshard op has different meshes for input and output. The result has "
         "non-empty unreduced axes.";

  // The relationship of the unreduced axes is "out = in + r2u + s2u", where
  // "r2u" is the replicated-to-unreduced axes and "s2u" is the
  // sharded-to-unreduced axes.
  SmallVector<AxisRefAttr> r2uAnds2uAxes =
      getAxisSetDiff(outUnreducedAxes, inUnreducedAxes, inMesh);
  if (r2uAnds2uAxes.empty()) {
    return false;
  }

  SDY_CHECK(getAxisSetDiff(inUnreducedAxes, outUnreducedAxes, inMesh).empty())
      << "Both input and output have unreduced axes that does not appear in "
         "the other.";
  SDY_CHECK(isa<BlockArgument>(input) || input.getDefiningOp<ReshardOp>())
      << "Input of sharded-to-unreduced reshard must be a block argument or a "
         "reshard op.";

  SmallVector<AxisRefAttr> s2uAxes;
  SmallVector<AxisRefListAttr> axesPerDim(inSharding.getRank());
  for (auto [inDimSharding, outDimSharding, axes] :
       llvm::zip_equal(inSharding.getDimShardings(),
                       outSharding.getDimShardings(), axesPerDim)) {
    ArrayRef<AxisRefAttr> inAxes = inDimSharding.getAxes();
    ArrayRef<AxisRefAttr> outAxes = outDimSharding.getAxes();
    PrefixStatus prefixStatus = isAxisListPrefixOf(outAxes, inAxes);
    if (prefixStatus == PrefixStatus::EQUAL) {
      axes = AxisRefListAttr::get(rewriter.getContext(), {});
    } else if (prefixStatus == PrefixStatus::STRICT_PREFIX) {
      SmallVector<AxisRefAttr> diff;
      if (!outAxes.empty() && outAxes.back() != inAxes[outAxes.size() - 1]) {
        std::optional<AxisRefAttr> suffix =
            inAxes[outAxes.size() - 1].getSuffixWithoutOverlap(outAxes.back(),
                                                               inMesh);
        SDY_CHECK(suffix);
        diff.push_back(*suffix);
      }
      diff.append(inAxes.begin() + outAxes.size(), inAxes.end());
      axes = AxisRefListAttr::get(rewriter.getContext(), diff);
      s2uAxes.append(diff);
    } else {
      SDY_LOG(FATAL)
          << "The reshard op needs to be decomposed to a sharded-to-unreduced "
             "AND other collective ops, which is not supported yet.";
    }
  }

  rewriter.setInsertionPoint(reshardOp);
  Value result = input;

  SmallVector<AxisRefAttr> r2uAxes =
      getAxisSetDiff(r2uAnds2uAxes, s2uAxes, inMesh);
  if (!r2uAxes.empty()) {
    SmallVector<AxisRefAttr> inPlusR2uAxes = llvm::to_vector(inUnreducedAxes);
    inPlusR2uAxes.append(r2uAxes.begin(), r2uAxes.end());
    sortAndMergeAxes(inPlusR2uAxes, inMesh);
    TensorShardingAttr r2uSharding =
        TensorShardingAttr::get(rewriter.getContext(), inSharding.getMeshName(),
                                inSharding.getDimShardings(),
                                outSharding.getReplicatedAxes(), inPlusR2uAxes);
    result = ReplicatedToUnreducedOp::create(rewriter, reshardOp.getLoc(),
                                             result, r2uAxes, r2uSharding);
  }
  if (!s2uAxes.empty()) {
    result = ShardedToUnreducedOp::create(rewriter, reshardOp.getLoc(), result,
                                          axesPerDim, outSharding);
  }

  rewriter.replaceOp(reshardOp, result);
  return true;
}

}  // namespace sdy
}  // namespace mlir
