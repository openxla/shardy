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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/axis_list_ref.h"
#include "shardy/dialect/sdy/ir/constants.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"    // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/utils.h"      // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"
#include "stablehlo/dialect/StablehloOps.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_INSERTEXPLICITRESHARDSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// Returns true iff any tensor factor sharding has non-empty overflow axes.
bool hasOverflowAxes(const ShardingProjection& projection) {
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    for (const auto& [_, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      if (!factorSharding.overflowAxes.empty()) {
        return true;
      }
    }
  }
  return false;
}

// Checks if factor sharding is compatible, that is, it satisfies:
// 1. Factors are sharded the same way across operands and results.
//
// Assumes factor shardings do not have overflow axes.
// TODO(enver): Handle the case when some factor shardings have overflow axes.
bool hasCompatibleFactorShardings(const ShardingProjection& projection) {
  FactorIndexToSharding factorIndexToCommonSharding;
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    // Detects conflicts within the same factor.
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      auto commonFactorShardingIt =
          factorIndexToCommonSharding.find(factorIndex);
      if (commonFactorShardingIt == factorIndexToCommonSharding.end()) {
        factorIndexToCommonSharding[factorIndex] = factorSharding;
        continue;
      }
      if (factorSharding.axisRefs != commonFactorShardingIt->second.axisRefs) {
        return false;
      }
    }
  }

  // TODO(enver): Detect conflicts across different factors.
  return true;
}

// Insert explicit reshards for operands and results that change by
// the given `projection` for a given `op`. The reshards are inserted only to
// make the given operation compatible.
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
// after a call on the stablehlo.dot operation, by the projection, i: {}, j: {},
// k: {"y"}, the module becomes:
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
void insertExplicitReshards(Operation* op, const ShardingProjection& projection,
                            UpdateTensorShardings updateTensorShardings,
                            IRRewriter& rewriter,
                            OpShardingRuleAttr shardingRule, StringRef meshName,
                            MeshAttr mesh) {
  rewriter.setInsertionPoint(op);
  for (int operandIndex : updateTensorShardings.updateOperands.set_bits()) {
    auto operand = op->getOperand(operandIndex);
    auto newTensorSharding =
        projection.getOperand(operandIndex)
            .createTensorShardingAttr(
                mesh.getContext(), shardingRule.getOperandMapping(operandIndex),
                shardingRule.getFactorSizes(), meshName, mesh);
    auto reshardOp = rewriter.create<ReshardOp>(operand.getLoc(), operand,
                                                newTensorSharding);
    op->setOperand(operandIndex, reshardOp);
  }

  rewriter.setInsertionPointAfter(op);
  for (int resultIndex : toSetBitsVector(updateTensorShardings.updateResults)) {
    auto result = op->getResult(resultIndex);
    auto newTensorSharding =
        projection.getResult(resultIndex)
            .createTensorShardingAttr(
                mesh.getContext(), shardingRule.getResultMapping(resultIndex),
                shardingRule.getFactorSizes(), meshName, mesh);
    auto reshardOp = rewriter.create<ReshardOp>(result.getLoc(), result,
                                                getSharding(result));
    rewriter.replaceAllUsesExcept(result, reshardOp, reshardOp);
    setSharding(result, newTensorSharding);
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

  // Checks if any two axes, one from this, and the other from `rhs`, overlap.
  bool overlaps(const FactorAxesPair& rhs) const {
    return axes.overlaps(rhs.axes);
  }

  void assignTo(SmallVector<AxisListRef>& axesPerFactor) const {
    axesPerFactor[factorIndex] = axes;
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
  int64_t count = 0;
  // The size of the source tensor. In case the factor-axes pair has multiple
  // source tensors, the size of the largest one. A tensor is a source for a
  // factor-axes pair if the axes is a prefix of the factor sharding on the
  // tensor.
  int64_t sourceTensorSize = 0;
  // The size of axes to shard further. Hence, if the factor is already assigned
  // to axes A, and this factor-axes pair has axes B, the size of further
  // sharding is size(B)/size(A), and where A is a strict prefix of B.
  int64_t shardingSize = 0;

  FactorAxesCandidate(FactorAxesPair factorAxes, int64_t count,
                      int64_t sourceTensorSize, int64_t shardingSize)
      : factorAxes(factorAxes),
        count(count),
        sourceTensorSize(sourceTensorSize),
        shardingSize(shardingSize) {}

  FactorAxesCandidate() = default;

  bool operator<(const FactorAxesCandidate& rhs) const {
    if (count != rhs.count) {
      return count < rhs.count;
    }
    if (sourceTensorSize != rhs.sourceTensorSize) {
      return sourceTensorSize < rhs.sourceTensorSize;
    }
    if (shardingSize != rhs.shardingSize) {
      return shardingSize < rhs.shardingSize;
    }
    // The following also ensures that, for two axes, if one is a strict prefix
    // of another, the strict prefix one is smaller.
    return factorAxes < rhs.factorAxes;
  }
};

using FactorAxesCandidatesMap =
    DenseMap<FactorAxesPair, FactorAxesCandidate, FactorAxesPairInfo>;

// Increment the count for the factor-axes pair, also modify source tensor size
// to keep the largest.
void updateFactorAxesCandidate(FactorAxesCandidatesMap& factorAxesCounts,
                               const FactorAxesPair& factorAxes,
                               int64_t sourceTensorSize, MeshAttr mesh) {
  if (auto factorAxesCountIt = factorAxesCounts.find(factorAxes);
      factorAxesCountIt != factorAxesCounts.end()) {
    FactorAxesCandidate& candidate = factorAxesCountIt->second;
    candidate.count++;
    candidate.sourceTensorSize =
        std::max(factorAxesCountIt->second.sourceTensorSize, sourceTensorSize);
    return;
  }
  factorAxesCounts.try_emplace(factorAxes, factorAxes, /*count=*/1,
                               sourceTensorSize,
                               factorAxes.axes.getShardingSize(mesh));
}

SmallVector<FactorAxesCandidate> findFactorAxesCandidates(
    const ShardingProjection& projection, int64_t numFactors,
    ArrayRef<int64_t> tensorSizes, MeshAttr mesh) {
  // Find sets of candidate axes per factor.
  SmallVector<DenseSet<AxisListRef, AxisListRefInfo>> axesSets(numFactors);
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
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
           projection.getOperands(), projection.getResults()))) {
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      if (factorSharding.axisRefs.empty()) {
        continue;
      }
      FactorAxesPair factorAxes(factorIndex,
                                AxisListRef(factorSharding.axisRefs));
      updateFactorAxesCandidate(factorAxesCandidatesMap, factorAxes,
                                tensorSizes[tensorIndex], mesh);
      // Increment counts for all its strict prefixes.
      for (const AxisListRef& axes : axesSets[factorIndex]) {
        if (axes.strictPrefixOf(factorAxes.axes)) {
          updateFactorAxesCandidate(factorAxesCandidatesMap,
                                    FactorAxesPair(factorIndex, axes),
                                    tensorSizes[tensorIndex], mesh);
        }
      }
    }
  }

  SmallVector<FactorAxesCandidate> factorAxesCandidates;
  for (const auto& [_, candidate] : factorAxesCandidatesMap) {
    factorAxesCandidates.push_back(candidate);
  }
  return factorAxesCandidates;
}

// Broadly the algorithm is, at each iteration, to pick a {factor,axis} pair
// with the largest count from a list that is initialized with all the
// pairs with non-zero count, assign the picked axis to the picked factor, and
// delete all the pairs from the list that is either with the picked factor,
// or with an axis that overlaps with the picked axis. Continue iterating
// until the list is empty.
SmallVector<AxisListRef> findCommonAxesUsingMajorityVoteHeuristic(
    const ShardingProjection& projection, int64_t numFactors,
    ArrayRef<int64_t> tensorSizes, MeshAttr mesh) {
  SmallVector<AxisListRef> factorAxisRefs(numFactors);
  SmallVector<FactorAxesCandidate> factorAxesCandidates =
      findFactorAxesCandidates(projection, numFactors, tensorSizes, mesh);
  // TODO(enver): Instead of taking an axes-array with the largest count, take
  // a prefix with the largest count.  For example, if a factor appears in 2
  // tensors, and one has sharding [x,y] and the other has sharding [x,z],
  // then the count of [x] prefix will be two for this factor.
  // TODO(enver): Assign an axis to a factor immediately if the count is more
  // than floor(n/2) where n is the number of tensors.
  // The first iteration is to find the initial best.
  FactorAxesPair bestFactorAxes;
  while (!factorAxesCandidates.empty()) {
    if (!bestFactorAxes.empty()) {
      bestFactorAxes.assignTo(factorAxisRefs);
    }
    // TODO(enver): Tie-breaking currently depends on the order of iteration.
    // Consider some heuristic for breaking ties.
    // Invalidate axes that overlaps with the picked one across all unseen
    // factors. During the iteration, also find the new best.
    FactorAxesCandidate nextBestFactorAxes;
    int64_t candidateIndex = 0;
    while (candidateIndex < factorAxesCandidates.size()) {
      FactorAxesCandidate& candidate = factorAxesCandidates[candidateIndex];
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
          factorAxesCandidates[candidateIndex] = factorAxesCandidates.back();
          factorAxesCandidates.pop_back();
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
          candidate.shardingSize = candidate.factorAxes.axes.getShardingSize(
              mesh,
              /*prefix=*/bestFactorAxes.axes);
          nextBestFactorAxes = std::max(nextBestFactorAxes, candidate);
          candidateIndex++;
        }
        continue;
      }
      if (candidate.factorAxes.overlaps(bestFactorAxes)) {
        // Clear the count of overlapping axis, effectively erasing.
        // TODO(enver): Instead of removing from the list, trim the axisRefs,
        // to use the largest prefix that does not overlap with bestAxisRefs.
        factorAxesCandidates[candidateIndex] = factorAxesCandidates.back();
        factorAxesCandidates.pop_back();
        continue;
      }
      nextBestFactorAxes = std::max(nextBestFactorAxes, candidate);
      candidateIndex++;
    }
    bestFactorAxes = nextBestFactorAxes.factorAxes;
  }
  return factorAxisRefs;
}

SmallVector<AxisListRef> findCommonAxes(const ShardingProjection& projection,
                                        int64_t numFactors,
                                        ArrayRef<int64_t> tensorSizes,
                                        MeshAttr mesh) {
  return findCommonAxesUsingMajorityVoteHeuristic(projection, numFactors,
                                                  tensorSizes, mesh);
}

// TODO(enver): Convert it to a OpShardingRuleAttr method.
// TODO(enver): Instead use ShapedType::getNumElements, as the factors might not
// be the exact size of the dim, e.g. concat.
SmallVector<int64_t> getTensorSizes(OpShardingRuleAttr shardingRule) {
  SmallVector<int64_t> tensorSizes;
  tensorSizes.reserve(shardingRule.getNumOperands() +
                      shardingRule.getNumResults());
  for (const TensorMappingAttr& tensorMapping :
       llvm::concat<const TensorMappingAttr>(
           shardingRule.getOperandMappings(),
           shardingRule.getResultMappings())) {
    int64_t tensorSize = 1;
    for (DimMappingAttr dimMapping : tensorMapping.getDimMappings()) {
      for (int64_t factorIndex : dimMapping.getFactorIndices()) {
        tensorSize *= shardingRule.getFactorSize(factorIndex);
      }
    }
    tensorSizes.push_back(tensorSize);
  }
  return tensorSizes;
}

struct InsertExplicitReshardsPass
    : public impl::InsertExplicitReshardsPassBase<InsertExplicitReshardsPass> {
  using InsertExplicitReshardsPassBase::InsertExplicitReshardsPassBase;

  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp);
    SymbolTable symbolTable(funcOp->getParentOfType<ModuleOp>());
    // TODO(enver): Handle data flow ops.
    funcOp.walk([&](Operation* op) {
      // TODO(enver): Check if data flow ops, data flow edge op, manual
      // computation op require extra check before creating sharding rule.

      if (isa<func::ReturnOp>(op)) {
        rewriter.setInsertionPoint(op);
        for (const auto& [index, opOperand] :
             llvm::enumerate(op->getOpOperands())) {
          Value operand = opOperand.get();
          TensorShardingAttr funcResultSharding =
              getFuncResultSharding(funcOp, index);
          TensorShardingAttr operandSharding = getSharding(operand);
          if (isFullyReplicated(operandSharding) &&
              isFullyReplicated(funcResultSharding)) {
            continue;
          }
          if (funcResultSharding != operandSharding) {
            // TODO(enver): Close all shardings and drop replicated axes before
            // this pass on the export pipeline.
            auto reshardOp = rewriter.create<ReshardOp>(
                operand.getLoc(), operand,
                funcResultSharding
                    ? funcResultSharding
                    : TensorShardingAttr::getFullyClosedLike(operandSharding));
            opOperand.set(reshardOp);
          }
        }
        return;
      }

      // NOTE: Creating a sharding rule requires data flow edges are present.
      OpShardingRuleAttr shardingRule =
          getOrCreateShardingRule(op, /*conservativePropagation=*/false,
                                  /*setShardingRuleOnOp=*/false);
      if (!shardingRule) {
        // Insert explicit reshards only on operations with sharding rules,
        // since all the operations of interest got their sharding rules.
        return;
      }
      std::optional<StringRef> meshName =
          getCommonMeshName(getShardings(op->getOperands()),
                            getShardings(op->getResults()), symbolTable);
      if (!meshName.has_value()) {
        // This means none of the operands or results have a sharding attribute
        // or the sharding attributes use different meshes. Skip if so.
        // TODO(enver): Actually, we are moving towards supporting multiple
        // explicit reshards so operands and results are all bound by the same
        // mesh.
        return;
      }

      // TODO(enver): Define get a SymbolTable at the start of the pass and use
      // that one to find meshes.
      MeshAttr mesh = getMeshAttr(op, meshName.value());
      assert(mesh && "unknown mesh");
      ShardingProjection shardingProjection =
          ShardingProjection::build(op, shardingRule, mesh);

      // Return without inserting reshards if any factor sharding has overflow
      // axes. This case is not handled yet.
      // TODO(enver): Handle the case when factor shardings have overflow axes.
      if (hasOverflowAxes(shardingProjection)) {
        return;
      }

      // Checks if factors are sharded the same way across operands and results.
      if (hasCompatibleFactorShardings(shardingProjection)) {
        return;
      }

      UpdateTensorShardings updateTensorShardings(shardingRule.getNumOperands(),
                                                  shardingRule.getNumResults());
      for (const auto& [index, axes] : llvm::enumerate(
               findCommonAxes(shardingProjection, shardingRule.getNumFactors(),
                              getTensorSizes(shardingRule), mesh))) {
        // TODO(enver): Add unit tests to test overflow axes are cleared after
        // handling the case that some factors have overflow axes.
        updateTensorShardings |= shardingProjection.updateSharding(
            index, axes.toVector(), /*overflowAxes=*/{});
      }

      insertExplicitReshards(op, shardingProjection, updateTensorShardings,
                             rewriter, shardingRule, *meshName, mesh);

      // TODO(enver): Remove sharding rules from ops.
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
