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
// 2. Factors that need replication are unsharded.
//
// Assumes factor shardings do not have overflow axes.
// TODO(enver): Handle the case when some factor shardings have overflow axes.
bool hasCompatibleFactorShardings(const ShardingProjection& projection,
                                  OpShardingRuleAttr shardingRule) {
  FactorIndexToSharding factorIndexToCommonSharding;
  // Factors that need replication should be unsharded across all operands and
  // results in order for it to have a compatible sharding.
  for (int64_t factorIndex : shardingRule.getNeedReplicationFactors()) {
    factorIndexToCommonSharding[factorIndex] = FactorSharding{};
  }
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
                shardingRule.getFactorSizes(), meshName, mesh,
                /*defaultIsClosed=*/true);
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
                shardingRule.getFactorSizes(), meshName, mesh,
                /*defaultIsClosed=*/true);
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

// A container for FactorAxesCandidates where the order of iteration does not
// matter, and provides methods to insert and remove candidates in constant-time
// while maintaining the best through explicit calls on its touchAt method.
class FactorAxesCandidateBag {
 public:
  FactorAxesCandidateBag(MeshAttr mesh) : mesh(mesh) {}

  // Returns whether the bag is empty.
  bool empty() const { return candidates.empty(); }

  // Inserts a new candidate to the bag. Performs in constant-time.
  void insert(const FactorAxesCandidate& candidate) {
    candidates.push_back(candidate);
    bestCandidate = std::max(bestCandidate, candidate);
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
    bestCandidate = std::max(bestCandidate, candidate);
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
  SmallVector<FactorAxesCandidate> candidates;
  FactorAxesCandidate bestCandidate;
  // Used for recalculating sharding size of a candidate.
  MeshAttr mesh;
};

FactorAxesCandidateBag findFactorAxesCandidates(
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

  FactorAxesCandidateBag factorAxesCandidates(mesh);
  for (const auto& [_, candidate] : factorAxesCandidatesMap) {
    factorAxesCandidates.insert(candidate);
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
  FactorAxesCandidateBag factorAxesCandidates =
      findFactorAxesCandidates(projection, numFactors, tensorSizes, mesh);
  // TODO(enver): Assign an axis to a factor immediately if the count is more
  // than floor(n/2) where n is the number of tensors.
  while (!factorAxesCandidates.empty()) {
    FactorAxesPair bestFactorAxes = factorAxesCandidates.best().factorAxes;
    factorAxesCandidates.resetBest();
    factorAxisRefs[bestFactorAxes.factorIndex] = bestFactorAxes.axes;
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
  return factorAxisRefs;
}

SmallVector<AxisListRef> findCommonAxes(const ShardingProjection& projection,
                                        int64_t numFactors,
                                        ArrayRef<int64_t> tensorSizes,
                                        MeshAttr mesh) {
  return findCommonAxesUsingMajorityVoteHeuristic(projection, numFactors,
                                                  tensorSizes, mesh);
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

      // Return without inserting reshards for operations with special
      // dimensions.
      // TODO(enver): Insert explicit reshards if special dimensions are
      // unsharded.
      // TODO(enver): Add need replication factors to fft.
      if (isa<stablehlo::ReverseOp, stablehlo::BroadcastInDimOp,
              stablehlo::DynamicSliceOp, stablehlo::DynamicUpdateSliceOp,
              stablehlo::PadOp, stablehlo::SliceOp, stablehlo::TransposeOp,
              stablehlo::FftOp, stablehlo::ReduceWindowOp, stablehlo::ScatterOp,
              stablehlo::SelectAndScatterOp, stablehlo::GatherOp,
              stablehlo::ReshapeOp, stablehlo::ConvolutionOp,
              stablehlo::CustomCallOp, stablehlo::ReduceOp,
              stablehlo::AllReduceOp, stablehlo::AllGatherOp,
              stablehlo::AllToAllOp, stablehlo::CollectivePermuteOp,
              stablehlo::ClampOp>(op)) {
        return;
      }

      // Checks if factors are sharded the same way across operands and results.
      if (hasCompatibleFactorShardings(shardingProjection, shardingRule)) {
        return;
      }

      // Return without inserting reshards for operations with factors that need
      // replication.
      // TODO(enver): Insert explicit reshards also for the case that the
      // factors that need replication are sharded.
      if (isa<stablehlo::CholeskyOp, stablehlo::BitcastConvertOp,
              stablehlo::ConcatenateOp, stablehlo::SortOp,
              stablehlo::TriangularSolveOp>(op)) {
        return;
      }

      UpdateTensorShardings updateTensorShardings(shardingRule.getNumOperands(),
                                                  shardingRule.getNumResults());
      for (const auto& [index, axes] : llvm::enumerate(
               findCommonAxes(shardingProjection, shardingRule.getNumFactors(),
                              shardingRule.getTensorSizes(), mesh))) {
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
