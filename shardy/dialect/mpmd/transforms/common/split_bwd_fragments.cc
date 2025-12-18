/* Copyright 2025 The MPMD Authors.

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
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/optimize/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_SPLITBWDFRAGMENTSPASS
#define GEN_PASS_DEF_SPLITANDPRIORITIZETRANSFERINDEPENDENTCOMPUTATIONSPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

bool IsBackwardsCandidateFragment(FragmentOp fragment) {
  return IsBackwardFragment(fragment) &&
         // The next are not strictly needed but make the pass idempotent.
         !IsSplitDropTransferred(fragment) && !IsSplitKeepTransferred(fragment);
}

// Finds the results that do not have any transfer ops as users.
BitVector FindNonTransferredResults(FragmentOp fragment) {
  BitVector mask;
  mask.reserve(fragment->getNumResults());
  for (OpResult value : fragment.getResults()) {
    mask.push_back(llvm::all_of(
        value.getUsers(), [](Operation* op) { return !isa<TransferOp>(op); }));
  }
  return mask;
}

// Returns the effective operands of an operation. For an op with regions (that
// is not isolated from above) this includes both its operands but also any
// other SSA values from an outer scope referenced inside any inner regions of
// the op. We use a set vector to preserve the original operand order, which
// eases testing.
SetVector<Value> GetEffectiveOperands(Operation* op) {
  SetVector<Value> operands;
  operands.insert(op->operand_begin(), op->operand_end());
  for (Region& region : op->getRegions()) {
    getUsedValuesDefinedAbove(region, operands);
  }
  return operands;
}

// Collects all operations that have results that have a dataflow to one of the
// values in the `values` list.
// NB: `values` are purposefully passed by-value as this function mutates them.
DenseSet<Operation*> CollectOpsFlowingTo(std::vector<Value> values) {
  DenseSet<Operation*> ops;
  while (!values.empty()) {
    Value value = values.back();
    values.pop_back();

    Operation* op = value.getDefiningOp();
    if (!op || ops.contains(op)) continue;

    ops.insert(op);
    for (Value operand : GetEffectiveOperands(op)) {
      values.push_back(operand);
    }
  }
  return ops;
}

// Collect the values that represent root values of computations that (i)
// flow to the values in the `values` and are either block arguments or produced
// by the `boundary_ops`.
SetVector<Value> CollectRootsForValues(
    std::vector<Value> values, const DenseSet<Operation*>& boundary_ops) {
  SetVector<Value> roots;
  while (!values.empty()) {
    Value value = values.back();
    values.pop_back();

    Operation* op = value.getDefiningOp();
    if (!op || boundary_ops.contains(op)) {
      roots.insert(value);
    } else {
      for (Value operand : GetEffectiveOperands(op)) {
        values.push_back(operand);
      }
    }
  }
  return roots;
}

// Splits a list of values by a mask to the elements for which the mask is true,
// and the elements for which the mask is false.
std::pair<std::vector<Value>, std::vector<Value>> SplitValuesByMask(
    ValueRange values, const BitVector& mask) {
  std::vector<Value> true_results;
  true_results.reserve(mask.count());
  std::vector<Value> false_results;
  false_results.reserve(values.size() - mask.count());
  for (auto [index, value] : llvm::enumerate(values)) {
    if (mask[index]) {
      true_results.push_back(value);
    } else {
      false_results.push_back(value);
    }
  }
  return std::make_pair(true_results, false_results);
}

// Returns `values` filtered to where the mask is set.
std::vector<Value> FilterByMask(ValueRange values, const BitVector& mask) {
  std::vector<Value> results;
  results.reserve(mask.count());
  for (auto [index, value] : llvm::enumerate(values)) {
    if (mask[index]) {
      results.push_back(value);
    }
  }
  return results;
}

// Maps a set of operations through an IRMapping.
DenseSet<Operation*> MapOpSet(const DenseSet<Operation*> ops,
                              const IRMapping& mapping) {
  DenseSet<Operation*> mapped_ops;
  for (Operation* op : ops) {
    mapped_ops.insert(mapping.lookup(op));
  }
  return mapped_ops;
}

// Traverses the region in reverse order, applies the mapping for each op, and
// erases any (mapped) op that is not a terminator and for which `should_erase`
// returns true. As such ops are not erased from the region, but rather the
// image of the region through the mapping.
void EraseOpsThroughIRMapping(Region& region, IRMapping& mapping,
                              IRRewriter& rewriter,
                              std::function<bool(Operation*)> should_erase) {
  for (Operation& op : llvm::reverse(region.front().getOperations())) {
    Operation* mapped_op = mapping.lookup(&op);
    if (!mapped_op->hasTrait<OpTrait::IsTerminator>() &&
        should_erase(mapped_op)) {
      rewriter.eraseOp(mapped_op);
    }
  }
}

// Pulls computations out of a fragment based on the `pullable_mask` that
// corresponds to the results of this fragment.
//
// For every result, if the corresponding bit of the mask is set, then we will
// pull (i.e. extract) as much computation as we can out of this fragment into
// another fragment -- subject to making sure that we keep any result with the
// mask unset as result of the original fragment. In the process we may need to
// pass some "residual" values returned from the first fragment into the second.
//
// In pseudocode (and assuming that the pullables and the non-pullables are
// cleanly separated, just for the sake of simplicity), assume original:
//
//  %pullables, %non_pullables = fragment (...) {   // original fragment
//     ...
//     return %ps, %nps
//  }
// Becomes:
//  %non_pullables, %residuals = fragment (...) {   // keep non pullables
//     ...
//     return %nps, %rs
//  }
//
//  %pullables = fragment (..., %residuals) {       // keep pullables
//     ...
//     return %ps
//  }
void PullResultsOutOf(IRRewriter& rewriter, FragmentOp fragment,
                      ArrayRef<Value> pullable_results,
                      const DenseSet<Operation*>& non_pullable_ops,
                      const BitVector& pullable_mask) {
  MLIRContext* context = rewriter.getContext();

  auto make_mesh_type = [&](Value value) -> Type {
    // We create a fully replicated mesh type as we assume currently that the
    // pass will run prior to SPMD propagation (otherwise we'd have to create
    // a distributed type that matches the sharding specification of the value.)
    return MeshTensorType::getFullyReplicated(
        context, fragment.getMeshName(),
        GetMeshOrFail(fragment, fragment.getMeshName()),
        cast<RankedTensorType>(value.getType()));
  };

  Region& region = fragment.getRegion();

  // If the non pullable ops and the terminator are all the ops of the fragment
  // then there is simply nothing to pull.
  if (non_pullable_ops.size() + 1 ==
      fragment->getBlock()->getOperations().size()) {
    return;
  }
  // We calculate the residuals as the root values flowing to the pullables.
  SetVector<Value> residuals = CollectRootsForValues(
      /*values=*/std::move(pullable_results),
      /*boundary_ops=*/non_pullable_ops);

  // 1. Create the fragment with only transferred and residual results.
  // ------------------------------------------------------------------

  // 1.1 Create the fragment return types.
  std::vector<Type> transfer_fragment_types;
  transfer_fragment_types.reserve(pullable_mask.size() - pullable_mask.count() +
                                  residuals.size());
  for (auto result : fragment.getResults()) {
    if (!pullable_mask[result.getResultNumber()]) {
      transfer_fragment_types.push_back(result.getType());
    }
  }
  std::transform(residuals.begin(), residuals.end(),
                 std::back_inserter(transfer_fragment_types), make_mesh_type);

  // 1.2 Create the fragment.
  rewriter.setInsertionPointAfter(fragment);
  FragmentOp transfer_fragment =
      FragmentOp::create(rewriter, fragment.getLoc(), transfer_fragment_types,
                         fragment->getOperands(), fragment.getOriginAttr(),
                         fragment.getMeshNameAttr(), fragment.getStageIdAttr());
  // TODO(jupvfranco): streamline fragment attribute copying in our codebase.
  CopyAttributes(fragment, transfer_fragment);
  transfer_fragment->setAttr(kSplitKeepTransferredAttrName,
                             UnitAttr::get(context));

  // 1.3 Clone the region, fix up the terminator, and clean up ops.
  {
    IRMapping mapping;
    Region& new_region = transfer_fragment.getRegion();
    region.cloneInto(&new_region, mapping);
    std::vector<Value> ret_values;
    auto* terminator = new_region.front().getTerminator();
    for (OpOperand& ret_value : terminator->getOpOperands()) {
      if (!pullable_mask[ret_value.getOperandNumber()]) {
        ret_values.push_back(ret_value.get());
      }
    }
    for (auto res_value : residuals) {
      ret_values.push_back(mapping.lookup(res_value));
    }
    terminator->setOperands(ret_values);

    // Erase the ops that are not in non_pullable_ops through the mapping.
    DenseSet<Operation*> keep_ops = MapOpSet(non_pullable_ops, mapping);
    EraseOpsThroughIRMapping(region, mapping, rewriter, [&](Operation* op) {
      return !keep_ops.contains(op);
    });
  }

  // 2. Create the fragment with residual arguments.
  // -----------------------------------------------

  // 2.1 Create the return types.
  std::vector<Type> pulled_fragment_types;
  pulled_fragment_types.reserve(pullable_mask.count());
  for (auto result : fragment.getResults()) {
    if (pullable_mask[result.getResultNumber()]) {
      pulled_fragment_types.push_back(result.getType());
    }
  }
  // 2.2 Create the operands.
  std::vector<Value> pulled_fragment_operands(fragment->operand_begin(),
                                              fragment->operand_end());
  for (auto residual_value :
       transfer_fragment.getResults().take_back(residuals.size())) {
    pulled_fragment_operands.push_back(residual_value);
  }

  // 2.3 Create the actual fragment.
  rewriter.setInsertionPointAfter(transfer_fragment);
  FragmentOp pulled_fragment =
      FragmentOp::create(rewriter, fragment.getLoc(), pulled_fragment_types,
                         pulled_fragment_operands, fragment.getOriginAttr(),
                         fragment.getMeshNameAttr(), fragment.getStageIdAttr());
  // TODO(jupvfranco): streamline fragment attribute copying in our codebase.
  CopyAttributes(fragment, pulled_fragment);
  pulled_fragment->setAttr(kSplitDropTransferredAttrName,
                           UnitAttr::get(context));

  // 2.4 Clone the region, fixup terminator and block arguments.
  {
    IRMapping mapping;
    Region& new_region = pulled_fragment.getRegion();
    region.cloneInto(&new_region, mapping);
    // Add residual block arguments to the region and replace uses of roots.
    for (auto res_value : residuals) {
      rewriter.replaceAllUsesWith(
          mapping.lookup(res_value),
          new_region.addArgument(res_value.getType(), res_value.getLoc()));
    }
    std::vector<Value> ret_values;
    Operation* terminator = new_region.front().getTerminator();
    for (OpOperand& ret_value : terminator->getOpOperands()) {
      if (pullable_mask[ret_value.getOperandNumber()]) {
        ret_values.push_back(ret_value.get());
      }
    }
    terminator->setOperands(ret_values);

    // Erase any ops contained in the non_pullable_ops through the mapping.
    DenseSet<Operation*> keep_ops = MapOpSet(non_pullable_ops, mapping);
    EraseOpsThroughIRMapping(region, mapping, rewriter, [&](Operation* op) {
      return keep_ops.contains(op);
    });
  }

  // Replace each original fragment result with a result from either the
  // transfer or the pulled fragment.
  auto transfer_fragment_it = transfer_fragment->result_begin();
  auto pulled_fragment_it = pulled_fragment->result_begin();
  for (OpResult result : fragment.getResults()) {
    if (!pullable_mask[result.getResultNumber()]) {
      rewriter.replaceAllUsesWith(result, *transfer_fragment_it);
      transfer_fragment_it++;
    } else {
      rewriter.replaceAllUsesWith(result, *pulled_fragment_it);
      pulled_fragment_it++;
    }
  }
  rewriter.eraseOp(fragment);
}

// Splits the fragment into two fragments, with as much computation as possible
// pulled out of the original fragment into the second.
void PullResultsMaximally(IRRewriter& rewriter, FragmentOp fragment,
                          BitVector result_pullable_mask) {
  const auto [pullable, non_pullable] = SplitValuesByMask(
      fragment.getBody()->getTerminator()->getOperands(), result_pullable_mask);

  PullResultsOutOf(rewriter, fragment, std::move(pullable),
                   CollectOpsFlowingTo(std::move(non_pullable)),
                   result_pullable_mask);
}

// Finds the results that are returned from the fragment, and marks them in the
// `result_mask`. Skips values that are already seen.
void FindAndMarkResults(Value val, DenseSet<Operation*>& seen,
                        BitVector& result_mask) {
  for (OpOperand& use : val.getUses()) {
    if (auto return_op = dyn_cast<ReturnOp>(use.getOwner())) {
      result_mask.set(use.getOperandNumber());
      continue;
    }

    if (seen.contains(use.getOwner())) {
      continue;
    }
    // Only add to `seen` if there are multiple operands,
    // otherwise this will never be visited again.
    if (use.getOwner()->getNumOperands() > 1) {
      seen.insert(use.getOwner());
    }

    for (auto result : use.getOwner()->getResults()) {
      FindAndMarkResults(result, seen, result_mask);
    }
  }
}

BitVector FindResultsFlowingFrom(ValueRange values, int num_results) {
  BitVector result_mask(num_results);
  DenseSet<Operation*> seen;
  // An early guess to the number of ops seen.
  seen.reserve(num_results);
  for (Value val : values) {
    FindAndMarkResults(val, seen, result_mask);
  }

  return result_mask;
}

DenseSet<Operation*> TransitiveUsersOf(std::vector<Value> values) {
  DenseSet<Operation*> transitive_users;
  while (!values.empty()) {
    Value value = values.back();
    values.pop_back();

    for (Operation* op : value.getUsers()) {
      if (transitive_users.contains(op)) {
        continue;
      }
      transitive_users.insert(op);
      for (Value res : op->getResults()) {
        values.push_back(res);
      }
    }
  }
  return transitive_users;
}

DenseSet<Operation*> OpsNotRelyingOn(std::vector<Value> values,
                                     FragmentOp fragment) {
  DenseSet<Operation*> ops_to_avoid = TransitiveUsersOf(std::move(values));
  DenseSet<Operation*> ops;

  for (Operation& op : fragment.getBody()->getOperations()) {
    if (ops_to_avoid.contains(&op)) {
      continue;
    }
    ops.insert(&op);
  }

  return ops;
}

// Splits the fragment into two fragments, with as much computation as possible
// pulled out of the original fragment into the first.
void PullOperandsOutMaximally(IRRewriter& rewriter, FragmentOp fragment,
                              BitVector arg_pullable_mask) {
  std::vector<Value> args_to_pull =
      FilterByMask(fragment.getBody()->getArguments(), arg_pullable_mask);
  BitVector result_pullable_mask =
      FindResultsFlowingFrom(args_to_pull, fragment.getNumResults());

  std::vector<Value> pullable = FilterByMask(
      fragment.getBody()->getTerminator()->getOperands(), result_pullable_mask);

  DenseSet<Operation*> non_pullable_ops =
      OpsNotRelyingOn(std::move(args_to_pull), fragment);

  if (non_pullable_ops.empty() && result_pullable_mask.all()) {
    // Everything will be pulled out, so we don't need to split.
    return;
  }

  // We achieve the effect of pulling out the operands by pulling out the
  // results that flow from the args, and maximally setting out the non-pullable
  // ops.
  PullResultsOutOf(rewriter, fragment, pullable, non_pullable_ops,
                   result_pullable_mask);
}

class SplitBwdFragmentsPass
    : public impl::SplitBwdFragmentsPassBase<SplitBwdFragmentsPass> {
  using SplitBwdFragmentsPassBase::SplitBwdFragmentsPassBase;

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    getOperation().walk([&](FragmentOp fragment) {
      if (IsBackwardsCandidateFragment(fragment)) {
        // The pullable values are those that are non-transferred.
        BitVector pullable_mask = FindNonTransferredResults(fragment);
        if (!pullable_mask.all() && pullable_mask.any()) {
          PullResultsMaximally(rewriter, fragment, pullable_mask);
        }
      }
    });
  }
};

// Finds the operands that are transfer results.
BitVector FindTransferredArgs(FragmentOp fragment) {
  BitVector mask(fragment->getNumOperands());
  for (OpOperand& operand : fragment->getOpOperands()) {
    if (isa_and_present<TransferOp>(operand.get().getDefiningOp())) {
      mask.set(operand.getOperandNumber());
    }
  }

  return mask;
}

bool IsSplitTransferIndependentCandidate(FragmentOp fragment) {
  return !IsSplitDropTransferred(fragment) && !IsSplitKeepTransferred(fragment);
}

class SplitAndPrioritizeTransferIndependentComputationsPass
    : public impl::SplitAndPrioritizeTransferIndependentComputationsPassBase<
          SplitAndPrioritizeTransferIndependentComputationsPass> {
  using SplitAndPrioritizeTransferIndependentComputationsPassBase::
      SplitAndPrioritizeTransferIndependentComputationsPassBase;

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    Block& func_body = getOperation().front();
    // Preorder walk to avoid walking into fragment bodies, since we don't need
    // to.
    func_body.walk<WalkOrder::PreOrder>([&](FragmentOp fragment) {
      if (IsSplitTransferIndependentCandidate(fragment)) {
        // The pullable values are those that are users of transfer
        // operands.
        BitVector operand_mask = FindTransferredArgs(fragment);
        if (!operand_mask.all() && operand_mask.any()) {
          PullOperandsOutMaximally(rewriter, fragment, operand_mask);
        }
      }
      return WalkResult::skip();
    });
  }
};

}  // namespace
}  // namespace mlir::mpmd
