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

#include "shardy/dialect/mpmd/transforms/common/simplify_region_op_base.h"

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"

namespace mlir::mpmd {

namespace {

// Returns a BitVector indicating for every source value whether it is a
// duplicate of a previous source and should therefore be erased, and replaces
// the corresponding target of each duplicate source with the corresponding
// target value of the first occurrence of that source.
BitVector DedupSrcTgtValues(ValueRange src_values, ValueRange tgt_values,
                            PatternRewriter& rewriter) {
  llvm::DenseMap<std::pair<Value, Type>, Value> mapping;
  BitVector erase_values(src_values.size());
  for (auto src_tgt_value_it :
       llvm::enumerate(llvm::zip(src_values, tgt_values))) {
    auto [src_value, tgt_value] = src_tgt_value_it.value();
    auto key = std::make_pair(src_value, tgt_value.getType());
    if (Value mapped_tgt = mapping.lookup(key)) {
      rewriter.replaceAllUsesWith(tgt_value, mapped_tgt);
      erase_values.set(src_tgt_value_it.index());
    } else {
      mapping[key] = tgt_value;
    }
  }

  return erase_values;
}

// Remove all results whose corresponding return operands are block
// arguments of the fragment, if the result type matches that of the operand
// corresponding to the block argument, and replace those results with the
// corresponding operand.
//
// In addition, remove operands whose corresponding block arguments have no more
// uses.
void RemoveNoopResults(Operation* op, BitVector& erase_operands,
                       BitVector& erase_results, PatternRewriter& rewriter) {
  SDY_CHECK_EQ(op->getNumRegions(), 1);
  Block& block = op->getRegion(0).front();
  Operation* return_op = block.getTerminator();
  for (BlockArgument arg : block.getArguments()) {
    int arg_num = arg.getArgNumber();
    Value operand = op->getOperand(arg_num);
    bool should_erase_operand = true;
    for (OpOperand& use : arg.getUses()) {
      int use_operand_num = use.getOperandNumber();
      if (use.getOwner() == return_op &&
          operand.getType() == op->getResult(use_operand_num).getType()) {
        rewriter.replaceAllUsesWith(op->getResult(use_operand_num), operand);
        erase_results.set(use_operand_num);
      } else {
        should_erase_operand = false;
      }
    }
    if (should_erase_operand) {
      erase_operands.set(arg_num);
    }
  }
}

// Checks if the result of an operation can be removed from its producer's
// results, i.e., if it is unused and if it's removal doesn't cause any op
// with side effects to be removed.
//
// If `result` is the only result of the region, then two things can happen:
// if the region is recursively pure, then returns `true` so that the caller
// will remove the whole region, which is safe (as long as `result` is not
// used); otherwise, if the region is not pure, then removing the result would
// create a region without results, which isn't well-formed, and therefore we do
// not allow for the value to be removed.

// In particular, it checks whether the value corresponds to a block
// argument of the region or whether the nested op producing the result is
// pure, in case the region results another value produced by an op with
// side-effects.
//
// Example 1:
//   %r:4 = region_based_op(...) (%arg0, %arg1, %arg2) {
//      %0 = add %arg1, arg1
//      %1 = all_gather arg2
//      %2 = add %1, %1
//      return %arg0, %0, %1, %2
//   }
//
// When unused: %r#0 can be removed from the fragment's results as its
// actual producer isn't nested in the region; %r#1 can also also be removed
// because its actual producer (%0) is pure; %r#2 _cannot_ be removed because
// it's actual producer (%1) is not pure; and %r#3 can be removed because it's
// actual producer (%2) is pure, even though it derives from an all-gather.
//
// The resulting region is:
//   %r = region_based_op(...) (%arg0, %arg1, %arg2) {
//      %1 = all_gather arg2
//      return %1
//   }
void RemoveUnusedResultsFromRegionOp(Operation* region_op,
                                     BitVector& erase_results,
                                     PatternRewriter& rewriter) {
  // Mark any unused result to be erased.
  for (OpResult result : region_op->getResults()) {
    if (!result.use_empty()) {
      // Cannot be removed as it is used.
      continue;
    }

    if (region_op->getNumResults() == 1) {
      if (isPure(region_op)) {
        // Can be removed as long as this means that the whole region will be
        // removed. Otherwise this could create a fragment without results,
        // which isn't valid ATM.
        // TODO(b/310958300): Revisit this once we can have 0-result fragments.
        erase_results.set(result.getResultNumber());
      }
      // otherwise, keep the value alive so that the fragment's isn't removed.
      continue;
    }

    Operation* terminator =
        result.getOwner()->getRegion(0).front().getTerminator();
    Operation* defining_op =
        terminator->getOperand(result.getResultNumber()).getDefiningOp();
    // The value can be removed if it is a block argument or if its defining op
    // is pure.
    if (!defining_op || isPure(defining_op)) {
      erase_results.set(result.getResultNumber());
    }
  }
}

}  // namespace

LogicalResult SimplifyRegionOp(Operation* op, PatternRewriter& rewriter,
                               SimplifiedRegionOpCreateFn create_op) {
  SDY_CHECK_EQ(op->getNumRegions(), 1);
  Region& region = op->getRegion(0);
  Block& block = region.front();
  Operation* return_op = block.getTerminator();
  bool has_operands = op->getNumOperands() > 0;
  if (has_operands) {
    SDY_CHECK_EQ(block.getNumArguments(), op->getNumOperands());
  }
  SDY_CHECK_EQ(return_op->getNumOperands(), op->getNumResults());

  BitVector erase_results =
      DedupSrcTgtValues(/*src_values=*/return_op->getOperands(),
                        /*tgt_values=*/op->getResults(), rewriter);

  BitVector erase_operands;
  if (has_operands) {
    erase_operands =
        DedupSrcTgtValues(/*src_values=*/op->getOperands(),
                          /*tgt_values=*/block.getArguments(), rewriter);
    RemoveNoopResults(op, erase_operands, erase_results, rewriter);
  }

  RemoveUnusedResultsFromRegionOp(op, erase_results, rewriter);

  // If all results must be erased, we erase the op.
  if (erase_results.all()) {
    rewriter.eraseOp(op);
    return success();
  }

  if (erase_operands.none() && erase_results.none()) {
    // No simplification needed.
    return failure();
  }

  // NOTE: we need to erase return operands before we erase block arguments
  // because the former might be a use of the latter.
  return_op->eraseOperands(erase_results);
  if (has_operands) {
    block.eraseArguments(erase_operands);
  }

  SmallVector<Value> new_operands =
      FilterRange<Value>(op->getOperands(), erase_operands);
  SmallVector<Type> new_result_types =
      FilterRange<Type>(op->getResultTypes(), erase_results);
  Operation* new_op = create_op(new_result_types, new_operands, erase_results);
  SDY_CHECK_EQ(new_op->getNumRegions(), 1);
  Region& new_region = new_op->getRegion(0);
  new_region.takeBody(region);

  // Simplify the region to make sure we remove any dead code.
  (void)simplifyRegions(rewriter, new_region);

  // replace all results of `op` that weren't erased with the results of
  // `new_op` (erased results were already replaced)
  int new_result_num = 0;
  for (OpResult result : op->getResults()) {
    if (!erase_results.test(result.getResultNumber())) {
      rewriter.replaceAllUsesWith(result, new_op->getResult(new_result_num++));
    }
  }

  // Explicitly erase the op. This will cause the rewriter to add the operands
  // to the worklist, and trigger simplification of the operands' producer
  // fragments, thus reducing the number of iterations needed.
  rewriter.eraseOp(op);
  return success();
}

}  // namespace mlir::mpmd
