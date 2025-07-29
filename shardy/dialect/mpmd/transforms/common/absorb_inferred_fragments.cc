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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/common/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_ABSORBINFERREDFRAGMENTSPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

// Checks if a fragment is a root fragment, i.e., if it is:
// - a user fragment, or
// - not used by any other fragment (e.g. a fragment used by the return op or a
// transfer only), or
// - not a user of a value produced by any other fragment (e.g., user of block
// arguments or transfers).
bool IsRoot(FragmentOp fragment) {
  return fragment.isUserFragment() ||
         llvm::none_of(
             fragment->getUses(),
             [](OpOperand& use) { return isa<FragmentOp>(use.getOwner()); }) ||
         llvm::none_of(fragment->getOperands(), [](Value operand) {
           return operand.getDefiningOp<FragmentOp>();
         });
}

// Returns the closest consumer of `op` (in program order), or nullptr if there
// is no consumer.
Operation* ClosestConsumer(Operation* op) {
  if (op->getUsers().empty()) {
    return nullptr;
  }
  // TODO: b/364281760 - This isBeforeInBlock seems to be a bottleneck in large
  // programs of many inferred fragments. This needs improvement, e.g., maybe
  // it's enough to check for dependencies between the ops.
  return *llvm::min_element(op->getUsers(), [](Operation* op1, Operation* op2) {
    return op1->isBeforeInBlock(op2);
  });
}

// Returns all the ops that produce values used by `op`.
SmallVector<Operation*> GetProducers(Operation* op) {
  SetVector<Operation*> producers;
  for (Value operand : op->getOperands()) {
    if (Operation* producer = operand.getDefiningOp()) {
      producers.insert(producer);
    }
  }
  return producers.takeVector();
}

// Returns the closest producer of `op` (in program order), or nullptr if there
// is no producer.
Operation* ClosestProducer(FragmentOp fragment) {
  SmallVector<Operation*> producers = GetProducers(fragment);
  if (producers.empty()) {
    return nullptr;
  }
  // TODO: b/364281760 - This isBeforeInBlock seems to be a bottleneck in large
  // programs of many inferred fragments. This needs improvement, e.g., maybe
  // it's enough to check for dependencies between the ops.
  return *llvm::min_element(producers, [](Operation* op1, Operation* op2) {
    // Being before in block means being further way
    // from the fragment.
    return op2->isBeforeInBlock(op1);
  });
}

// Returns all the transfers that are consumers of `fragment`.
SmallVector<TransferOp> GetTransferConsumers(FragmentOp fragment) {
  SetVector<TransferOp> transfers;
  for (Operation* user : fragment->getUsers()) {
    if (auto transfer = dyn_cast<TransferOp>(user)) {
      transfers.insert(transfer);
    }
  }
  return transfers.takeVector();
}

// Moves any transfer in `transfers` right before its first consumer, or removes
// it if it has no consumer.
void MoveTransfersToConsumerSites(ArrayRef<TransferOp> transfers,
                                  RewriterBase& rewriter) {
  for (TransferOp transfer : transfers) {
    if (Operation* closest_consumer = ClosestConsumer(transfer)) {
      rewriter.moveOpBefore(transfer, closest_consumer);
    } else {
      // If the transfer doesn't have a consumer, we simply remove it.
      rewriter.eraseOp(transfer);
    }
  }
}

// Returns all the inferred producer fragments that can be merged into the root
// fragment `root`. This is a set with any inferred fragment that is a producer
// of `root` and has root as its closest consumer. This function may delay
// transfers, by moving them next to their consumers, in order to create more
// merging opportunities.
SmallVector<FragmentOp> DelayTransfersAndGetMergeableInferredProducers(
    FragmentOp root, RewriterBase& rewriter) {
  SmallVector<Operation*> producers = GetProducers(root);
  SetVector<FragmentOp> mergeable_producers;
  for (Operation* producer : producers) {
    if (auto producer_fragment = dyn_cast<FragmentOp>(producer)) {
      if (producer_fragment.isUserFragment()) {
        continue;
      }

      // Try to move any transfers that are consumers of `producer_fragment` out
      // of the way.
      SmallVector<TransferOp> transfers =
          GetTransferConsumers(producer_fragment);
      MoveTransfersToConsumerSites(transfers, rewriter);

      if (ClosestConsumer(producer_fragment) != root) {
        continue;
      }
      mergeable_producers.insert(producer_fragment);
    }
  }
  return mergeable_producers.takeVector();
}

// Given a root fragment `R`, finds the closest inferred producer `P` of `R`
// that can be merged into `R` and merges it.
//
// A fragment P can be merged into `R` if:
// - it is an inferred fragment,
// - it is the closest producer of `R` in program order, and
// - it has `R` as its closest consumer.
//
// Transfers may be delayed in order to create more merging opportunities.
//
// For example, say we had the program:
//
//         /---> C
//       P -------> R
//     P' --------/
//
// where reading left-to-right we get operations executed in order: P', P, C, R.
//
// P is the closest producer of R in program order. However, if we merged P into
// R, we would break op dominance, since C is a consumer of P and it would
// appear before R (in program order). This merge cannot happen, unless C is a
// transfer that does not depend on R. In this case, we could move C out of the
// way, and merge P into R, obtaining:
//
//                 /--> C
//               PR
//   P' --------/
//
// Finally, P' too can be merged into PR, as well as C, if it's a fragment.
//
// Note that the location of R is preserved, as well as its attributes.
//
// TODO: b/364281760 - For now we don't actually check the dependency between C
// and R. We simply check if the transfer producer appears before R. Going
// forward, we should improve this.
class AbsorbClosestProducerPattern : public OpRewritePattern<FragmentOp> {
  using OpRewritePattern<FragmentOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(FragmentOp op,
                                PatternRewriter& rewriter) const override {
    if (!IsRoot(op)) {
      return failure();
    }

    SmallVector<FragmentOp> inferred_to_absorb =
        DelayTransfersAndGetMergeableInferredProducers(op, rewriter);
    if (inferred_to_absorb.empty()) {
      return failure();
    }

    FragmentOp inferred_producer = inferred_to_absorb.front();

    // We now merge `inferred_producer` into `op`, at the location of `op` and
    // preserving its attributes.
    DictionaryAttr discardable_attrs = op->getDiscardableAttrDictionary();
    auto new_fragment = MergeRegionOps(
        inferred_producer, op, rewriter,
        /*num_static_args=*/0, /*replace_producer_use_in_consumer_block=*/
        [](OpOperand&, Value) {
          SDY_CHECK(false) << "Fragment ops shouldn't have free variables";
        },
        op.getOriginAttr(), op.getMeshNameAttr(),
        /*stage_id=*/op.getStageIdAttr());
    new_fragment->setDiscardableAttrs(discardable_attrs);
    return success();
  }
};

// Returns all the transfers that are producers of `fragment`.
SetVector<TransferOp> GetTransferProducers(FragmentOp fragment) {
  SetVector<TransferOp> transfers;
  for (Operation* producer : GetProducers(fragment)) {
    if (auto transfer = dyn_cast<TransferOp>(producer)) {
      transfers.insert(transfer);
    }
  }
  return transfers;
}

// Move each transfer next to its producer, or to the top of the block if it is
// a block argument.
void MoveTransfersToProducerSites(ArrayRef<TransferOp> transfers,
                                  RewriterBase& rewriter) {
  for (TransferOp transfer : transfers) {
    if (auto arg = dyn_cast<BlockArgument>(transfer.getOperand())) {
      rewriter.moveOpBefore(transfer, arg.getOwner(), arg.getOwner()->begin());
    } else {
      rewriter.moveOpAfter(transfer, transfer.getOperand().getDefiningOp());
    }
  }
}

// Returns all the inferred consumer fragments that can be merged into the root
// fragment `root`. This is a set with any inferred fragment that is a consumer
// of `root` and has root as its closest producer.
SetVector<FragmentOp> EagerlyScheduleTransfersAndGetMergeableInferredConsumers(
    FragmentOp root, RewriterBase& rewriter) {
  SetVector<FragmentOp> consumers;
  for (Operation* user : root->getUsers()) {
    if (auto consumer_fragment = dyn_cast<FragmentOp>(user)) {
      if (consumer_fragment.isUserFragment()) {
        continue;
      }

      // Try to move any transfers that are producers of `consumer_fragment` out
      // of the way.
      SetVector<TransferOp> transfers = GetTransferProducers(consumer_fragment);
      MoveTransfersToProducerSites(transfers.getArrayRef(), rewriter);

      if (ClosestProducer(consumer_fragment) != root) {
        continue;
      }
      consumers.insert(consumer_fragment);
    }
  }
  return consumers;
}

// Given a root fragment `R`, finds the closest inferred consumer `C` of `R`
// that can be merged into `R` and merges it.
//
// A fragment C can be merged into `R` if:
// - it is an inferred fragment,
// - it is the closest consumer of `R` in program order, and
// - it has `R` as its closest producer.
//
// Transfers may be eagerly scheduled in order to create more merging
// opportunities.
//
// For example, say we had the program:
//     /------------> C'
//   R --------> C
//       P ---/
// where reading left-to-right we get operations executed in order: R, P, C, C'.
//
// C is the closest consumer of R in program order. However, if we merged C into
// R, we would break op dominance, since P is a producer of C and it would
// appear after R (in program order). This merge cannot happen, unless P is a
// transfer that does not depend on R. In this case, we could move it out of the
// way, and merge C into R, obtaining:
//        /------------> C'
//      RC
//   P -/
//
// Finally, C' too can be merged into R.
//
// Note that the location of R is preserved, as well as its attributes.
//
// TODO: b/364281760 - For now we don't actually check the dependency between X
// and C. We simply check if the transfer producer appears before C. Going
// forward, we should improve this.
class AbsorbClosestConsumerPattern : public OpRewritePattern<FragmentOp> {
  using OpRewritePattern<FragmentOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(FragmentOp op,
                                PatternRewriter& rewriter) const override {
    if (!IsRoot(op)) {
      return failure();
    }

    SetVector<FragmentOp> mergeable_consumers =
        EagerlyScheduleTransfersAndGetMergeableInferredConsumers(op, rewriter);
    if (mergeable_consumers.empty()) {
      // Nothing to absorb.
      return failure();
    }

    FragmentOp inferred_consumer = mergeable_consumers.front();
    // We now merge `inferred_consumer` into `op`, at the location of `op` and
    // preserving its attributes.
    DictionaryAttr discardable_attrs = op->getDiscardableAttrDictionary();
    Operation* new_fragment_dest = op->getNextNode();
    if (new_fragment_dest == inferred_consumer) {
      new_fragment_dest = new_fragment_dest->getNextNode();
    }
    FragmentOp new_fragment = MergeRegionOps(
        op, inferred_consumer, rewriter,
        /*num_static_args=*/0, /*replace_producer_use_in_consumer_block=*/
        [](OpOperand&, Value) {
          SDY_CHECK(false) << "Fragment ops shouldn't have free variables";
        },
        op.getOriginAttr(), op.getMeshNameAttr(),
        /*stage_id=*/op.getStageIdAttr());
    rewriter.moveOpBefore(new_fragment, new_fragment_dest);
    new_fragment->setDiscardableAttrs(discardable_attrs);
    return success();
  }
};

class AbsorbInferredFragmentsPass
    : public impl::AbsorbInferredFragmentsPassBase<
          AbsorbInferredFragmentsPass> {
  using AbsorbInferredFragmentsPassBase::AbsorbInferredFragmentsPassBase;

 private:
  void runOnFunc(func::FuncOp func) override {
    if (!IsMpmdFunction(func)) {
      return;
    }

    if (!absorbOnEntryPointFunction && IsEntryPointFunction(func)) {
      return;
    }

    RewritePatternSet merge_patterns(func.getContext());
    merge_patterns
        .add<AbsorbClosestProducerPattern, AbsorbClosestConsumerPattern>(
            func.getContext());
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled)
        .enableFolding(false);
    if (failed(
            applyPatternsGreedily(func, std::move(merge_patterns), config))) {
      return signalPassFailure();
    }

    if (IsEntryPointFunction(func)) {
      func.getBody().walk([](FragmentOp fragment) {
        if (!fragment.isUserFragment()) {
          SDY_LOG(WARNING)
              << "Non entry-point MPMD function includes inferred "
                 "fragments, which could cause performance issues.";
        }
      });
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
