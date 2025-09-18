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

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/optimize/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/optimize/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_REMATFRAGMENTPASS
#include "shardy/dialect/mpmd/transforms/optimize/passes.h.inc"

namespace {

// Rematerializes one forward_fragment matched with a backward_fragment.
// When `merge_remat_fragments=true`, we merge the rematerialized fragment with
// its backward consumer. We mark the resulting remat fragment (either merged or
// unmerged), as this is often useful for debugging.
//
// Requires: CanRemat(forward_fragment, backward_fragment).
void RematFragment(IRRewriter& rewriter, FragmentOp forward_fragment,
                   FragmentOp backward_fragment, bool merge_remat_fragments) {
  SDY_CHECK(CanRemat(forward_fragment, backward_fragment));
  rewriter.setInsertionPoint(backward_fragment);
  // Clone the forward fragment.
  FragmentOp remat_fragment =
      cast<FragmentOp>(rewriter.clone(*forward_fragment));
  rewriter.replaceUsesWithIf(
      forward_fragment.getResults(), remat_fragment.getResults(),
      [&](OpOperand& use) { return use.getOwner() == backward_fragment; });
  if (merge_remat_fragments) {
    std::optional<uint32_t> backward_fragment_call_counter =
        TryToFindCallCounter(backward_fragment);
    FragmentOp merged_fragment = MergeRegionOps(
        remat_fragment, backward_fragment, rewriter,
        /*num_static_args=*/0, /*replace_producer_use_in_consumer_block=*/
        [](OpOperand&, Value) {
          SDY_CHECK(false) << "Fragment ops shouldn't have free variables";
        },
        GetFragmentOriginUnion(remat_fragment, backward_fragment, rewriter),
        backward_fragment.getMeshNameAttr(),
        backward_fragment.getStageIdAttr());
    // Set the call counter of the merged fragment to the call counter of the
    // backward fragment.
    if (backward_fragment_call_counter.has_value()) {
      merged_fragment->setAttr(
          kCallCounterAttrName,
          rewriter.getUI32IntegerAttr(backward_fragment_call_counter.value()));
    }

    MarkAsRemat(merged_fragment, rewriter);
  } else {
    MarkAsRemat(remat_fragment, rewriter);
  }
}

// Iterates over a function, identify forward and backward fragment pairs that
// need rematerialization and rematerializes one by one. If there are multiple
// backward fragments matching a forward fragment, remat all of them.
void RematFragments(IRRewriter& rewriter, func::FuncOp func_op,
                    bool merge_remat_fragments) {
  SmallVector<FragmentOp> all_forward_fragments;
  for (Operation& op : func_op.getOps()) {
    if (auto fragment = dyn_cast<FragmentOp>(&op);
        fragment && IsForwardFragment(fragment)) {
      all_forward_fragments.push_back(fragment);
    }
  }

  for (FragmentOp forward_fragment : all_forward_fragments) {
    // Get the users of forward_fragment that can be rematerialized and sort
    // them by their program order. The sorting is needed because `getUsers()`
    // does not guarantee returning the users in program order and when a user
    // that is later in the program appear first and gets rematted first (by
    // `RematFragment`), it prevents the earlier users to match and remat
    // correctly.
    DenseSet<Operation*> users;
    for (Operation* user : forward_fragment->getUsers()) {
      if (auto backward_fragment = dyn_cast<FragmentOp>(user);
          backward_fragment && CanRemat(forward_fragment, backward_fragment)) {
        users.insert(user);
      }
    }
    if (users.size() > 1 && SDY_VLOG_IS_ON(1)) {
      std::string fragment_metadata;
      llvm::raw_string_ostream fragment_metadata_stream(fragment_metadata);
      forward_fragment.printFragmentMetadata(fragment_metadata_stream);
      SDY_LOG(INFO)
          << "A forward fragment matched multiple backward fragments. "
             "metadata= "
          << fragment_metadata;
    }

    // In the case where multiple backward fragments match the same forward
    // fragment, we have a few options:
    // 1. Only add a remat fragment in front of the first backward fragment
    // and replace uses for the forward fragment in the other backward
    // fragments or
    // 2. Add a remat fragment in front of all backward fragments.
    // Theoretically we could also do anything in between, e.g., add a remat
    // fragment in front of m out of the n matched backward fragments (where 1
    // <= m <= n). All of these options are valid strategies. The tradeoff is
    // that the more remat fragments we add, the more we trade runtime for
    // memory. We are choosing option 2 because it's easier to implement. We
    // may support both options later to have more control over remat.
    for (Operation* user : users) {
      auto backward_fragment = dyn_cast<FragmentOp>(user);
      RematFragment(rewriter, forward_fragment, backward_fragment,
                    merge_remat_fragments);
    }
  }
}

class RematFragmentPass
    : public impl::RematFragmentPassBase<RematFragmentPass> {
  using RematFragmentPassBase::RematFragmentPassBase;

 private:
  void runOnFunc(func::FuncOp func_op) override {
    MLIRContext* context = func_op->getContext();
    IRRewriter rewriter(context);
    RematFragments(rewriter, func_op, mergeRematFragments);
  }
};

}  // namespace
}  // namespace mlir::mpmd
