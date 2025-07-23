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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/common/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_FRAGMENTDCEPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

bool EmptyFragment(FragmentOp fragment) {
  return fragment.getResults().empty() && fragment->getOperands().empty() &&
         llvm::all_of(fragment.getBody()->getOperations(),
                      [](const Operation& op) { return isa<ReturnOp>(&op); });
}

FragmentOp EliminateUnusedResults(FragmentOp fragment, RewriterBase& rewriter) {
  BitVector unused_results(fragment.getNumResults());
  for (OpResult result : fragment->getResults()) {
    if (result.use_empty()) {
      unused_results.set(result.getResultNumber());
    }
  }
  if (unused_results.none()) {
    return fragment;
  }

  // Note: even if unused_results.all(), we may want to keep the fragment, in
  // case it contains ops with side-effects.

  rewriter.setInsertionPoint(fragment);

  Operation* terminator = fragment->getRegion(0).front().getTerminator();
  terminator->eraseOperands(unused_results);
  auto new_fragment = rewriter.create<FragmentOp>(
      fragment.getLoc(),
      FilterRange<Type>(/*range=*/fragment.getResultTypes(),
                        /*erase=*/unused_results),
      fragment.getOperands(), fragment.getOriginAttr(),
      fragment.getMeshNameAttr(), fragment.getStageIdAttr());
  // Copy all attributes except `origin` and `mesh_name`, which were copied
  // during the creation of the new fragment.
  CopyAttributes(fragment, new_fragment,
                 /*elided_attrs_set=*/{"origin", "mesh_name"});
  new_fragment.getRegion().takeBody(fragment.getRegion());

  BitVector& used_results = unused_results.flip();
  for (auto [old_result_index, new_result] :
       llvm::zip(used_results.set_bits(), new_fragment.getResults())) {
    rewriter.replaceAllUsesWith(fragment->getResult(old_result_index),
                                new_result);
  }

  rewriter.eraseOp(fragment);
  return new_fragment;
}

// Simplifies the region of the fragment. This will apply several
// simplifications to the region, such as removing dead code, which can cause
// block arguments of the fragment to become unused, thus triggering the
// `EliminateUnusedArgumentsPattern`. This is why we simplify the fragment with
// a pattern instead of waiting for the pass to apply simplification at the end
// of all rewrites.
void SimplifyFragmentRegion(FragmentOp fragment, RewriterBase& rewriter) {
  (void)simplifyRegions(rewriter, fragment.getRegion());
}

void EliminateUnusedArguments(FragmentOp fragment, RewriterBase& rewriter) {
  Block& block = fragment.getRegion().front();
  BitVector unused_arguments(fragment.getNumOperands());
  for (BlockArgument arg : block.getArguments()) {
    if (arg.use_empty()) {
      unused_arguments.set(arg.getArgNumber());
    }
  }
  if (unused_arguments.none()) {
    return;
  }

  block.eraseArguments(unused_arguments);
  fragment->setOperands(FilterRange<Value>(/*range=*/fragment.getOperands(),
                                           /*erase=*/unused_arguments));
}

class FragmentDcePass : public impl::FragmentDcePassBase<FragmentDcePass> {
  using FragmentDcePassBase::FragmentDcePassBase;

 private:
  void runOnFunc(func::FuncOp func_op) override {
    IRRewriter rewriter(func_op.getContext());
    // Make sure we process users before producers.
    func_op.walk<WalkOrder::PreOrder, ReverseIterator>(
        [&rewriter](Operation* op) {
          if (auto fragment = dyn_cast<FragmentOp>(op)) {
            FragmentOp new_fragment =
                EliminateUnusedResults(fragment, rewriter);
            SimplifyFragmentRegion(new_fragment, rewriter);
            EliminateUnusedArguments(new_fragment, rewriter);
            if (EmptyFragment(new_fragment)) {
              rewriter.eraseOp(new_fragment);
            }
            // Fragments cannot nest other fragments, so no need to visit the
            // fragment's region.
            return WalkResult::skip();
          }
          // Region simplification above cleans up ops inside fragments. Erasure
          // here cleans up ops outside.
          if (isa<TransferOp>(op)) {
            if (op->use_empty() && isPure(op) &&
                !op->hasTrait<OpTrait::IsTerminator>()) {
              rewriter.eraseOp(op);
              return WalkResult::skip();
            }
          }
          return WalkResult::advance();
        });
  }
};

}  // namespace
}  // namespace mlir::mpmd
