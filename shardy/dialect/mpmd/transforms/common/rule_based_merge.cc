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
#include <vector>

#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/optimize/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_RULEBASEDMERGEPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

using FragmentMergeRuleMap =
    DenseMap<FragmentInfo, const FragmentMergeRule*, FragmentInfoMapInfo>;

// TODO(petebu): Consider doing a forward walk through the function while
// merging matching fragments instead of using a GreedyPatternRewriter.
class RuleBasedMergingPattern : public OpRewritePattern<FragmentOp> {
  using OpRewritePattern<FragmentOp>::OpRewritePattern;

 public:
  RuleBasedMergingPattern(MLIRContext* context,
                          const FragmentMergeRules& merging_rules,
                          const FragmentMergeRuleMap& fragment_merge_rule_map)
      : OpRewritePattern<FragmentOp>(context),
        fragment_merge_rule_map_(fragment_merge_rule_map) {}

  LogicalResult matchAndRewrite(FragmentOp merge_into_fragment,
                                PatternRewriter& rewriter) const override {
    FragmentInfo fragment_info = GetFragmentInfo(merge_into_fragment);
    // Find the merge rule for the fragment, if there is one.
    auto it = fragment_merge_rule_map_.find(fragment_info);
    if (it == fragment_merge_rule_map_.end()) {
      return failure();
    }
    const FragmentMergeRule* rule = it->second;

    // Get all the merge candidates for the merge rule.
    std::vector<FragmentOp> merge_candidates =
        GetMergeCandidates(merge_into_fragment, rule);
    if (merge_candidates.empty()) {
      return failure();
    }
    merge_candidates.push_back(merge_into_fragment);

    // Sort the merge candidates topologically.
    SmallVector<Operation*> sorted_merge_candidates;
    sorted_merge_candidates.reserve(merge_candidates.size());
    for (auto& merge_candidate : merge_candidates) {
      sorted_merge_candidates.push_back(merge_candidate);
    }
    if (!computeTopologicalSorting(sorted_merge_candidates)) {
      SDY_LOG(ERROR) << "Cycle detected in sorted_merge_candidates";
      return failure();
    }

    // Merge the fragments.
    Operation* new_fragment_dest = sorted_merge_candidates[0]->getNextNode();
    FragmentOp new_fragment = dyn_cast<FragmentOp>(*sorted_merge_candidates[0]);
    for (int i = 1; i < sorted_merge_candidates.size(); ++i) {
      FragmentOp merge_candidate =
          dyn_cast<FragmentOp>(*sorted_merge_candidates[i]);
      if (new_fragment_dest == merge_candidate) {
        new_fragment_dest = new_fragment_dest->getNextNode();
      }
      new_fragment = MergeRegionOps(
          new_fragment, merge_candidate, rewriter,
          /*num_static_args=*/0, /*replace_producer_use_in_consumer_block=*/
          [](OpOperand&, Value) {
            SDY_CHECK(false) << "Fragment ops shouldn't have free variables";
          },
          GetFragmentOriginUnion(new_fragment, merge_candidate, rewriter),
          new_fragment.getMeshNameAttr(),
          /*stage_id=*/new_fragment.getStageIdAttr());
    }
    SetFragmentInfo(new_fragment, rule->target, rewriter);
    // TODO(petebu): Consider making the position of the new fragment a
    // parameter of the rule.
    SDY_CHECK(new_fragment_dest != nullptr);
    rewriter.moveOpBefore(new_fragment, new_fragment_dest);
    return success();
  }

 private:
  std::vector<FragmentOp> GetMergeCandidates(
      FragmentOp merge_into_fragment, const FragmentMergeRule* rule) const {
    Block& parent_body =
        merge_into_fragment->getParentOfType<func::FuncOp>().getBody().front();

    std::vector<FragmentOp> merge_candidates;
    for (auto fragment : parent_body.getOps<FragmentOp>()) {
      if (fragment == merge_into_fragment) {
        continue;
      }

      // Merge candidates must be on the same mesh.
      if (fragment.getMeshName() != merge_into_fragment.getMeshName()) {
        continue;
      }

      FragmentInfo fragment_info = GetFragmentInfo(fragment);
      auto it = fragment_merge_rule_map_.find(fragment_info);
      if (it != fragment_merge_rule_map_.end() && it->second == rule) {
        merge_candidates.push_back(fragment);
      }
    }

    return merge_candidates;
  }

  const FragmentMergeRuleMap& fragment_merge_rule_map_;
};

class RuleBasedMergePass
    : public impl::RuleBasedMergePassBase<RuleBasedMergePass> {
  using RuleBasedMergePassBase::RuleBasedMergePassBase;

 private:
  FragmentMergeRuleMap fragment_merge_rule_map_;
  FrozenRewritePatternSet patterns;

  void runOnFunc(func::FuncOp func) override {
    if (!IsMpmdFunction(func)) {
      return;
    }

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    config.enableFolding(false);
    config.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(func, patterns, config))) {
      return signalPassFailure();
    }

    if (!sortTopologically(&func.getBody().front())) {
      func.emitError() << "Cycle detected in the program, not all fragments "
                          "could be properly merged.";
      return signalPassFailure();
    }

    // Remove control dependencies if requested. This is typically done when
    // RuleBasedSchedulePass ran before this pass and added control
    // dependencies.
    if (removeControlDependencies) {
      RemoveAllControlDependencies(func);
    }
  }

  LogicalResult initialize(MLIRContext* context) final {
    for (const FragmentMergeRule& rule : rules) {
      for (const FragmentInfo& fragment : rule.sources) {
        SDY_CHECK(!fragment_merge_rule_map_.contains(fragment))
            << "Fragment " << llvm::to_string(fragment)
            << " is already part of another merge rule.";
        fragment_merge_rule_map_[fragment] = &rule;
      }
    }
    RewritePatternSet patternsInternal(context);
    patternsInternal.add<RuleBasedMergingPattern>(context, rules,
                                                  fragment_merge_rule_map_);
    patterns = std::move(patternsInternal);
    return success();
  }
};

}  // namespace
}  // namespace mlir::mpmd
