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

#include "shardy/dialect/mpmd/transforms/common/merge_fragments.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/optimize/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_MERGEINFERREDFRAGMENTSPASS
#define GEN_PASS_DEF_MERGEFORWARDWITHBACKWARDPASS
#define GEN_PASS_DEF_MERGEUSERDEFINEDFRAGMENTSINTOSCHEDULINGUNITSPASS
#define GEN_PASS_DEF_VERIFYSTAGEMERGINGPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

#define DEBUG_TYPE "mpmd-merge-fragments"

namespace {

using ::mlir::func::FuncOp;

bool AreStageIdsConsistent(FragmentOp producer_op, FragmentOp consumer_op) {
  IntegerAttr producer_stage_id = producer_op.getStageIdAttr();
  IntegerAttr consumer_stage_id = consumer_op.getStageIdAttr();
  return !producer_stage_id || !consumer_stage_id ||
         producer_stage_id == consumer_stage_id;
}

// Returns the stage_id integer attribute of the resulting merged fragment.
IntegerAttr GetMergedStageIdAttribute(FragmentOp producer_op,
                                      FragmentOp consumer_op) {
  SDY_CHECK(AreStageIdsConsistent(producer_op, consumer_op))
      << "Merging requires both fragments to have the same stage id.";
  IntegerAttr producer_stage = producer_op.getStageIdAttr();
  return producer_stage ? producer_stage : consumer_op.getStageIdAttr();
}

bool TypeHasOneElement(Type type) {
  auto tensor_type = dyn_cast<RankedTensorType>(type);
  if (!tensor_type || !tensor_type.hasStaticShape()) return false;
  for (int64_t i = 0; i < tensor_type.getRank(); ++i) {
    if (tensor_type.getDimSize(i) != 1) return false;
  }
  return true;
}

// Returns true if `op` is an inter-mesh TransferOp whose global type has only
// one element.
bool IsNonScalarInterMeshTransfer(Operation* op) {
  TransferOp transfer_op = DynCastInterMeshTransfer(op);
  return transfer_op &&
         !TypeHasOneElement(transfer_op.getType().getGlobalTensorType());
}

// We only clone fragments with at most one non-return op.
inline const int kInferredFragmentMaxSizeForCloning = 2;

void PrintMergeDebug(FragmentOp producer_op) {
  LLVM_DEBUG({
    llvm::dbgs() << "\n\n=== Processing fragment: '"
                 << PrintOperationForLog(
                        producer_op,
                        OpPrintingFlags().assumeVerified().skipRegions())
                 << "'\n\n";
  });
}

LogicalResult mergeFailure(StringRef message, bool log_failure = true) {
  if (log_failure) {
    LLVM_DEBUG({ llvm::dbgs() << "Merge failed.\n" << message << "\n"; });
  }
  return failure();
}

FragmentOp FragmentOrNull(FailureOr<FragmentOp> merge_result) {
  if (succeeded(merge_result)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Merge successful.\n"
                   << (*merge_result)->getParentOfType<FuncOp>() << "\n";
    });
    return *merge_result;
  }

  return nullptr;
}

// Merges the call counters of the producer and consumer fragments.
//
// - If only one fragment has a call counter, we use that.
// - If both fragments have the same call counter, we use that.
// - If both fragments have different call counters but exactly one of them is
// user-defined, then we return the user-defined fragment's call counter.
// - Otherwise, we return std::nullopt. I.e. this information will be lost.
std::optional<int> MergeCallCounters(FragmentOp producer_op,
                                     FragmentOp consumer_op) {
  std::optional<int> producer_call_count = TryToFindCallCounter(producer_op);
  std::optional<int> consumer_call_count = TryToFindCallCounter(consumer_op);

  if (producer_call_count && !consumer_call_count) {
    return producer_call_count;
  }
  if (consumer_call_count && !producer_call_count) {
    return consumer_call_count;
  }

  if (producer_call_count && consumer_call_count) {
    if (producer_call_count == consumer_call_count) {
      return producer_call_count;
    }

    std::string debug_info;
    {
      llvm::raw_string_ostream debug_stream(debug_info);
      debug_stream << "producer counter=" << *producer_call_count
                   << " vs consumer counter=" << *consumer_call_count
                   << ". Producer origin=" << producer_op.getOrigin()
                   << " Consumer origin=" << consumer_op.getOrigin();
    }

    // Both are either user-defined or inferred.
    if (producer_op.isUserFragment() == consumer_op.isUserFragment()) {
      SDY_LOG(INFO) << "[MpmdMergeFragments] Ignoring call_counter since "
                       "different - "
                    << debug_info;
      return std::nullopt;
    }

    // Exactly one of the fragments is user-defined.
    SDY_LOG(INFO) << "[MpmdMergeFragments] Preferring call_counter of "
                     "user-defined fragment over inferred fragment - "
                  << debug_info;
    if (!producer_op.isUserFragment()) {
      return consumer_call_count;
    }
    if (!consumer_op.isUserFragment()) {
      return producer_call_count;
    }
    SDY_CHECK(false);
  }

  return std::nullopt;
}

// Returns a list of attributes that must be preserved in the merged fragment.
// Note: origins are preserved by default and require no extra work.
SmallVector<std::pair<StringRef, Attribute>> MergedAttributes(
    FragmentOp producer_op, FragmentOp consumer_op) {
  SmallVector<std::pair<StringRef, Attribute>> attributes;

  if (std::optional<int> merged_call_count =
          MergeCallCounters(producer_op, consumer_op)) {
    IRRewriter rewriter(producer_op.getContext());
    attributes.emplace_back(kCallCounterAttrName,
                            rewriter.getUI32IntegerAttr(*merged_call_count));
  }
  return attributes;
}

}  // namespace

FailureOr<FragmentOp> MergeFragmentBasePass::GetMergeCandidate(
    FragmentOp producer_op, OpOrderMap& order) const {
  SDY_CHECK(llvm::all_of(producer_op->getUsers(), [&](Operation* user) {
    return producer_op->getBlock() == user->getBlock();
  })) << "Expected all users of the producer op to be in the same block";

  SmallVector<FragmentOp> sorted_fragment_users;
  for (Operation* user : producer_op->getUsers()) {
    if (auto fragment = dyn_cast<FragmentOp>(user)) {
      sorted_fragment_users.push_back(fragment);
    }
  }
  std::stable_sort(sorted_fragment_users.begin(), sorted_fragment_users.end(),
                   [&](FragmentOp a, FragmentOp b) {
                     return FastIsBeforeInBlock(a, b, order);
                   });

  if (sorted_fragment_users.empty()) {
    return mergeFailure("Fragment is not used by any other fragment.");
  }

  if (AllowMergingWithAnyConsumer()) {
    // Find the closest user that is mergeable.
    auto* find_it = llvm::find_if(sorted_fragment_users, [&](FragmentOp user) {
      return succeeded(AllowMerging(producer_op, user, /*log_failure=*/false));
    });
    if (find_it == sorted_fragment_users.end()) {
      return mergeFailure("Failed to find a fragment user to merge with");
    }
    return *find_it;
  }

  // Merge with the closest user.
  FragmentOp mergeable_user = *sorted_fragment_users.begin();
  if (failed(AllowMerging(producer_op, mergeable_user, /*log_failure=*/true))) {
    return failure();
  }
  return mergeable_user;
}

bool MergeFragmentBasePass::FastIsBeforeInBlock(Operation* op1, Operation* op2,
                                                OpOrderMap& order) const {
  auto it1 = order.find(op1);
  auto it2 = order.find(op2);
  SDY_CHECK(it1 != order.end() && it2 != order.end())
      << "Expected both ops to be in the order map";
  return it1->second < it2->second;
}

// Returns true if the producer can be merged with the consumer at the
// position of the producer: i.e., all consumer's operands are produced by
// the producer or by an earlier op.
bool MergeFragmentBasePass::CanMergeAtProducer(Operation* producer,
                                               Operation* consumer,
                                               OpOrderMap& order) const {
  return llvm::all_of(consumer->getOpOperands(), [&](OpOperand& operand) {
    Operation* operand_producer = operand.get().getDefiningOp();
    return !operand_producer || operand_producer == producer ||
           FastIsBeforeInBlock(operand_producer, producer, order);
  });
}

// Returns true if the producer can be merged with the consumer at the
// position of the consumer: i.e., no ops in between the consumer and
// producer uses the producer's results.
bool MergeFragmentBasePass::CanMergeAtConsumer(Operation* producer,
                                               Operation* consumer,
                                               OpOrderMap& order) const {
  return llvm::none_of(producer->getUsers(), [&](Operation* op) {
    return FastIsBeforeInBlock(op, consumer, order);
  });
}

// Tries to merge the fragment and returns the merged fragment, or an error
// status if merging isn't possible.
FailureOr<FragmentOp> MergeFragmentBasePass::MergeFragmentsRewrite(
    FragmentOp producer_op, RewriterBase& rewriter, OpOrderMap& order) const {
  FragmentOp mergeable_user;
  if (FailureOr<FragmentOp> merge_candidate =
          GetMergeCandidate(producer_op, order);
      succeeded(merge_candidate)) {
    mergeable_user = *merge_candidate;
  } else {
    return failure();
  }

  // Find the position to merge the producer and the consumer.
  // By default, we merge with the consumer. Otherwise check if we can merge
  // with the producer.
  bool can_merge_at_consumer = true;
  if (!CanMergeAtConsumer(producer_op, mergeable_user, order)) {
    can_merge_at_consumer = false;
    if (!CanMergeAtProducer(producer_op, mergeable_user, order)) {
      return mergeFailure(
          "The consumer and producer can't be merged at any position.");
    }
    mergeable_user->moveAfter(producer_op);
  }

  // Proceeding with actual merging.
  // NOTE: Merging `producer_op` and `closest_user` may delay transfers
  // depending on the definition of AllowMerging.

  if (AllowCloningProducerFragment(producer_op)) {
    rewriter.setInsertionPoint(producer_op);
    FragmentOp cloned_producer_op =
        cast<FragmentOp>(rewriter.clone(*producer_op));
    for (auto [original_result, cloned_result] :
         llvm::zip(producer_op.getResults(), cloned_producer_op.getResults())) {
      rewriter.replaceUsesWithIf(
          original_result, cloned_result, [mergeable_user](OpOperand& operand) {
            // We also replace uses in any func return ops, so that the
            // original fragment can be removed if it's completely merged.
            return operand.getOwner() == mergeable_user ||
                   isa<func::ReturnOp>(operand.getOwner());
          });
    }
    order[cloned_producer_op] = order[producer_op];
    producer_op = cloned_producer_op;
  }

  SmallVector<std::pair<StringRef, Attribute>> merged_attributes =
      MergedAttributes(producer_op, mergeable_user);

  // Now we can merge `producer_op` with `consumer_op`.
  FragmentOp merged_fragment = MergeRegionOps(
      producer_op, mergeable_user, rewriter,
      /*num_static_args=*/0, /*replace_producer_use_in_consumer_block=*/
      [](OpOperand&, Value) {
        SDY_CHECK(false) << "Fragment ops shouldn't have free variables";
      },
      GetFragmentOriginUnion(producer_op, mergeable_user, rewriter),
      producer_op.getMeshNameAttr(),
      /*stage_id=*/GetMergedStageIdAttribute(producer_op, mergeable_user));

  for (const auto [attr_name, attr] : merged_attributes) {
    merged_fragment->setAttr(attr_name, attr);
  }

  // Notice that we merge at the producer by first moving the consumer right
  // after the producer above, still `can_merge_at_consumer` is with respect
  // to the original consumer, and hence we update the order map accordingly.
  order[merged_fragment] =
      can_merge_at_consumer ? order[mergeable_user] : order[producer_op];
  return merged_fragment;
}

// Merges fragments recursively, attempting to clone the producer
// fragment if possible. We may want to clone to avoid introducing
// dependencies.
//
// Pre-condition: All users of the producer_op have been processed by this
// rewrite, i.e., we do the rewrite in post-order traversal.
void MergeFragmentBasePass::MergeFragmentsRecursivelyRewrite(
    FragmentOp producer_op, RewriterBase& rewriter, OpOrderMap& order) const {
  bool clone_producer_op = AllowCloningProducerFragment(producer_op);
  FragmentOp last_merged = producer_op;
  // Because all users have been processed, we will never need to merge
  // transitive users. E.g., when processing f1, we won't have a mergeable
  // chain f1 -> f2 -> f3 because if f2 and f3 could be merged, they would
  // already be merged. But we still need to do recursive merging for when f1
  // has multiple users.
  while (last_merged) {
    // TODO(b/313631663) - Remove cloning from merging.
    FragmentOp to_merge = clone_producer_op ? producer_op : last_merged;
    PrintMergeDebug(to_merge);
    last_merged =
        FragmentOrNull(MergeFragmentsRewrite(to_merge, rewriter, order));
  }

  if (clone_producer_op && producer_op.use_empty() && isPure(producer_op)) {
    rewriter.eraseOp(producer_op);
  }
}

void MergeFragmentBasePass::runOnFunc(FuncOp func_op) {
  MLIRContext* context = func_op->getContext();
  IRRewriter rewriter(context);

  OpOrderMap order;
  // Traversals are cheap but growing the map is expensive, so we pre-grow it.
  order.reserve(
      std::distance(func_op.getOps().begin(), func_op.getOps().end()) + 1);
  for (auto [index, op] : llvm::enumerate(func_op.getOps())) {
    // NB: we start at 1 to have a way to verify that an op is not in the map
    // by checking if order[op] == 0.
    order[&op] = index + 1;
  }

  // We do a post-order traversal as we want to process all users before the
  // op itself, for guarantees around unused fragment removal (see below).
  //
  // We prove that every mergeable fragment indeed gets merged:
  // Suppose not, i.e., there's a fragment f1 that can be merged with a
  // consumer fragment f2, but was not merged. When f1 was visited, we
  // didn't merge with f2. This means that f2 did not exist at that point
  // (otherwise we would merge). I.e., f2 is the result of merging some
  // fragments g1,..., gn, none of which can be merged into f1 and at
  // least one of which is a consumer of f1. Assume g1 is this consumer
  // fragment. But whatever makes g1 unmergeable with f1 will also make f2
  // unmergeable with f1, e.g., having an inter-mesh transfer operand in g1.
  // Thus such an f2 cannot exist.
  func_op.walk<WalkOrder::PostOrder, ReverseIterator>([&](FragmentOp fragment) {
    // We do some simple folding to avoid merging removable fragments.
    // When we act on a fragment, all its users have been visited
    // (because of the post-order traversal), so we are guaranteed not
    // to merge with any removable fragments as they would've already
    // been removed.
    if (fragment.use_empty() && isPure(fragment)) {
      rewriter.eraseOp(fragment);
    } else {
      // We need to merge recursively, because a fragment `f1` could be
      // independently used by multiple other fragments, e.g., `f2(f1)`
      // and `f3(f1)`. So we need to recursively merge until all
      // eligible consuming fragments are merged.
      MergeFragmentsRecursivelyRewrite(fragment, rewriter, order);
    }
    return WalkResult::skip();
  });
}

namespace {

class MergeInferredFragmentsPass
    : public impl::MergeInferredFragmentsPassBase<MergeInferredFragmentsPass> {
  using MergeInferredFragmentsPassBase::MergeInferredFragmentsPassBase;

 protected:
  LogicalResult AllowMerging(FragmentOp producer_op, FragmentOp consumer_op,
                             bool log_failure) const final {
    if (producer_op.isUserFragment() && consumer_op.isUserFragment()) {
      return mergeFailure("Cannot merge two user fragments", log_failure);
    }

    if (IsSplitKeepTransferred(producer_op) &&
        IsSplitDropTransferred(consumer_op)) {
      return mergeFailure(
          "Fragments were split, so they should not be merged back.",
          log_failure);
    }

    // If the producer or the consumer op are inferred fragments, then we
    // consider them negligible, i.e., can't delay transfers. Thus, it is safe
    // to merge the fragments.
    return success();
  }

  bool AllowCloningProducerFragment(FragmentOp producer_op) const final {
    return cloneInferredFragments && !producer_op.isUserFragment() &&
           isPure(producer_op) &&
           producer_op.getBody()->getOperations().size() <=
               kInferredFragmentMaxSizeForCloning &&
           producer_op->getNumResults() == 1;
  }

  bool AllowMergingWithAnyConsumer() const final { return mergeAnyConsumer; }

  FailureOr<FragmentOp> GetMergeCandidate(FragmentOp producer_op,
                                          OpOrderMap& order) const final {
    if (!mergeSideways) {
      return MergeFragmentBasePass::GetMergeCandidate(producer_op, order);
    }

    SDY_CHECK(!cloneInferredFragments && !mergeAnyConsumer)
        << "merge-sideways cannot be used with clone-inferred-fragments "
           "or merge-any-consumer";

    Operation* current = producer_op->getNextNode();
    while (current) {
      if (auto fragment = dyn_cast<FragmentOp>(current);
          fragment && fragment.getMeshName() == producer_op.getMeshName()) {
        current = fragment;
        break;
      }
      if (auto transfer = dyn_cast<TransferOp>(current);
          transfer &&
          transfer.getType().getMeshName() == producer_op.getMeshName()) {
        // If we transfer back into the same mesh, then abort, because the
        // next node could have used the transferred value.
        return mergeFailure(
            "The closest fragment in the same mesh is a transfer. We do not "
            "merge sideways for simplicity, to avoid having to check "
            "dependencies.");
      }
      current = current->getNextNode();
    }

    if (!current) {
      return mergeFailure("No mergeable fragment in the same mesh.");
    }

    auto merge_candidate = cast<FragmentOp>(current);
    if (failed(AllowMerging(producer_op, merge_candidate,
                            /*log_failure=*/true))) {
      return failure();
    }
    return merge_candidate;
  }
};

class MergeForwardWithBackwardPass
    : public impl::MergeForwardWithBackwardPassBase<
          MergeForwardWithBackwardPass> {
  using MergeForwardWithBackwardPassBase::MergeForwardWithBackwardPassBase;

 protected:
  LogicalResult AllowMerging(FragmentOp producer_op, FragmentOp consumer_op,
                             bool log_failure) const final {
    // Check that the producer forward fragment and is immediately before
    // the consumer backward fragment. This is only true for the last stage
    // for 1F1B so it will not merge any fragments in previous stages, which
    // is the intended behavior.
    if (!IsExecutedImmediatelyAfter(producer_op, consumer_op)) {
      return mergeFailure(
          "The consumer fragment must appear immediately after the "
          "producer fragment (modulo fragments in other meshes).",
          log_failure);
    }
    if (!IsForwardFragment(producer_op) || !IsBackwardFragment(consumer_op)) {
      return mergeFailure(
          "The producer fragment must be a forward fragment and the "
          "consumer fragment must be a backward fragment.",
          log_failure);
    }
    return success();
  }

  bool AllowCloningProducerFragment(FragmentOp producer_op) const final {
    return false;
  }

  bool AllowMergingWithAnyConsumer() const final { return false; }
};

class MergeUserDefinedFragmentsIntoSchedulingUnitsPass
    : public impl::MergeUserDefinedFragmentsIntoSchedulingUnitsPassBase<
          MergeUserDefinedFragmentsIntoSchedulingUnitsPass> {
  using MergeUserDefinedFragmentsIntoSchedulingUnitsPassBase::
      MergeUserDefinedFragmentsIntoSchedulingUnitsPassBase;

 protected:
  LogicalResult AllowMerging(FragmentOp producer_op, FragmentOp consumer_op,
                             bool log_failure) const final {
    if (!producer_op.isUserFragment() || !consumer_op.isUserFragment()) {
      return mergeFailure("Cannot merge inferred fragments", log_failure);
    }

    std::optional<int> producer_transpose_count =
        TryToFindSingleTransposeCount(producer_op);
    SDY_CHECK(producer_transpose_count.has_value())
        << "Expected one and only one transpose count for a user-defined "
           "producer op.";

    std::optional<int> consumer_transpose_count =
        TryToFindSingleTransposeCount(consumer_op);
    SDY_CHECK(consumer_transpose_count.has_value())
        << "Expected one and only one transpose count for a user-defined "
           "consumer op.";

    if (*producer_transpose_count != *consumer_transpose_count) {
      return mergeFailure(
          "Cannot merge fragments with different transpose counts.",
          log_failure);
    }

    std::optional<int> producer_call_count = TryToFindCallCounter(producer_op);
    std::optional<int> consumer_call_count = TryToFindCallCounter(consumer_op);
    if (producer_call_count.has_value() && consumer_call_count.has_value() &&
        *producer_call_count != *consumer_call_count) {
      return mergeFailure("Cannot merge fragments with different call counts.",
                          log_failure);
    }

    if (!AreStageIdsConsistent(producer_op, consumer_op)) {
      return mergeFailure(
          "If both fragments have stage_ids, then they must be identical or "
          "they cannot be merged.",
          log_failure);
    }
    return success();
  }

  bool AllowCloningProducerFragment(FragmentOp producer_op) const final {
    // User defined fragments can never be cloned.
    return false;
  }

  bool AllowMergingWithAnyConsumer() const final { return false; }
};

// Attributes which should uniquely identify each fragment if stages have been
// assigned, needed to check if stage assignment is consistent.
struct FragmentSignature {
  StringRef mesh;
  int64_t stage_id;
  int64_t transpose_count;
  std::optional<int64_t> call_count;
  std::optional<SplitFragmentType> split_type;
  bool is_remat;
};

// Two fragments match _iff_ they have the same mesh assignment, same stage
// assignment, same transpose_count, same split fragment type, same remat
// flag, and the call_counts are consistent, i.e., they are the same or one of
// them is undefined. This property is not transitive and therefore we don't
// define it as an equality operator.
bool FragmentSignaturesMatch(FragmentSignature signature,
                             FragmentSignature other) {
  return signature.mesh == other.mesh && signature.stage_id == other.stage_id &&
         signature.transpose_count == other.transpose_count &&
         (!signature.call_count.has_value() || !other.call_count.has_value() ||
          *signature.call_count == *other.call_count) &&
         (signature.split_type == other.split_type) &&
         (signature.is_remat == other.is_remat);
}

class VerifyStageMergingPass
    : public impl::VerifyStageMergingPassBase<VerifyStageMergingPass> {
  using VerifyStageMergingPassBase::VerifyStageMergingPassBase;

 private:
  void runOnFunc(FuncOp func_op) override {
    // We keep track of various attributes relevant to stage merging for every
    // fragment in the module. If we find any two equivalent fragments (see
    // `FragmentSignature` above), then we emit an error.
    std::vector<FragmentSignature> visited_fragments;
    bool has_error = false;
    func_op.walk([&](FragmentOp fragment) {
      if (IntegerAttr stage_id_attr = fragment.getStageIdAttr()) {
        StringRef mesh = fragment.getMeshName();
        std::optional<int64_t> transpose_count =
            TryToFindSingleTransposeCount(fragment);
        // If the fragment has a stage, then it is a user-defined fragment,
        // which means it has a transpose count.
        SDY_CHECK(transpose_count.has_value());

        FragmentSignature fragment_signature = {mesh,
                                                stage_id_attr.getInt(),
                                                *transpose_count,
                                                TryToFindCallCounter(fragment),
                                                GetSplitFragmentType(fragment),
                                                IsRemat(fragment)};
        if (llvm::any_of(visited_fragments,
                         [&fragment_signature](const FragmentSignature& other) {
                           return FragmentSignaturesMatch(fragment_signature,
                                                          other);
                         })) {
          emitError(fragment.getLoc())
              << "A valid program cannot have more than one fragment with the "
                 "same mesh, stage, transpose, call counts, split type, and "
                 "remat flag but found multiple fragments with the same "
                 "attributes: [mesh="
              << fragment_signature.mesh
              << ", stage_id=" << fragment_signature.stage_id
              << ", transpose_count=" << fragment_signature.transpose_count
              << ", call_count=" << fragment_signature.call_count.value_or(-1)
              << ", split_type=" << fragment_signature.split_type
              << ", is_remat=" << fragment_signature.is_remat
              << "] for current fragment with origin: " << fragment.getOrigin();
          has_error = true;
        } else {
          visited_fragments.push_back(fragment_signature);
        }
      }
    });

    if (has_error) {
      signalPassFailure();
    }
  }
};

}  // namespace

void AddMergeInferredFragmentsPasses(OpPassManager& pm,
                                     bool absorb_on_entry_point_function,
                                     bool clone_inferred_fragments) {
  // Absorb inferred fragments into user defined fragments to guarantee that
  // the shape of the program in the body of a call-op or microbatching loop
  // resembles the shape the user defined, via named computations + stage/mesh
  // mapping (e.g., a V shaped pipeline).
  pm.addNestedPass<FuncOp>(createAbsorbInferredFragmentsPass(
      AbsorbInferredFragmentsPassOptions{absorb_on_entry_point_function}));

  // Clean up the module by merging inferred fragments with user defined
  // fragments (i.e., fragments that result from NamedComputations).
  {
    MergeInferredFragmentsPassOptions merge_inferred_options;
    merge_inferred_options.cloneInferredFragments = clone_inferred_fragments;
    merge_inferred_options.mergeAnyConsumer = true;
    pm.addNestedPass<FuncOp>(
        createMergeInferredFragmentsPass(std::move(merge_inferred_options)));
  }
  {
    MergeInferredFragmentsPassOptions merge_inferred_options;
    merge_inferred_options.mergeSideways = true;
    pm.addNestedPass<FuncOp>(
        createMergeInferredFragmentsPass(std::move(merge_inferred_options)));
  }
}

}  // namespace mlir::mpmd

#undef DEBUG_TYPE
