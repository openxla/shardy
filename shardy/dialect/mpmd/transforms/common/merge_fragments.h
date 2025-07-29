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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_MERGE_FRAGMENTS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_MERGE_FRAGMENTS_H_

#include <cstdint>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/common/distributed_function_pass.h"

namespace mlir::mpmd {

class MergeFragmentBasePass : public DistributedFunctionPass {
 public:
  explicit MergeFragmentBasePass(TypeID passID)
      : DistributedFunctionPass(passID) {}

 protected:
  // A map we will use to store the order of operations during merging.
  using OpOrderMap = DenseMap<Operation*, int32_t>;

  // Checks whether `producer_op` and `consumer_op` may be merged. Must be
  // defined by subclasses of the pass.
  //
  // If `log_failure` is true, then the reason merging isn't allowed will be
  // logged as a debug message.
  virtual LogicalResult AllowMerging(FragmentOp producer_op,
                                     FragmentOp consumer_op,
                                     bool log_failure) const = 0;

  // Checks whether a `producer_op` fragment may be cloned during merge. Must be
  // defined by subclasses of the pass.
  virtual bool AllowCloningProducerFragment(FragmentOp producer_op) const = 0;

  // Whether a producer fragment should be merged with the closest mergeable
  // consumer or with the closest consumer.
  virtual bool AllowMergingWithAnyConsumer() const = 0;

  virtual FailureOr<FragmentOp> GetMergeCandidate(FragmentOp producer_op,
                                                  OpOrderMap& order) const;

 private:
  bool FastIsBeforeInBlock(Operation* op1, Operation* op2,
                           OpOrderMap& order) const;

  // Returns true if the producer can be merged with the consumer at the
  // position of the producer: i.e., all consumer's operands are produced by
  // the producer or by an earlier op.
  bool CanMergeAtProducer(Operation* producer, Operation* consumer,
                          OpOrderMap& order) const;

  // Returns true if the producer can be merged with the consumer at the
  // position of the consumer: i.e., no ops in between the consumer and
  // producer uses the producer's results.
  bool CanMergeAtConsumer(Operation* producer, Operation* consumer,
                          OpOrderMap& order) const;

  // Tries to merge the fragment and returns the merged fragment, or a failure
  // if merging isn't possible.
  FailureOr<FragmentOp> MergeFragmentsRewrite(FragmentOp producer_op,
                                              RewriterBase& rewriter,
                                              OpOrderMap& order) const;

  // Merges fragments recursively, attempting to clone the producer
  // fragment if possible. We may want to clone to avoid introducing
  // dependencies.
  //
  // Pre-condition: All users of the producer_op have been processed by this
  // rewrite, i.e., we do the rewrite in post-order traversal.
  void MergeFragmentsRecursivelyRewrite(FragmentOp producer_op,
                                        RewriterBase& rewriter,
                                        OpOrderMap& order) const;

  void runOnFunc(func::FuncOp func_op) override;
};

// Adds the sequence of passes that merges inferred fragments with user defined
// fragments.
void AddMergeInferredFragmentsPasses(mlir::OpPassManager& pm,
  bool absorb_on_entry_point_function,
  bool clone_inferred_fragments);

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_MERGE_FRAGMENTS_H_
