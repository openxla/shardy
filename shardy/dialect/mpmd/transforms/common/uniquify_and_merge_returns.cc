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
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_UNIQUIFYANDMERGERETURNSPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

using ::mlir::func::FuncOp;

// Creates a tiny identity fragment that takes `value` as input and returns it
// `num_copies` times. This fragment is intended to be immediately merged.
FragmentOp CreateIdentityFragment(Value value, int64_t num_copies,
                                  StringRef mesh_name, Operation* insert_before,
                                  OpBuilder& builder) {
  builder.setInsertionPoint(insert_before);
  auto loc = insert_before->getLoc();
  auto mesh_type = cast<MeshTensorType>(value.getType());

  SmallVector<Type> result_types(num_copies, mesh_type);

  auto fragment_op = FragmentOp::create(
      builder, loc, result_types, /*operands=*/ValueRange{value},
      /*user_origin=*/ArrayAttr::get(builder.getContext(), {}),
      /*mesh_name=*/mesh_name, /*stage_id=*/IntegerAttr());

  Block& fragment_block = fragment_op.getRegion().emplaceBlock();
  sdy::MeshAttr mesh_attr = GetMeshOrFail(fragment_op, mesh_name);
  BlockArgument inner_arg = fragment_block.addArgument(
      GetGlobalTensorTypeFromMeshType(value, mesh_attr), value.getLoc());

  SmallVector<Value> return_values(num_copies, inner_arg);
  auto block_builder = OpBuilder::atBlockEnd(&fragment_block);
  ReturnOp::create(block_builder, loc, return_values);

  return fragment_op;
}

class UniquifyAndMergeReturnsPass
    : public impl::UniquifyAndMergeReturnsPassBase<
          UniquifyAndMergeReturnsPass> {
  using UniquifyAndMergeReturnsPassBase::UniquifyAndMergeReturnsPassBase;

 protected:
  void runOnFunc(FuncOp func_op) override {
    if (!IsMpmdFunction(func_op)) {
      return;
    }

    Operation* return_op = func_op.getBody().front().getTerminator();
    IRRewriter rewriter(&getContext());

    // Process values that need uniquification. We rebuild the map after each
    // merge because MergeRegionOps erases the old producer, invalidating any
    // Value handles we stored from its results.
    bool changed = true;
    while (changed) {
      changed = false;

      llvm::MapVector<Value, SmallVector<int64_t>> value_to_return_indices;
      for (OpOperand& operand : return_op->getOpOperands()) {
        value_to_return_indices[operand.get()].push_back(
            operand.getOperandNumber());
      }

      for (auto& [value, return_indices] : value_to_return_indices) {
        bool is_block_arg = isa<BlockArgument>(value);
        bool is_duplicate = return_indices.size() > 1;

        if (!is_block_arg && !is_duplicate) {
          continue;  // Single use, non-block-arg: no work needed.
        }

        auto mesh_type = cast<MeshTensorType>(value.getType());
        StringRef mesh_name = mesh_type.getMeshName();

        if (auto producing_fragment = value.getDefiningOp<FragmentOp>()) {
          // Collect ALL duplicate results from this same producer, since
          // MergeRegionOps will RAUW all of the producer's results.  If we
          // only handle one result, the others stay duplicated.
          SmallVector<std::pair<Value, SmallVector<int64_t>>>
              results_needing_copies;
          for (auto& [v, ri] : value_to_return_indices) {
            if (v.getDefiningOp() == producing_fragment.getOperation() &&
                ri.size() > 1) {
              results_needing_copies.push_back({v, ri});
            }
          }

          // Build a single identity fragment that takes all duplicate results
          // as inputs and returns extra copies for each.
          int64_t total_copies = 0;
          for (auto& [v, ri] : results_needing_copies) {
            total_copies += ri.size() - 1;  // First use stays as original.
          }

          if (total_copies == 0) continue;

          // Build the identity fragment manually: one input per duplicate
          // result, each returned (num_dups - 1) times.
          rewriter.setInsertionPointAfter(producing_fragment);
          auto loc = producing_fragment->getLoc();

          SmallVector<Value> operands;
          SmallVector<Type> result_types;
          for (auto& [v, ri] : results_needing_copies) {
            operands.push_back(v);
            auto mt = cast<MeshTensorType>(v.getType());
            int64_t copies = ri.size() - 1;
            result_types.insert(result_types.end(), copies, mt);
          }

          auto identity = FragmentOp::create(
              rewriter, loc, result_types, operands,
              /*user_origin=*/ArrayAttr::get(rewriter.getContext(), {}),
              /*mesh_name=*/mesh_name, /*stage_id=*/IntegerAttr());

          Block& block = identity.getRegion().emplaceBlock();
          sdy::MeshAttr mesh_attr = GetMeshOrFail(identity, mesh_name);
          SmallVector<Value> return_values;
          for (auto& [v, ri] : results_needing_copies) {
            BlockArgument arg = block.addArgument(
                GetGlobalTensorTypeFromMeshType(v, mesh_attr), v.getLoc());
            int64_t copies = ri.size() - 1;
            return_values.insert(return_values.end(), copies, arg);
          }
          auto block_builder = OpBuilder::atBlockEnd(&block);
          ReturnOp::create(block_builder, loc, return_values);

          // Merge into producer.
          FragmentOp merged = MergeRegionOps(
              producing_fragment, identity, rewriter,
              /*num_static_args=*/0,
              /*replace_producer_use_in_consumer_block=*/
              [](OpOperand&, Value) {
                SDY_CHECK(false)
                    << "Fragment ops shouldn't have free variables";
              },
              GetFragmentOriginUnion(producing_fragment, identity, rewriter),
              producing_fragment.getMeshNameAttr(),
              /*stage_id=*/producing_fragment.getStageIdAttr());

          // Fix up return operands.  The identity copies are the last
          // total_copies results of the merged fragment.
          int64_t merged_num_results = merged->getNumResults();
          int64_t copy_idx = merged_num_results - total_copies;
          for (auto& [v, ri] : results_needing_copies) {
            for (int64_t i = 1; i < static_cast<int64_t>(ri.size()); ++i) {
              return_op->setOperand(ri[i], merged->getResult(copy_idx++));
            }
          }
          changed = true;
          break;  // Restart: map is invalidated.
        }

        if (!is_block_arg) continue;

        int64_t num_copies = return_indices.size();

        // Find an existing fragment on the same mesh to merge into.
        FragmentOp target_fragment = nullptr;
        func_op.walk([&](FragmentOp fragment) {
          if (fragment.getMeshName() == mesh_name) {
            target_fragment = fragment;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

        FragmentOp identity = CreateIdentityFragment(
            value, num_copies, mesh_name, return_op, rewriter);

        if (target_fragment) {
          identity->moveAfter(target_fragment);

          FragmentOp merged = MergeRegionOps(
              target_fragment, identity, rewriter,
              /*num_static_args=*/0,
              /*replace_producer_use_in_consumer_block=*/
              [](OpOperand&, Value) {
                SDY_CHECK(false)
                    << "Fragment ops shouldn't have free variables";
              },
              GetFragmentOriginUnion(target_fragment, identity, rewriter),
              target_fragment.getMeshNameAttr(),
              /*stage_id=*/target_fragment.getStageIdAttr());

          int64_t merged_num_results = merged->getNumResults();
          for (int64_t i = 0; i < num_copies; ++i) {
            return_op->setOperand(
                return_indices[i],
                merged->getResult(merged_num_results - num_copies + i));
          }
        } else {
          // No existing fragment — keep the identity as a fallback.
          for (int64_t i = 0; i < num_copies; ++i) {
            return_op->setOperand(return_indices[i], identity->getResult(i));
          }
        }
        changed = true;
        break;  // Restart: map is invalidated.
      }
    }

    // Post-condition: verify that the uniquification contract holds.
    // Every return operand must be unique and no block arguments may be
    // returned directly.
    llvm::SmallDenseSet<Value> seenValues;
    for (OpOperand& operand : return_op->getOpOperands()) {
      Value v = operand.get();
      SDY_CHECK(!isa<BlockArgument>(v))
          << "UniquifyAndMergeReturnsPass: block argument still returned "
             "directly at index "
          << operand.getOperandNumber();
      SDY_CHECK(seenValues.insert(v).second)
          << "UniquifyAndMergeReturnsPass: duplicate return operand at index "
          << operand.getOperandNumber();
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
