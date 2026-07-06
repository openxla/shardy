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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_UNIQUIFYANDMERGERETURNSPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

// Holds information about a value that needs to be uniquified, and which
// fragment to merge it into.
struct UniqueReturnInfo {
  // The return operand index that needs a new unique value.
  int64_t return_index;
  // The value being duplicated.
  Value value;
};

// Rebuilds a fragment with extra results that duplicate existing inner return
// values. `extra_results` maps from the inner return operand index to the
// number of extra copies needed.
//
// Returns a mapping from (original value, copy index) to the new fragment
// result index.
FragmentOp RebuildFragmentWithExtraResults(
    FragmentOp fragment_op,
    // Maps: inner return operand index -> number of extra copies.
    llvm::SmallDenseMap<int64_t, int64_t>& inner_return_idx_to_extra_copies,
    // Output: maps (fragment result index of original, copy_index) -> new
    // result index.
    llvm::SmallDenseMap<std::pair<int64_t, int64_t>, int64_t>&
        extra_result_indices,
    OpBuilder& builder) {
  Operation* inner_return_op = fragment_op.getRegion().front().getTerminator();

  // Build the new return types and inner return operands.
  SmallVector<Type> new_result_types(fragment_op->getResultTypes());
  SmallVector<Value> new_inner_return_operands(inner_return_op->getOperands());

  for (auto& [inner_idx, num_copies] : inner_return_idx_to_extra_copies) {
    Value inner_value = inner_return_op->getOperand(inner_idx);
    Type result_type = fragment_op->getResult(inner_idx).getType();
    for (int64_t copy = 0; copy < num_copies; ++copy) {
      extra_result_indices[{inner_idx, copy}] = new_result_types.size();
      new_result_types.push_back(result_type);
      new_inner_return_operands.push_back(inner_value);
    }
  }

  // Clone the fragment with the new result types.
  builder.setInsertionPoint(fragment_op);
  auto new_fragment_op = FragmentOp::create(
      builder, fragment_op.getLoc(), new_result_types,
      fragment_op->getOperands(), fragment_op.getOriginAttr(),
      fragment_op.getMeshNameAttr(), fragment_op.getStageIdAttr());

  // Move the region from the old fragment to the new one.
  new_fragment_op.getRegion().takeBody(fragment_op.getRegion());

  // Update the inner return op to return the extra values.
  Operation* new_inner_return =
      new_fragment_op.getRegion().front().getTerminator();
  builder.setInsertionPoint(new_inner_return);
  ReturnOp::create(builder, new_inner_return->getLoc(),
                   new_inner_return_operands);
  new_inner_return->erase();

  // Replace all uses of the old fragment results with new fragment results.
  for (int64_t i = 0; i < static_cast<int64_t>(fragment_op->getNumResults());
       ++i) {
    fragment_op->getResult(i).replaceAllUsesWith(new_fragment_op->getResult(i));
  }

  // Copy any extra attributes.
  for (NamedAttribute attr : fragment_op->getAttrs()) {
    if (!new_fragment_op->hasAttr(attr.getName())) {
      new_fragment_op->setAttr(attr.getName(), attr.getValue());
    }
  }

  fragment_op->erase();
  return new_fragment_op;
}

// Rebuilds a fragment with an extra passthrough operand. The block argument
// is added to the fragment as a new operand, and the inner body passes it
// through as an additional result.
//
// Returns a pair of (new_fragment_op, new_result_index).
std::pair<FragmentOp, int64_t> RebuildFragmentWithPassthrough(
    FragmentOp fragment_op, Value block_arg_value, int64_t num_copies,
    llvm::SmallDenseMap<std::pair<int64_t, int64_t>, int64_t>&
        extra_result_indices,
    int64_t passthrough_id, OpBuilder& builder) {
  Operation* inner_return_op = fragment_op.getRegion().front().getTerminator();

  sdy::MeshAttr mesh_attr =
      GetMeshOrFail(fragment_op, fragment_op.getMeshName());

  // Build new operands (existing + the new block arg value).
  SmallVector<Value> new_operands(fragment_op->getOperands());
  new_operands.push_back(block_arg_value);

  // Build new result types (existing + copies of the passthrough).
  SmallVector<Type> new_result_types(fragment_op->getResultTypes());
  for (int64_t copy = 0; copy < num_copies; ++copy) {
    extra_result_indices[{passthrough_id, copy}] = new_result_types.size();
    new_result_types.push_back(cast<MeshTensorType>(block_arg_value.getType()));
  }

  // Build new inner return operands.
  SmallVector<Value> new_inner_return_operands(inner_return_op->getOperands());

  // Clone the fragment.
  builder.setInsertionPoint(fragment_op);
  auto new_fragment_op = FragmentOp::create(
      builder, fragment_op.getLoc(), new_result_types, new_operands,
      fragment_op.getOriginAttr(), fragment_op.getMeshNameAttr(),
      fragment_op.getStageIdAttr());

  // Move the region.
  new_fragment_op.getRegion().takeBody(fragment_op.getRegion());

  // Add the new block argument for the passthrough operand.
  Block& block = new_fragment_op.getRegion().front();
  Value new_block_arg = block.addArgument(
      GetGlobalTensorTypeFromMeshType(block_arg_value, mesh_attr),
      block_arg_value.getLoc());

  // Update the inner return op to also return the passthrough value.
  for (int64_t copy = 0; copy < num_copies; ++copy) {
    new_inner_return_operands.push_back(new_block_arg);
  }

  Operation* new_inner_return =
      new_fragment_op.getRegion().front().getTerminator();
  builder.setInsertionPoint(new_inner_return);
  ReturnOp::create(builder, new_inner_return->getLoc(),
                   new_inner_return_operands);
  new_inner_return->erase();

  // Replace all uses of old fragment results.
  for (int64_t i = 0; i < static_cast<int64_t>(fragment_op->getNumResults());
       ++i) {
    fragment_op->getResult(i).replaceAllUsesWith(new_fragment_op->getResult(i));
  }

  // Copy extra attributes.
  for (NamedAttribute attr : fragment_op->getAttrs()) {
    if (!new_fragment_op->hasAttr(attr.getName())) {
      new_fragment_op->setAttr(attr.getName(), attr.getValue());
    }
  }

  fragment_op->erase();
  return {new_fragment_op, extra_result_indices[{passthrough_id, 0}]};
}

// Creates a new inferred fragment for a block argument that has no existing
// same-mesh fragment to merge into. This is the fallback case (e.g., identity
// function).
void CreateFallbackFragmentForBlockArg(StringRef mesh_name,
                                       Value block_arg_value,
                                       SmallVector<int64_t>& return_indices,
                                       Operation* return_op,
                                       OpBuilder& builder) {
  builder.setInsertionPoint(return_op);

  int64_t num_results = return_indices.size();
  SmallVector<Type> result_types(
      num_results, cast<MeshTensorType>(block_arg_value.getType()));

  auto fragment_op = FragmentOp::create(
      builder, return_op->getLoc(), result_types,
      /*operands=*/ValueRange{block_arg_value},
      /*user_origin=*/ArrayAttr::get(builder.getContext(), {}),
      /*mesh_name=*/mesh_name, /*stage_id=*/IntegerAttr());

  Block& fragment_block = fragment_op.getRegion().emplaceBlock();
  sdy::MeshAttr mesh_attr = GetMeshOrFail(fragment_op, mesh_name);
  Value inner_arg = fragment_block.addArgument(
      GetGlobalTensorTypeFromMeshType(block_arg_value, mesh_attr),
      block_arg_value.getLoc());

  SmallVector<Value> inner_returns(num_results, inner_arg);
  auto block_builder = OpBuilder::atBlockEnd(&fragment_block);
  ReturnOp::create(block_builder, return_op->getLoc(), inner_returns);

  for (int64_t i = 0; i < num_results; ++i) {
    return_op->setOperand(return_indices[i], fragment_op->getResult(i));
  }
}

class UniquifyAndMergeReturnsPass
    : public impl::UniquifyAndMergeReturnsPassBase<
          UniquifyAndMergeReturnsPass> {
  using UniquifyAndMergeReturnsPassBase::UniquifyAndMergeReturnsPassBase;

 protected:
  void runOnFunc(func::FuncOp func_op) override {
    if (!IsMpmdFunction(func_op)) {
      return;
    }

    Operation* return_op = func_op.getBody().front().getTerminator();
    OpBuilder builder(&getContext());

    // Step 1: Identify values that need uniquification.
    // For each value, track all the return indices where it appears.
    llvm::MapVector<Value, SmallVector<int64_t>> value_to_return_indices;
    for (OpOperand& operand : return_op->getOpOperands()) {
      value_to_return_indices[operand.get()].push_back(
          operand.getOperandNumber());
    }

    // A value needs uniquification if:
    // - It appears more than once in the return, OR
    // - It is a block argument (needs to be returned through a fragment).
    // We skip values that appear exactly once and are not block arguments.

    // Group by producing fragment, or by mesh for block arguments.
    // fragment_to_extras[fragment] = list of (inner_return_idx, extra_copies)
    llvm::DenseMap<FragmentOp, llvm::SmallDenseMap<int64_t, int64_t>>
        fragment_to_extras;

    // For values produced by a fragment: track which return indices map to
    // which fragment result, and how many extras are needed.
    // produced_value_fixups[fragment][inner_return_idx] = list of return
    // indices to fix up (the extra ones beyond the first use).
    llvm::DenseMap<FragmentOp,
                   llvm::SmallDenseMap<int64_t, SmallVector<int64_t>>>
        fragment_extra_return_indices;

    // For block arguments that need merging into an existing fragment.
    struct BlockArgMergeInfo {
      Value block_arg;
      StringRef mesh_name;
      SmallVector<int64_t> return_indices;
    };
    SmallVector<BlockArgMergeInfo> block_arg_merges;

    for (auto& [value, return_indices] : value_to_return_indices) {
      bool is_block_arg = isa<BlockArgument>(value);
      bool is_duplicate = return_indices.size() > 1;

      if (!is_block_arg && !is_duplicate) {
        // Single use, non-block-arg: no work needed.
        continue;
      }

      if (auto defining_op = value.getDefiningOp<FragmentOp>()) {
        // Value is produced by a fragment. We can add extra results directly.
        // Find which result index of the fragment this value corresponds to.
        int64_t fragment_result_idx = -1;
        for (int64_t i = 0;
             i < static_cast<int64_t>(defining_op->getNumResults()); ++i) {
          if (defining_op->getResult(i) == value) {
            fragment_result_idx = i;
            break;
          }
        }
        SDY_CHECK(fragment_result_idx >= 0)
            << "Value should be a result of its defining FragmentOp";

        // The first occurrence stays as-is. Extra copies needed = size - 1
        // for duplicates, or size for block args (but block args won't reach
        // here).
        int64_t num_extras = return_indices.size() - 1;
        if (num_extras > 0) {
          fragment_to_extras[defining_op][fragment_result_idx] = num_extras;
          // The first return index keeps using the original result.
          // The rest need new results.
          SmallVector<int64_t> extras(return_indices.begin() + 1,
                                      return_indices.end());
          fragment_extra_return_indices[defining_op][fragment_result_idx] =
              std::move(extras);
        }
      } else if (is_block_arg) {
        // Block argument. Need to merge into an existing same-mesh fragment.
        auto mesh_type = cast<MeshTensorType>(value.getType());
        block_arg_merges.push_back(
            {value, mesh_type.getMeshName(), std::move(return_indices)});
      }
      // Other cases (e.g., value produced by transfer): these would need
      // uniquification too. For now, we handle them like the original uniquify
      // would - but since the export pipeline typically doesn't have bare
      // transfers feeding into returns, this shouldn't occur in practice.
    }

    // Step 2: Process fragment-produced values - add extra results.
    for (auto& [fragment_op, extras_map] : fragment_to_extras) {
      llvm::SmallDenseMap<std::pair<int64_t, int64_t>, int64_t>
          extra_result_indices;
      // Look up the return-index fixup map *before* rebuilding, because
      // RebuildFragmentWithExtraResults erases fragment_op.
      auto& return_idx_map = fragment_extra_return_indices[fragment_op];
      FragmentOp new_fragment = RebuildFragmentWithExtraResults(
          fragment_op, extras_map, extra_result_indices, builder);

      // Fix up the return op operands.
      for (auto& [inner_idx, ret_indices] : return_idx_map) {
        for (int64_t copy = 0; copy < static_cast<int64_t>(ret_indices.size());
             ++copy) {
          int64_t new_result_idx = extra_result_indices[{inner_idx, copy}];
          return_op->setOperand(ret_indices[copy],
                                new_fragment->getResult(new_result_idx));
        }
      }
    }

    // Step 3: Process block arguments - merge into existing fragments.
    // First, find all existing fragments per mesh.
    llvm::MapVector<StringRef, SmallVector<FragmentOp>> mesh_to_fragments;
    func_op.walk([&](FragmentOp fragment) {
      mesh_to_fragments[fragment.getMeshName()].push_back(fragment);
    });

    // Use a counter for passthrough IDs to avoid collisions.
    int64_t passthrough_id_counter = -1;

    for (auto& merge_info : block_arg_merges) {
      auto *it = mesh_to_fragments.find(merge_info.mesh_name);
      if (it != mesh_to_fragments.end() && !it->second.empty()) {
        // Merge into the first available fragment on this mesh.
        FragmentOp target_fragment = it->second.front();
        llvm::SmallDenseMap<std::pair<int64_t, int64_t>, int64_t>
            extra_result_indices;

        auto [new_fragment, first_result_idx] = RebuildFragmentWithPassthrough(
            target_fragment, merge_info.block_arg,
            merge_info.return_indices.size(), extra_result_indices,
            passthrough_id_counter, builder);

        // Update the mesh_to_fragments map so subsequent block args on the
        // same mesh target the updated fragment.
        it->second.front() = new_fragment;

        // Fix up return op operands.
        for (int64_t i = 0;
             i < static_cast<int64_t>(merge_info.return_indices.size()); ++i) {
          int64_t new_result_idx =
              extra_result_indices[{passthrough_id_counter, i}];
          return_op->setOperand(merge_info.return_indices[i],
                                new_fragment->getResult(new_result_idx));
        }
        --passthrough_id_counter;
      } else {
        // No existing fragment on this mesh. Fallback: create a new fragment.
        CreateFallbackFragmentForBlockArg(
            merge_info.mesh_name, merge_info.block_arg,
            merge_info.return_indices, return_op, builder);
      }
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
