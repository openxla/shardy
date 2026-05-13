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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_UNIQUIFYFUNCTIONINPUTSOUTPUTSPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

using ValueToReturnIndices = llvm::MapVector<Value, SmallVector<int64_t>>;

// Creates an identity FragmentOp. The caller sets the insertion point.
// Returns nullptr if the map is empty.
FragmentOp CreateIdentityFragment(StringRef mesh_name, Operation* return_op,
                                  ValueToReturnIndices& value_to_return_indices,
                                  OpBuilder& builder) {
  if (value_to_return_indices.empty()) {
    return nullptr;
  }

  SmallVector<Value> fragment_operands;
  fragment_operands.reserve(value_to_return_indices.size());
  SmallVector<Type> fragment_return_types;
  for (const auto& [value, return_indices] : value_to_return_indices) {
    fragment_operands.push_back(value);
    fragment_return_types.insert(fragment_return_types.end(),
                                 return_indices.size(),
                                 cast<MeshTensorType>(value.getType()));
  }

  auto loc = return_op->getLoc();
  auto fragment_op = FragmentOp::create(
      builder, loc, fragment_return_types, fragment_operands,
      /*user_origin=*/ArrayAttr::get(builder.getContext(), {}),
      /*mesh_name=*/mesh_name, /*stage_id=*/IntegerAttr());
  SetInferredByAttr(fragment_op, "uniquify", builder);
  Block& fragment_block = fragment_op.getRegion().emplaceBlock();

  SmallVector<Value> returned_values;
  returned_values.reserve(fragment_return_types.size());
  // The index of the fragment result that we should use to replace the
  // function return op operand.
  int fragment_result_index = 0;
  sdy::MeshAttr mesh_attr = GetMeshOrFail(fragment_op, mesh_name);
  for (const auto& [value, return_indices] : value_to_return_indices) {
    // Add a single block argument for this value and return it as many times
    // as it's used.
    returned_values.insert(
        returned_values.end(), return_indices.size(),
        fragment_block.addArgument(
            GetGlobalTensorTypeFromMeshType(value, mesh_attr), value.getLoc()));

    for (int64_t index : return_indices) {
      return_op->setOperand(index,
                            fragment_op->getResult(fragment_result_index++));
    }
  }
  auto block_builder = OpBuilder::atBlockEnd(&fragment_block);
  ReturnOp::create(block_builder, loc, returned_values);
  return fragment_op;
}

// Creates a uniquify fragment and merges it into the original using
// MergeRegionOps.
void CreateAndMergeUniquifyFragment(
    FragmentOp fragment, Operation* func_return_op,
    ValueToReturnIndices& value_to_return_indices, OpBuilder& builder) {
  // Insert right after the fragment to preserve dominance.
  builder.setInsertionPointAfter(fragment);
  FragmentOp uniquify_fragment = CreateIdentityFragment(
      fragment.getMeshName(), func_return_op, value_to_return_indices, builder);
  if (!uniquify_fragment) {
    return;
  }

  // Immediately merge the fragment into the uniquify fragment.
  IRRewriter rewriter(fragment.getContext());
  MergeRegionOps(
      fragment, uniquify_fragment, rewriter,
      /*num_static_args=*/0,
      /*replace_producer_use_in_consumer_block=*/
      [](OpOperand&, Value) {
        SDY_CHECK(false) << "Fragment ops shouldn't have free variables";
      },
      GetFragmentOriginUnion(fragment, uniquify_fragment, rewriter),
      fragment.getMeshNameAttr(),
      /*stage_id=*/fragment.getStageIdAttr());
}

// Replaces the return values of the function with transfer ops.
// If we have
// func.func @func(%arg0: !mesh_1_tensor) ->
// (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) {
//  return %arg0, %arg0, %arg0 : !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
// }
// Then we would introduce 3 transfers:
// func.func @func(%arg0: !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor,
// !mesh_1_tensor) {
//  %0 = transfer(%arg0) : !mesh_1_tensor -> !mesh_1_tensor
//  %1 = transfer(%arg0) : !mesh_1_tensor -> !mesh_1_tensor
//  %2 = transfer(%arg0) : !mesh_1_tensor -> !mesh_1_tensor
//  return %0, %1, %2 : !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
// }
// This is needed to ensure that each returned value can hold a different
// sharding. If the same value is returned multiple times, each result position
// could have a different sharding requirement. By inserting a TransferOp, we
// create a new SSA value for each result position, allowing different
// shardings to be applied.
// We always insert a TransferOp for block arguments to ensure that sharding
// constraints can be applied to an op result rather than a block argument.
// For other values, if they are used in more than one return position, as an
// optimization, we only insert a TransferOp for all but the first return
// position.
void uniquifyReturnsWithTransferOps(func::FuncOp func_op,
                                    MLIRContext* context) {
  Operation* return_op = func_op.getBody().front().getTerminator();
  llvm::SmallDenseSet<Value> seen_values;
  OpBuilder builder(context);
  builder.setInsertionPoint(return_op);
  for (OpOperand& operand : return_op->getOpOperands()) {
    if (!seen_values.contains(operand.get())) {
      seen_values.insert(operand.get());
      if (!isa<BlockArgument>(operand.get())) {
        continue;
      }
    }
    auto transfer_op = TransferOp::create(
        builder, return_op->getLoc(),
        cast<MeshTensorType>(operand.get().getType()), operand.get());
    operand.set(transfer_op->getResult(0));
  }
}

class UniquifyFunctionInputOutputsPass
    : public impl::UniquifyFunctionInputsOutputsPassBase<
          UniquifyFunctionInputOutputsPass> {
  using UniquifyFunctionInputsOutputsPassBase::
      UniquifyFunctionInputsOutputsPassBase;

 protected:
  void runOnFunc(func::FuncOp func_op) override {
    if (!IsMpmdFunction(func_op)) {
      // This is not the main function. Do nothing.
      return;
    }

    if (this->useTransferInsteadOfFragment) {
      uniquifyReturnsWithTransferOps(func_op, &getContext());
      return;
    }

    Operation* return_op = func_op.getBody().front().getTerminator();
    OpBuilder builder(&getContext());

    // Linear scan of return operands.
    // - For FragmentOp results: collect duplicate return positions per
    // fragment.
    // - For block arguments: collect into per-mesh map.
    llvm::SmallDenseSet<Value> seen_values;
    llvm::MapVector<FragmentOp, ValueToReturnIndices> fragments_to_uniquify;
    llvm::MapVector<StringRef, ValueToReturnIndices> block_args_per_mesh;

    for (OpOperand& operand : return_op->getOpOperands()) {
      Value v = operand.get();
      bool first_occurrence = seen_values.insert(v).second;

      // A value needs uniquification if:
      // - It's a block argument (even first occurrence, since sharding
      //   constraints need an op result).
      // - It's a duplicate (any occurrence after the first).
      bool is_block_arg = isa<BlockArgument>(v);
      if (first_occurrence && !is_block_arg) {
        continue;
      }

      if (auto fragment = dyn_cast_or_null<FragmentOp>(v.getDefiningOp())) {
        // Value comes from a FragmentOp. Record duplicate return position.
        fragments_to_uniquify[fragment][v].push_back(
            operand.getOperandNumber());
      } else {
        // Block argument (or non-fragment op result). Collect per mesh.
        auto mesh_type = cast<MeshTensorType>(v.getType());
        block_args_per_mesh[mesh_type.getMeshName()][v].push_back(
            operand.getOperandNumber());
      }
    }

    // Uniquify fragment returns by merging in identity fragments.
    for (auto& [fragment, value_to_return_indices] : fragments_to_uniquify) {
      CreateAndMergeUniquifyFragment(fragment, return_op,
                                     value_to_return_indices, builder);
    }

    // Create identity fragments for block arguments.
    builder.setInsertionPoint(return_op);
    for (auto& [mesh_name, value_to_return_indices] : block_args_per_mesh) {
      CreateIdentityFragment(mesh_name, return_op, value_to_return_indices,
                             builder);
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
