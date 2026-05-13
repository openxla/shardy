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
#include "llvm/ADT/STLExtras.h"
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

bool CanMoveAfter(Operation* op_to_move, Operation* target_op) {
  if (op_to_move->getBlock() != target_op->getBlock()) return false;
  if (!op_to_move->isBeforeInBlock(target_op)) return false;

  Operation* current = op_to_move->getNextNode();
  while (current) {
    for (Value result : op_to_move->getResults()) {
      if (llvm::is_contained(current->getOperands(), result)) {
        return false;
      }
    }
    for (Value operand : op_to_move->getOperands()) {
      if (operand.getDefiningOp() == current) {
        return false;
      }
    }

    if (current == target_op) break;
    current = current->getNextNode();
  }
  return true;
}

void CreateReturnFragmentForMesh(StringRef mesh_name, Operation* return_op,
                                 ValueToReturnIndices& value_to_return_indices,
                                 OpBuilder& builder) {
  // We remove any entries that require no work, in order to avoid too many
  // checks.
  value_to_return_indices.remove_if([](const auto& it) {
    if (it.second.size() == 1) {
      return !isa<BlockArgument>(it.first);
    }
    return false;
  });

  if (value_to_return_indices.empty()) {
    return;
  }

  builder.setInsertionPoint(return_op);
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

  // Try to find an existing same-mesh fragment to merge the newly created
  // inferred fragment into. We track two things:
  //   - latest_operand_producer: the latest op that produces any operand of
  //     the inferred fragment (needed for positioning constraints).
  //   - merge_target: the latest same-mesh fragment we can merge into.
  FragmentOp merge_target = nullptr;
  Operation* latest_operand_producer = nullptr;

  // Updates merge_target if `op` is a same-mesh fragment that appears later
  // in the block than the current candidate.
  auto updateMergeTarget = [&](Operation* op) {
    auto frag = dyn_cast<FragmentOp>(op);
    if (frag && frag.getMeshName() == mesh_name &&
        (!merge_target || merge_target->isBeforeInBlock(frag))) {
      merge_target = frag;
    }
  };

  // First, scan the operand producers for a merge candidate.
  for (Value v : fragment_operands) {
    Operation* op = v.getDefiningOp();
    if (!op) continue;
    if (!latest_operand_producer ||
        latest_operand_producer->isBeforeInBlock(op)) {
      latest_operand_producer = op;
    }
    updateMergeTarget(op);
  }

  // If no producer fragment on the same mesh was found among the operands,
  // look for any fragment on the same mesh in the block (sideways merge).
  // Only do this when there are actual producer ops (not just block arguments).
  if (!merge_target && latest_operand_producer) {
    for (Operation& op : *return_op->getBlock()) {
      if (&op == return_op || &op == fragment_op) continue;
      updateMergeTarget(&op);
    }
  }

  // Give up if no merge candidate was found, or if the candidate can't be
  // moved after the latest operand producer (which would break dominance).
  if (!merge_target || (merge_target != latest_operand_producer &&
                        !CanMoveAfter(merge_target, latest_operand_producer))) {
    return;
  }

  // Position the merge target after the latest operand producer so all
  // operands of the inferred fragment are available, then merge.
  if (merge_target != latest_operand_producer) {
    merge_target->moveAfter(latest_operand_producer);
  }

  fragment_op->moveAfter(merge_target);
  IRRewriter rewriter(builder.getContext());
  MergeRegionOps(
      merge_target, fragment_op, rewriter,
      /*num_static_args=*/0, /*replace_producer_use_in_consumer_block=*/
      [](OpOperand&, Value) {
        SDY_CHECK(false) << "Fragment ops shouldn't have free variables";
      },
      GetFragmentOriginUnion(merge_target, fragment_op, rewriter),
      merge_target.getMeshNameAttr(),
      /*stage_id=*/merge_target.getStageIdAttr());
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
    // value_to_return_indices_per_mesh[mesh_name] = value_to_return_indices
    // where value_to_return_indices[v] contains a sequence of the indices in
    // return op where v is used.
    llvm::MapVector<StringRef, ValueToReturnIndices>
        value_to_return_indices_per_mesh;
    for (OpOperand& operand : return_op->getOpOperands()) {
      auto mesh_type = dyn_cast<MeshTensorType>(operand.get().getType());
      SDY_CHECK(mesh_type);
      StringRef mesh_name = mesh_type.getMeshName();
      value_to_return_indices_per_mesh[mesh_name][operand.get()].push_back(
          operand.getOperandNumber());
    }

    OpBuilder builder(&getContext());
    for (auto& [mesh_name, value_to_return_indices] :
         value_to_return_indices_per_mesh) {
      CreateReturnFragmentForMesh(mesh_name, return_op, value_to_return_indices,
                                  builder);
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
