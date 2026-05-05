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

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_WRAPBLOCKARGRETURNSPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

using ValueToReturnIndices = llvm::MapVector<Value, SmallVector<int64_t>>;

// Creates an identity fragment for block arguments returned by the function
// on a given mesh. Each block argument gets a single fragment input and one
// result per return position it appears in.
void WrapBlockArgsForMesh(StringRef mesh_name, Operation* return_op,
                          ValueToReturnIndices& value_to_return_indices,
                          OpBuilder& builder) {
  if (value_to_return_indices.empty()) {
    return;
  }

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
  fragment_op->setAttr(
      kInferredByAttr,
      builder.getArrayAttr({builder.getStringAttr("wrap_block_arg_returns")}));
  Block& fragment_block = fragment_op.getRegion().emplaceBlock();

  SmallVector<Value> returned_values;
  returned_values.reserve(fragment_return_types.size());
  int fragment_result_index = 0;
  sdy::MeshAttr mesh_attr = GetMeshOrFail(fragment_op, mesh_name);
  for (const auto& [value, return_indices] : value_to_return_indices) {
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
}

class WrapBlockArgReturnsPass
    : public impl::WrapBlockArgReturnsPassBase<WrapBlockArgReturnsPass> {
  using WrapBlockArgReturnsPassBase::WrapBlockArgReturnsPassBase;

 protected:
  void runOnFunc(func::FuncOp func_op) override {
    if (!IsMpmdFunction(func_op)) {
      return;
    }

    Operation* return_op = func_op.getBody().front().getTerminator();
    llvm::MapVector<StringRef, ValueToReturnIndices>
        value_to_return_indices_per_mesh;
    for (OpOperand& operand : return_op->getOpOperands()) {
      if (!isa<BlockArgument>(operand.get())) continue;
      auto mesh_type = dyn_cast<MeshTensorType>(operand.get().getType());
      SDY_CHECK(mesh_type);
      StringRef mesh_name = mesh_type.getMeshName();
      value_to_return_indices_per_mesh[mesh_name][operand.get()].push_back(
          operand.getOperandNumber());
    }

    OpBuilder builder(&getContext());
    for (auto& [mesh_name, value_to_return_indices] :
         value_to_return_indices_per_mesh) {
      WrapBlockArgsForMesh(mesh_name, return_op, value_to_return_indices,
                           builder);
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
