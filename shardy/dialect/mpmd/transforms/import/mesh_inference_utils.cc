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

#include "shardy/dialect/mpmd/transforms/import/mesh_inference_utils.h"

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_arg_res_attrs.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/import/meshes_with_origins.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir::mpmd {

using ::mlir::func::FuncOp;

bool IsMeshlessOp(Operation* op) {
  return op && !op->hasTrait<OpTrait::IsTerminator>() &&
         !op->getParentOfType<FragmentOp>() &&
         !sdy::inDialect<MpmdDialect>(op) && !isa<ModuleOp, FuncOp>(op);
}

MeshesWithOrigins GetUseSet(MeshesWithOriginsAttr origins) {
  return MeshesWithOrigins::CreateUseSet(origins);
}

MeshesWithOrigins GetUseSet(Operation* op) {
  if (!op) {
    return MeshesWithOrigins::CreateUseSet({});
  }
  return GetUseSet(op->getAttrOfType<MeshesWithOriginsAttr>(kMpmdUseSet));
}

MeshesWithOrigins GetArgUseSet(FuncOp func, int arg) {
  return GetUseSet(
      func.getArgAttrOfType<MeshesWithOriginsAttr>(arg, kMpmdUseSet));
}

MeshesWithOrigins GetResUseSet(FuncOp func, int res) {
  return GetUseSet(
      func.getResultAttrOfType<MeshesWithOriginsAttr>(res, kMpmdUseSet));
}

MeshesWithOrigins GetUseSet(ForOp for_op, int arg_number) {
  return GetUseSet(dyn_cast_if_present<MeshesWithOriginsAttr>(
      GetArgAttr(for_op, arg_number, kMpmdUseSet)));
}

SetVector<StringRef> GetUseMeshes(Operation* op) {
  return GetUseSet(op).MeshNamesOrEmpty();
}

void SetUseSet(Operation* op, MeshesWithOrigins use_set, OpBuilder& builder) {
  if (!use_set.empty()) {
    op->setAttr(kMpmdUseSet, use_set.ToAttr(builder));
  }
}

void SetArgUseSet(FuncOp func, int arg_number, MeshesWithOrigins use_set,
                  OpBuilder& builder) {
  if (!use_set.empty()) {
    func.setArgAttr(arg_number, kMpmdUseSet, use_set.ToAttr(builder));
  }
}

void SetResUseSet(FuncOp func, int res_number, MeshesWithOrigins use_set,
                  OpBuilder& builder) {
  if (!use_set.empty()) {
    func.setResultAttr(res_number, kMpmdUseSet, use_set.ToAttr(builder));
  }
}

namespace {
// ForOp has one arg attr per region arg (not per operand).
SmallVector<Attribute> GetArgAttrsOrCreateDefault(ForOp op) {
  return op->hasAttr(kArgAttrName)
             ? llvm::to_vector(
                   op->getAttrOfType<ArrayAttr>(kArgAttrName).getValue())
             : SmallVector<Attribute>(
                   op.getRegion().getNumArguments(),
                   DictionaryAttr::get(op->getContext(), {}));
}
}  // namespace

void SetUseSet(ForOp for_op, int arg_number, MeshesWithOrigins use_set,
               OpBuilder& builder) {
  if (!use_set.empty()) {
    auto arg_attrs = GetArgAttrsOrCreateDefault(for_op);
    InsertAttr(arg_attrs[arg_number], kMpmdUseSet,
                           use_set.ToAttr(builder));
    SetArgAttrs(for_op, arg_attrs);
  }
}

// If the given `result` is the target of a mesh data flow edge, returns the
// source of that edge, otherwise returns a nullptr.
//
// When an operand of an op flows into the result of the same or a parent op (in
// case the op is a region terminator), and both ops can accept either local or
// mesh tensors, we define a mesh data flow edge between the operand and the
// result.
OpOperand* GetMeshDataFlowSrc(OpResult result) {
  if (!result) {
    return nullptr;
  }
  Operation* op = result.getOwner();

  if (auto call_op = dyn_cast<CallOp>(op)) {
    return &GetCalleeFunc(call_op).front().getTerminator()->getOpOperand(
        result.getResultNumber());
  }
  if (auto for_op = dyn_cast<ForOp>(op)) {
    return &for_op.getRegion().front().getTerminator()->getOpOperand(
        result.getResultNumber());
  }

  return nullptr;
}

MeshesWithOrigins GetSrcSet(MeshesWithOriginsAttr origins) {
  return MeshesWithOrigins(origins);
}

MeshesWithOrigins GetSrcSet(Operation* op, StringRef set_attr_name) {
  if (!op) {
    return MeshesWithOrigins();
  }
  return GetSrcSet(op->getAttrOfType<MeshesWithOriginsAttr>(set_attr_name));
}

MeshesWithOrigins GetSrcSet(Operation* op) {
  return GetSrcSet(op, kMpmdSrcSet);
}

MeshesWithOrigins GetSrcSet(FuncOp func, int arg_number) {
  return GetSrcSet(dyn_cast_if_present<MeshesWithOriginsAttr>(
      func.getArgAttr(arg_number, kMpmdSrcSet)));
}

MeshesWithOrigins GetSrcSet(ForOp for_op, int arg_number) {
  return GetSrcSet(dyn_cast_if_present<MeshesWithOriginsAttr>(
      GetArgAttr(for_op, arg_number, kMpmdSrcSet)));
}

MeshesWithOrigins GetSrcSet(OpOperand& op_operand) {
  Value operand = op_operand.get();
  // Pass through data flow ops.
  while (auto target = dyn_cast<OpResult>(operand)) {
    OpOperand* source = GetMeshDataFlowSrc(target);
    if (!source) {
      break;
    }
    operand = source->get();
  }

  if (Operation* defining_op = operand.getDefiningOp()) {
    if (isa<AssignOp>(defining_op)) {
      return MeshesWithOrigins();
    }
    return GetSrcSet(defining_op);
  }

  Operation* defining_op =
      cast<BlockArgument>(operand).getOwner()->getParentOp();
  if (auto func_op = dyn_cast<FuncOp>(defining_op)) {
    return GetSrcSet(func_op,
                     cast<BlockArgument>(operand).getArgNumber());
  }
  if (auto for_op = dyn_cast<ForOp>(defining_op)) {
    return GetSrcSet(for_op, cast<BlockArgument>(operand).getArgNumber());
  }

  return MeshesWithOrigins();
}

std::optional<SetVector<StringRef>> GetSrcMeshes(Operation* op) {
  return GetSrcSet(op).MaybeMeshNames();
}

std::optional<SetVector<StringRef>> GetSrcMeshes(FuncOp func, int arg_number) {
  return GetSrcSet(func, arg_number).MaybeMeshNames();
}

std::optional<SetVector<StringRef>> GetSrcMeshes(OpOperand& op_operand) {
  return GetSrcSet(op_operand).MaybeMeshNames();
}

void SetSrcSet(Operation* op, MeshesWithOrigins src_set, OpBuilder& builder) {
  op->setAttr(kMpmdSrcSet, src_set.ToAttr(builder));
}

void SetSrcSet(FuncOp func, int arg_number, MeshesWithOrigins src_set,
               OpBuilder& builder) {
  func.setArgAttr(arg_number, kMpmdSrcSet, src_set.ToAttr(builder));
}

void SetSrcSet(ForOp for_op, int arg_number, MeshesWithOrigins src_set,
               OpBuilder& builder) {
  auto arg_attrs = GetArgAttrsOrCreateDefault(for_op);
  InsertAttr(arg_attrs[arg_number], kMpmdSrcSet,
                         src_set.ToAttr(builder));
  SetArgAttrs(for_op, arg_attrs);
}

bool ClearUseSet(Operation* op) {
  return op->removeAttr(kMpmdUseSet) != nullptr;
}
bool ClearUseSet(FuncOp func) {
  bool removed_something = false;
  for (BlockArgument arg : func.getArguments()) {
    removed_something |=
        func.removeArgAttr(arg.getArgNumber(), kMpmdUseSet) != nullptr;
  }
  for (OpResult res : func->getResults()) {
    removed_something |=
        func.removeResultAttr(res.getResultNumber(), kMpmdUseSet) != nullptr;
  }

  return removed_something;
}

bool ClearUseSetAndSrcSet(Operation* op) {
  bool removed_something = false;
  removed_something |= ClearUseSet(op);
  removed_something |= op->removeAttr(kMpmdSrcSet) != nullptr;
  return removed_something;
}

bool ClearUseSetAndSrcSet(FuncOp func) {
  bool removed_something = false;
  for (BlockArgument arg : func.getArguments()) {
    removed_something |=
        func.removeArgAttr(arg.getArgNumber(), kMpmdUseSet) != nullptr;
    removed_something |=
        func.removeArgAttr(arg.getArgNumber(), kMpmdSrcSet) != nullptr;
  }

  for (int res_number = 0; res_number < func.getNumResults(); ++res_number) {
    removed_something |=
        func.removeResultAttr(res_number, kMpmdUseSet) != nullptr;
    removed_something |=
        func.removeResultAttr(res_number, kMpmdSrcSet) != nullptr;
  }

  return removed_something;
}

bool ClearUseSetAndSrcSet(ForOp for_op) {
  bool removed_something = false;
  if (for_op->hasAttr(kArgAttrName)) {
    auto arg_attrs =
        for_op->getAttrOfType<ArrayAttr>(kArgAttrName).getValue();
    SmallVector<Attribute> new_arg_attrs;
    for (auto arg_attr : arg_attrs) {
      removed_something |= RemoveAttr(arg_attr, kMpmdUseSet);
      removed_something |= RemoveAttr(arg_attr, kMpmdSrcSet);
      if (!cast<DictionaryAttr>(arg_attr).empty()) {
        new_arg_attrs.push_back(arg_attr);
      }
    }
    if (new_arg_attrs.empty()) {
      for_op->removeAttr(kArgAttrName);
    } else {
      SetArgAttrs(for_op, new_arg_attrs);
    }
  }
  return removed_something;
}

void ClearCanConvertAttr(FuncOp func) {
  func->walk([](Operation* op) { op->removeAttr(kCanConvertToReduce); });
}

namespace {

// Copies the meshes from `use` into `base_set`. If `use` is in `source_block`,
// then we copy it directly from `use`. If it's in a nested-block (e.g. used in
// a while-loop as a free variable), then we copy it over from the parent op in
// the same block.
void CopyMeshesFromUse(Block* source_block, OpOperand& use,
                       MeshesWithOrigins& base_set) {
  Operation* user = use.getOwner();

  // TODO: b/341882915 - If we canonicalize chains of broadcasts, we might not
  // need to special case the broadcast producers.
  if (IsTerminalNodeInAnalysis(user) &&
      !isa_and_present<BroadcastOp>(use.get().getDefiningOp())) {
    return;
  }

  if (isa<UnassignOp>(user)) {
    return;
  }

  if (Operation* ancestor = GetAncestorInBlock(source_block, user);
      ancestor != user) {
    base_set.Union(GetUseSet(ancestor));
    return;
  }

  if (auto call_op = dyn_cast<CallOp>(use.getOwner())) {
    base_set.Union(
        GetArgUseSet(GetCalleeFunc(call_op), use.getOperandNumber()));
    return;
  }

  if (auto func_return = dyn_cast<func::ReturnOp>(use.getOwner())) {
    auto func = cast<FuncOp>(func_return->getParentOp());
    if (!IsEntryPointFunction(func)) {
      base_set.Union(GetResUseSet(func, use.getOperandNumber()));
      return;
    }
  }

  if (auto for_op = dyn_cast<ForOp>(use.getOwner())) {
    base_set.Union(GetUseSet(for_op, use.getOperandNumber()));
    return;
  }

  base_set.Union(GetUseSet(user));
}

}  // namespace

void UpdateTransitiveUses(Operation* op, MeshesWithOrigins& base_set) {
  for (OpOperand& use : op->getUses()) {
    CopyMeshesFromUse(op->getBlock(), use, base_set);
  }
}

void UpdateTransitiveUses(Value value, MeshesWithOrigins& base_set) {
  for (OpOperand& use : value.getUses()) {
    CopyMeshesFromUse(value.getParentBlock(), use, base_set);
  }
}

bool IsTerminalNodeInAnalysis(Operation* op) {
  return isa<BroadcastOp, ReduceOp>(op);
}

bool IsCallOpInCallChain(CallOp call_op) {
  return llvm::any_of(call_op->getUsers(), [&](Operation* user) {
    if (auto user_call_op = dyn_cast<CallOp>(user)) {
      return user_call_op.getCallee() == call_op.getCallee();
    }
    return false;
  });
}

bool CallOpHasUseSetPopulated(CallOp call_op) {
  FuncOp callee_func = GetCalleeFunc(call_op);
  return llvm::any_of(callee_func.getArguments(), [&](BlockArgument arg) {
    return callee_func.getArgAttr(arg.getArgNumber(), kMpmdUseSet);
  });
}

}  // namespace mlir::mpmd
