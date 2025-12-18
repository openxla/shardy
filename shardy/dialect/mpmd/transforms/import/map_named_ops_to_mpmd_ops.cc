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
#include <optional>
#include <string_view>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_assignment_map.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_MAPNAMEDOPSTOMPMDOPSPASS
#define GEN_PASS_DEF_INLINENESTEDUSEREXPOSEDOPSPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

void InlineNamedComputation(NamedComputationOp op, RewriterBase& rewriter) {
  Block* block = op.getBody();
  auto terminator = cast<ReturnOp>(block->getTerminator());
  rewriter.inlineBlockBefore(block, op, op->getOperands());
  rewriter.replaceOp(op, terminator->getOperands());
  rewriter.eraseOp(terminator);
}

// Converts a named_computation to a fragment assigning it to a mesh (the first
// value of `computation_assignment`) and to a stage (the second value of
// `computation_assignment`) if defined.
void MapNamedComputationToMesh(
    NamedComputationOp named_computation_op,
    const MeshStageAssignment& computation_assignment, IRRewriter& rewriter) {
  rewriter.setInsertionPoint(named_computation_op);
  std::pair<StringRef, std::optional<StringRef>> mesh_name_and_memory_kind =
      TryToExtractMemoryKindFromMeshName(computation_assignment.first);
  StringRef mesh_name = mesh_name_and_memory_kind.first;
  if (mesh_name_and_memory_kind.second.has_value()) {
    // TODO: b/374994155 - Support memory_kind assignment in named computations.
    SDY_LOG(WARNING) << "Named computation "
                     << std::string_view(named_computation_op.getName())
                     << " has memory_kind="
                     << std::string_view(
                            mesh_name_and_memory_kind.second.value())
                     << ", which will be ignored by Shardy:MPMD.";
  }

  sdy::MeshAttr mesh = GetMeshOrFail(named_computation_op, mesh_name);

  // Create AssignOps for each operand of the named_computation. These will be
  // the operands of the new FragmentOp.
  SmallVector<Value> new_operands;
  Location named_computation_loc = named_computation_op.getLoc();

  IRMapping operand_to_assigned;
  for (Value operand : named_computation_op.getOperands()) {
    if (!operand_to_assigned.contains(operand)) {
      auto assign_op = AssignOp::create(
          rewriter, named_computation_loc, operand, mesh_name, mesh,
          /*origin=*/named_computation_op.getName());
      operand_to_assigned.map(operand, assign_op.getResult());
    }
    // Reuse assign ops for repeated operands.
    new_operands.push_back(operand_to_assigned.lookup(operand));
  }

  SmallVector<Type> result_types;
  for (Value result : named_computation_op.getResults()) {
    result_types.push_back(MeshTensorType::getFullyReplicated(
        rewriter.getContext(), mesh_name, mesh,
        cast<RankedTensorType>(result.getType())));
  }

  auto user_origin_attrs = ArrayAttr::get(
      rewriter.getContext(), {named_computation_op.getOriginAttr()});

  // Create a new fragment op replacing the named_computation op.
  IntegerAttr stage_id = IntegerAttr();
  if (computation_assignment.second.has_value()) {
    stage_id = rewriter.getI64IntegerAttr(*computation_assignment.second);
    SDY_CHECK(!user_origin_attrs.empty())
        << "Inferred fragments cannot be assigned to stages.";
  }
  FragmentOp new_fragment =
      FragmentOp::create(rewriter, named_computation_loc, result_types,
                         new_operands, user_origin_attrs, mesh_name, stage_id);
  new_fragment.getRegion().takeBody(named_computation_op.getRegion());

  // Create UnassignOps for each result of the new fragment.
  SmallVector<Value> new_results;
  for (Value new_result : new_fragment.getResults()) {
    new_results.push_back(
        UnassignOp::create(rewriter, named_computation_loc, new_result,
                           /*origin=*/named_computation_op.getName())
            .getResult());
  }
  rewriter.replaceOp(named_computation_op, new_results);
}

std::optional<MeshStageAssignment> GetMeshStageAssignment(
    NamedComputationOp op, const UserAssignmentMap& assignment_map) {
  // We assume that `assignment_` contains a valid mapping between
  // named_computations and mesh names, i.e., each key in the mapping
  // is the name of an actual named_computation in the function being
  // partitioned and each mesh name is part of the topology.
  auto assignment_it = assignment_map.find(std::string_view(op.getName()));
  if (assignment_it == assignment_map.end()) {
    return std::nullopt;
  }
  return assignment_it->second;
}

std::optional<MeshTensorType> GetMeshTensorTypeFromAssignment(
    NamedTensorOp op, const UserAssignmentMap& assignment_map) {
  auto assignment_it = assignment_map.find(std::string_view(op.getName()));
  if (assignment_it == assignment_map.end()) {
    return std::nullopt;
  }
  // We ignore the stage assignment of any named tensor.
  // StringRef mesh_name = assignment_it->second.first;
  std::pair<StringRef, std::optional<StringRef>> mesh_name_and_memory_kind =
      TryToExtractMemoryKindFromMeshName(assignment_it->second.first);
  StringRef mesh_name_without_memory_kind = mesh_name_and_memory_kind.first;
  StringAttr memory_kind = {};
  if (mesh_name_and_memory_kind.second.has_value()) {
    memory_kind = StringAttr::get(op.getContext(),
                                  mesh_name_and_memory_kind.second.value());
  }

  std::optional<int64_t> stage_id = assignment_it->second.second;
  if (stage_id.has_value()) {
    SDY_VLOG(2) << "Named tensor " << std::string_view(op.getName())
                << " was assigned to stage " << *stage_id
                << " but this will be ignored by Shardy:MPMD.";
  }
  return MeshTensorType::get(op.getContext(), mesh_name_without_memory_kind,
                             cast<RankedTensorType>(op.getTensor().getType()),
                             memory_kind);
}

// Returns whether it succeeded in converting the named_tensor.
bool MapNamedTensorToUnassignOfAssign(NamedTensorOp named_tensor_op,
                                      IRRewriter& rewriter,
                                      const UserAssignmentMap& assignment) {
  rewriter.setInsertionPoint(named_tensor_op);
  std::optional<MeshTensorType> mesh_tensor =
      GetMeshTensorTypeFromAssignment(named_tensor_op, assignment);
  if (mesh_tensor.has_value()) {
    auto assign_op = AssignOp::create(rewriter, named_tensor_op.getLoc(),
                                      *mesh_tensor, named_tensor_op.getTensor(),
                                      /*origin=*/named_tensor_op.getNameAttr());
    rewriter.replaceOpWithNewOp<UnassignOp>(
        named_tensor_op, assign_op.getResult(),
        /*origin=*/named_tensor_op.getNameAttr());
  } else {
    // No references to this name in `assignment_`, thus it's unused, so
    // remove it.
    rewriter.replaceOp(named_tensor_op, named_tensor_op.getOperand());
  }
  return true;
}

class InlineNestedUserExposedOpsPass
    : public impl::InlineNestedUserExposedOpsPassBase<
          InlineNestedUserExposedOpsPass> {
  using InlineNestedUserExposedOpsPassBase::
      InlineNestedUserExposedOpsPassBase;

 private:
  void runOnFunc(func::FuncOp func_op) override {
    IRRewriter rewriter(func_op.getContext());
    bool pass_must_signal_failure = false;

    // 1. Inline any named_computation, named tensor, broadcast and reduce ops
    // that is nested in a named_computation, checking that its mesh assignment
    // matches that of the parent.
    func_op.getBody().walk([&](Operation* op) {
      auto parent = op->getParentOfType<NamedComputationOp>();
      if (!parent) {
        return WalkResult::advance();
      }
      std::optional<MeshStageAssignment> parent_mesh_assignment =
          GetMeshStageAssignment(parent, assignment.value);
      if (auto named_computation = dyn_cast<NamedComputationOp>(op)) {
        std::optional<MeshStageAssignment> op_assignment =
            GetMeshStageAssignment(named_computation, assignment.value);
        if (op_assignment.has_value() &&
            op_assignment != parent_mesh_assignment) {
          named_computation.emitError("NamedComputation '")
              << named_computation.getName()
              << "' is nested in a NamedComputation '" << parent.getName()
              << "' which has a different mesh or stage assignment.";
          pass_must_signal_failure = true;
        };
        rewriter.setInsertionPoint(named_computation);
        InlineNamedComputation(named_computation, rewriter);
        return WalkResult::advance();
      }
      if (auto named_tensor = dyn_cast<NamedTensorOp>(op)) {
        std::optional<MeshTensorType> mesh_tensor =
            GetMeshTensorTypeFromAssignment(named_tensor, assignment.value);
        if (mesh_tensor.has_value() && mesh_tensor->getMemoryKind()) {
          SDY_LOG(WARNING) << "Named tensor "
                           << std::string_view(named_tensor.getName())
                           << " has memory_kind="
                           << std::string_view(
                                  mesh_tensor->getMemoryKind().strref())
                           << ", which will be ignored by Shardy:MPMD has it's "
                              "nested in a NamedComputation.";
        }
        if (mesh_tensor.has_value() &&
            mesh_tensor->getMeshName() != parent_mesh_assignment->first) {
          named_tensor.emitError("NamedTensor '")
              << named_tensor.getName() << "' is nested in a NamedComputation '"
              << parent.getName() << "' which has a different mesh assignment.";
          pass_must_signal_failure = true;
        }
      }
      if (isa<NamedTensorOp, BroadcastOp, ReduceOp>(op)) {
        rewriter.replaceAllUsesWith(op->getResult(0), op->getOperand(0));
        rewriter.eraseOp(op);
      }
      return WalkResult::advance();
    });

    if (pass_must_signal_failure) {
      return signalPassFailure();
    }
  }
};

class MapNamedOpsToMpmdOpsPass
    : public impl::MapNamedOpsToMpmdOpsPassBase<MapNamedOpsToMpmdOpsPass> {
  using MapNamedOpsToMpmdOpsPassBase::MapNamedOpsToMpmdOpsPassBase;

 private:
  void runOnFunc(func::FuncOp func_op) override {
    IRRewriter rewriter(func_op.getContext());
    bool pass_must_signal_failure = false;

    func_op.getBody().walk([&](Operation* op) {
      if (auto named_computation = dyn_cast<NamedComputationOp>(op)) {
        if (std::optional<MeshStageAssignment> op_assignment =
                GetMeshStageAssignment(named_computation, assignment.value)) {
          MapNamedComputationToMesh(named_computation, *op_assignment,
                                    rewriter);
        } else {
          named_computation.emitError("Top-level NamedComputation '")
              << named_computation.getName()
              << "' is not assigned to a mesh in the user-defined "
                 "named-to-mesh "
                 "assignment.";
          pass_must_signal_failure = true;
        }
        // No need to visit the body of the named_computation.
        return WalkResult::skip();
      }

      if (auto named_tensor = dyn_cast<NamedTensorOp>(op)) {
        if (!MapNamedTensorToUnassignOfAssign(named_tensor, rewriter,
                                              assignment.value)) {
          pass_must_signal_failure = true;
        }
      }
      return WalkResult::advance();
    });

    if (pass_must_signal_failure) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::mpmd
