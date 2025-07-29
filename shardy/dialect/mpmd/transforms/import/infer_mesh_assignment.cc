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

#include "shardy/dialect/mpmd/transforms/import/infer_mesh_assignment.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_inference_origins.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_inference_utils.h"
#include "shardy/dialect/mpmd/transforms/import/meshes_with_origins.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/import/sharding_constraints.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_INFERMESHPOPULATEUSESETPASS
#define GEN_PASS_DEF_INFERMESHPOPULATESRCSETPASS
#define GEN_PASS_DEF_INFERMESHASSIGNUSINGINPUTOUTPUTCONSTRAINTSPASS
#define GEN_PASS_DEF_INFERMESHASSIGNMESHFORFUNCLEAVESPASS
#define GEN_PASS_DEF_INFERMESHCONVERTREDUCEOPSPASS
#define GEN_PASS_DEF_INFERMESHREWRITEUSINGANALYSISPASS
#define GEN_PASS_DEF_INFERMESHFINALIZEPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

using ::llvm::DenseMap;
using ::mlir::func::FuncOp;

// The maximum number of iterations for use-set propagation of call-op chains.
//
// This is to prevent infinite loops in case of bugs. Arbitrarily chosen.
constexpr int kMaxCallChainUseSetIterations = 10;

// Returns success if either the mesh names of `assign_op` and `unassign_op`
// match and `should_mesh_names_match` is true, or they don't match and
// `should_mesh_names_match` is false.
//
// It's enough to check whether the mesh names match, since both assign and
// unassign require their mesh tensor to be fully replicated.
LogicalResult MatchIntraMeshAssignOfUnassign(AssignOp assign_op,
                                             UnassignOp unassign_op,
                                             PatternRewriter& rewriter,
                                             bool should_mesh_names_match) {
  StringRef unassign_mesh_name =
      unassign_op.getTensor().getType().getMeshName();
  StringRef assign_mesh_name = assign_op.getResult().getType().getMeshName();
  bool mesh_names_match = unassign_mesh_name == assign_mesh_name;
  if (mesh_names_match != should_mesh_names_match) {
    return rewriter.notifyMatchFailure(assign_op, [&](Diagnostic& diag) {
      diag << "Expected the mesh name of UnassignOp to "
           << (should_mesh_names_match ? "" : "not ")
           << "match that of the AssignOp: \"" << unassign_mesh_name
           << "\" != \"" << assign_mesh_name << "\"";
    });
  }

  return success();
}

// Verifies that all of the following constraints hold:
//
// - All the inputs and outputs of `func_op` are mesh tensors.
// - All non-mpmd ops are nested within fragments.
// - There are no assign or unassign ops.
//
// Note that we call emitError on the op that breaks the constraint, which will
// be useful for debugging.
//
// The errors here should be caught earlier by validation, but this acts as a
// sanity check in case something slips through.
LogicalResult VerifyMeshAssignment(FuncOp func_op) {
  FunctionType func_type = func_op.getFunctionType();

  bool has_error = false;
  for (auto [index, input_type] : llvm::enumerate(func_type.getInputs())) {
    if (!isa<MeshTensorType>(input_type)) {
      func_op.emitError("function input ")
          << index << " is not a MeshTensorType, but a " << input_type;
      has_error = true;
    }
  }
  for (auto [index, output_type] : llvm::enumerate(func_type.getResults())) {
    if (!isa<MeshTensorType>(output_type)) {
      func_op.emitError("function output ")
          << index << " is not a MeshTensorType, but a " << output_type;
      has_error = true;
    }
  }

  func_op.walk([&has_error](Operation* op) {
    if (auto fragment = dyn_cast<FragmentOp>(op)) {
      // Skip fragment ops and their regions for efficiency, as they are already
      // assigned.
      return WalkResult::skip();
    }
    if (isa<AssignOp, UnassignOp, BroadcastOp>(op)) {
      has_error = true;
      op->emitError(
          "assigns, unassigns or broadcasts are not allowed after mesh "
          "inference.");
    }
    if (IsMeshlessOp(op)) {
      has_error = true;
      op->emitError("no more meshless ops are expected at this point.");
    }

    if (ClearUseSetAndSrcSet(op)) {
      SDY_LOG(WARNING) << "Use set or src set still present on op "
                       << std::string_view(op->getName().getStringRef())
                       << " after mesh inference. This shouldn't happen.";
    }

    return WalkResult::advance();
  });

  return failure(has_error);
}

// Eliminates no-op assign of unassign when the mesh tensors of both are the
// same.
//
// In symbols:
//   assign (unassign %v m) m ~> %v
//
// Note that the unassign can have additional users, in which case it won't be
// erased following this rewrite and will trigger other patterns.
class AssignOfUnassignSameMeshPattern final
    : public OpRewritePattern<AssignOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AssignOp op,
                                PatternRewriter& rewriter) const override {
    auto unassign_op = op.getTensor().getDefiningOp<UnassignOp>();
    if (!unassign_op) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "Expected the operand of the AssignOp to be the result of an "
                "UnassignOp";
      });
    }

    if (failed(
            MatchIntraMeshAssignOfUnassign(op, unassign_op, rewriter,
                                           /*should_mesh_names_match=*/true))) {
      return failure();
    }

    rewriter.replaceOp(op, unassign_op.getTensor());

    return success();
  }
};

// Replaces an assign of unassign between different meshes, where the operand of
// the unassign is also used by a transfer with the same return type as the
// assign and in the same or an enclosing block, with the result of that
// transfer.
//
// If the transfer is after the assign in the same block, or after the ancestor
// of the assign in the transfer's block, we move the transfer before it, so
// that all users of the assign are after the transfer.
//
// In symbols:
//
//   Y = unassign<M1>(X)
//   Z = assign<M2>(Y)
//   op(Z)
//   W = transfer<M1->M2>(X)
//   other_user(Y)
//   ~>
//   Y = unassign<M1>(X)
//   W = transfer<M1->M2>(X)
//   op(W)
//   other_user(Y)
//
// Note that the unassign can have additional users, in which case it won't be
// erased following this rewrite and will trigger other patterns.
class DedupAssignOfUnassignAndTransferPattern final
    : public OpRewritePattern<AssignOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AssignOp op,
                                PatternRewriter& rewriter) const override {
    auto unassign_op = op.getTensor().getDefiningOp<UnassignOp>();
    if (!unassign_op) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "Expected the operand of the AssignOp the result of an "
                "UnassignOp";
      });
    }

    if (failed(MatchIntraMeshAssignOfUnassign(
            op, unassign_op, rewriter,
            /*should_mesh_names_match=*/false))) {
      return failure();
    }

    auto users = unassign_op.getTensor().getUsers();
    auto transfer_op_it = llvm::find_if(users, [&](Operation* user) {
      auto transfer_op = dyn_cast<TransferOp>(user);
      return transfer_op && transfer_op.getType() == op.getType() &&
             HasAncestorInBlock(transfer_op->getBlock(), op);
    });

    if (transfer_op_it == users.end()) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "Expected the operand of the UnassignOp to be used by a "
             << "TransferOp with the same return type as the AssignOp and in "
             << "the same or an enclosing block";
      });
    }

    auto transfer_op = cast<TransferOp>(*transfer_op_it);

    // If the transfer is after the assign, we need to move the transfer before
    // it, so that all uses of the assign are after the transfer.
    Operation* ancestor_in_block =
        GetAncestorInBlock(transfer_op->getBlock(), op);
    if (ancestor_in_block->isBeforeInBlock(transfer_op)) {
      transfer_op->moveBefore(ancestor_in_block);
    }

    rewriter.replaceOp(op, transfer_op.getResult());

    return success();
  }
};

// Replaces an assign of unassign between different meshes, where the operand of
// the unassign is a block argument of a func op, with an inter-mesh
// transfer.
//
// In symbols:
//
// func.func f(arg0: mesh_tensor<m1>) {
//   x = unassign arg0
//   y = assign x -> m2
//   ...
// }
//
// ~~>
//
// func.func f(arg0: mesh_tensor<m1>) {
//   y = transfer arg0   m1 -> m2
//   ...
// }
//
// Note that the unassign can have additional users, in which case it won't be
// erased following this rewrite and will trigger other patterns.
class AssignOfUnassignFuncArgPattern final : public OpRewritePattern<AssignOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AssignOp op,
                                PatternRewriter& rewriter) const override {
    auto unassign_op = op.getTensor().getDefiningOp<UnassignOp>();
    if (!unassign_op) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "Expected the operand of the AssignOp to be the result of an "
                "UnassignOp";
      });
    }

    auto block_arg = dyn_cast<BlockArgument>(unassign_op.getTensor());
    if (!block_arg || !isa<FuncOp>(block_arg.getOwner()->getParentOp()) ||
        !IsEntryPointFunction(unassign_op->getParentOfType<FuncOp>())) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "Expected the operand of the unassign_op to be an argument of "
             << "the entry point FuncOp";
      });
    }

    if (failed(MatchIntraMeshAssignOfUnassign(
            op, unassign_op, rewriter,
            /*should_mesh_names_match=*/false))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TransferOp>(op, op.getType(),
                                            unassign_op.getTensor());

    return success();
  }
};

// This pattern replaces assign(unassign(%v, m1), m2) ~~> transfer(%v, m1->m2).
class AssignOfUnassignPattern final : public OpRewritePattern<AssignOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AssignOp op,
                                PatternRewriter& rewriter) const override {
    auto unassign_op = op.getTensor().getDefiningOp<UnassignOp>();
    if (!unassign_op) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "Expected the operand of the AssignOp to be the result of an "
                "UnassignOp";
      });
    }

    StringRef unassign_mesh_name =
        unassign_op.getTensor().getType().getMeshName();
    StringRef assign_mesh_name = op.getType().getMeshName();
    if (unassign_mesh_name == assign_mesh_name) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "Expected the mesh name of UnassignOp to be different from the "
                "AssignOp but got \""
             << unassign_mesh_name << "\" == \"" << assign_mesh_name << "\"";
      });
    }

    auto transfer = rewriter.replaceOpWithNewOp<TransferOp>(
        op, op.getType(), unassign_op.getTensor());
    // TODO: b/329221688 - Improve these logs to make it easier to debug.
    SDY_VLOG(2) << "Created cross-mesh transfer "
                << PrintOperationForLog(transfer);
    return success();
  }
};

FragmentOp CreateReduceFragment(ArrayRef<Value> mesh_tensors,
                                StringRef mesh_name,
                                ReductionType reduction_type,
                                RewriterBase& rewriter) {
  return FragmentOp::createMeshFragmentWithGlobalBody(
      mesh_tensors.front().getLoc(), /*user_origin=*/{}, mesh_name,
      mesh_tensors, mesh_tensors.front().getType(), rewriter,
      [reduction_type](ArrayRef<Value> args, OpBuilder& block_builder) {
        Value accumulator = args.front();
        for (Value val : llvm::drop_begin(args)) {
          accumulator =
              CreateStablehloReduceOp(reduction_type, {accumulator, val},
                                      val.getLoc(), block_builder)
                  ->getResult(0);
        }
        return SmallVector<Value>({accumulator});
      });
}

// This pattern lowers mpmd.reduce to reductions and transfers.
// In symbols:
//
// %v = mpmd.reduce<R>(%v0,... %vn) use_set=m3
//
// ~~>
//
// %v_m1 = R(%v0, R(...)) # on mesh1
// %v_m2 = R(...) # on mesh2
// %v = R(transfer(%v_m1), transfer(%v_m2)) # on mesh3
class LowerMpmdReducePattern final : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp mpmd_reduce,
                                PatternRewriter& rewriter) const override {
    rewriter.setInsertionPointAfter(mpmd_reduce);

    // Group the mesh tensor operands by mesh.
    DenseMap<StringRef, SmallVector<Value>> mesh_name_to_mesh_tensor_operands;
    for (auto operand : mpmd_reduce.getTensors()) {
      auto unassign = operand.getDefiningOp<UnassignOp>();
      SDY_CHECK(unassign);
      mesh_name_to_mesh_tensor_operands
          [unassign.getTensor().getType().getMeshName()]
              .push_back(unassign.getTensor());
    }

    // Create local reductions in each mesh, and keep track of the results.
    SmallVector<Value> local_reductions;
    local_reductions.reserve(mesh_name_to_mesh_tensor_operands.size());
    for (const auto& [mesh_name, mesh_tensors] :
         mesh_name_to_mesh_tensor_operands) {
      if (mesh_tensors.size() == 1) {
        // Nothing to reduce.
        local_reductions.push_back(mesh_tensors.front());
      } else {
        local_reductions.push_back(
            CreateReduceFragment(mesh_tensors, mesh_name,
                                 mpmd_reduce.getReductionType(), rewriter)
                .getResult(0));
      }
    }

    // Group the assign users of the mpmd.reduce op by mesh.
    DenseMap<StringRef, SmallVector<AssignOp>> mesh_name_to_assign_users;
    for (Operation* user : mpmd_reduce->getUsers()) {
      auto assign = dyn_cast<AssignOp>(user);
      SDY_CHECK(assign);
      mesh_name_to_assign_users[assign.getType().getMeshName()].push_back(
          assign);
    }

    // For each destination mesh, transfer all the local reduce results to that
    // mesh, and do another local reduce on that destination mesh.
    for (StringRef user_mesh : GetUseMeshes(mpmd_reduce)) {
      SmallVector<AssignOp> assign_users =
          mesh_name_to_assign_users.lookup(user_mesh);
      SDY_CHECK(!assign_users.empty())
          << "Each mesh in the use_set should "
             "correspond to at least one assign user.";
      MeshTensorType user_type = assign_users.front().getType();
      SmallVector<Value> transferred_intermediates;
      for (Value reduced_val : local_reductions) {
        if (reduced_val.getType() == user_type) {
          transferred_intermediates.push_back(reduced_val);
        } else {
          transferred_intermediates.push_back(rewriter.create<TransferOp>(
              reduced_val.getLoc(), user_type, reduced_val));
        }
      }

      FragmentOp target_mesh_reduce =
          CreateReduceFragment(transferred_intermediates, user_mesh,
                               mpmd_reduce.getReductionType(), rewriter);
      for (AssignOp user : assign_users) {
        rewriter.replaceAllUsesWith(user, target_mesh_reduce.getResult(0));
      }
    }

    return success();
  }
};

WalkResult PopulateUseSet(Operation* op, OpBuilder& builder);

// Populates the use_set of the callee func of the given call_op.
// Returns true if the use_set of the callee func has changed.
bool PopulateUseSetForCalleeFunc(CallOp call_op, FuncOp callee_func,
                                 OpBuilder& builder) {
  bool has_changed = false;
  // Propagate through to the callee result.
  for (OpResult call_result : call_op->getResults()) {
    MeshesWithOrigins use_set =
        GetResUseSet(callee_func, call_result.getResultNumber());

    int original_num_uses = use_set.size();
    UpdateTransitiveUses(call_result, use_set);
    if (use_set.size() != original_num_uses) {
      has_changed = true;
      SetResUseSet(callee_func, call_result.getResultNumber(), use_set,
                   builder);
    }
  }

  // We only need to populate the use_set of the callee func again if the
  // use_set of the results have changed, or if we've not populated it yet.
  // Otherwise, the use_set of the body will be unchanged.
  if (has_changed || !CallOpHasUseSetPopulated(call_op)) {
    // Populate the func's use_set.
    callee_func.walk<WalkOrder::PostOrder, ReverseIterator>(
        [&builder](Operation* op) { return PopulateUseSet(op, builder); });
    return true;
  }
  return false;
}

// This populates the use_set of a CallOp and its callee func. We populate the
// CallOp as if it were a region op, since the behaviour of mesh inference
// should be the same whether or not the CallOp is inlined. Treating it as a
// region op gives similar behaviour as inlining: we populate the
// use_set of the callee func whenever we see the CallOp. The main difference is
// that the ops in the callee func are repeatedly processed (instead of
// processing different ops). This can cause the inference to give problematic
// results and we validate this in a separate pass. E.g.
//
// x1 = ...
// c1 = call @f(x1)
// assign c1 -> m1
//
// x2 = ...
// c2 = call @f(x2)
// assign c2 -> m2
//
// func f (%arg0) {
//   %add = stablehlo.add %arg0, %arg0
//   return %add
// }
//
// So we need to propagate through each time we see the call to get the correct
// use_set {m1,m2} for %add and also for %x1. But note that %x2 will have
// use_set {m2} only. This means the use_set for %x2 is actually invalid since
// it is used in an add which has use_set {m1,m2}. We allow this here but will
// raise an error in a later validation pass, as we don't allow edges out of a
// CallOp to have multiple entries in the use_set – in such a case, there's no
// sensible mesh assignment for the func arg or return value: we don't allow
// setting the signature of f to accept tensors from both m1 and m2.
//
// If the call op is in a call chain, we keep propagating the use_set
// until a fixed point, as that will be required for the mesh assignments to be
// valid. Note that this is a little bit like the ForOp.
void PopulateUseSetForCallOp(CallOp call_op, OpBuilder& builder) {
  FuncOp callee_func = GetCalleeFunc(call_op);

  bool has_changed = PopulateUseSetForCalleeFunc(call_op, callee_func, builder);

  // If the call op is in a call chain, we need to keep propagating the use_set
  // until a fixed point.
  if (IsCallOpInCallChain(call_op)) {
    for (int i = 0; i < kMaxCallChainUseSetIterations && has_changed; ++i) {
      has_changed = PopulateUseSetForCalleeFunc(call_op, callee_func, builder);
    }
  }
}

// This populates the use_set of a ForOp.
void PopulateUseSetForForOp(ForOp for_op, OpBuilder& builder) {
  for (BlockArgument arg : for_op.getRegion().getArguments()) {
    MeshesWithOrigins use_set = GetUseSet(for_op, arg.getArgNumber());
    UpdateTransitiveUses(arg, use_set);
    SetUseSet(for_op, arg.getArgNumber(), use_set, builder);
  }
}

// Populates the use set of the defining op of op_result inside the call_op
// callee func.
void UpdateUseSetForCallOpUsingOpResult(CallOp call_op, OpResult op_result,
                                        OpBuilder& builder) {
  FuncOp callee_func = GetCalleeFunc(call_op);
  MeshesWithOrigins use_set =
      GetResUseSet(callee_func, op_result.getResultNumber());
  UpdateTransitiveUses(op_result, use_set);
  SetResUseSet(callee_func, op_result.getResultNumber(), use_set, builder);
}

// This propagates the use_set from the return op through to the body of the
// matching ForOp.
void PopulateUseSetForForOpTerminator(ReturnOp return_op, OpBuilder& builder) {
  // TODO(petebu): Attach use-sets to ForOp res attr to avoid multi-step
  // propagation.
  if (auto for_op = dyn_cast<ForOp>(return_op->getParentOp())) {
    for (auto [for_result, for_return_operand] :
         llvm::zip(for_op->getResults(), return_op->getOperands())) {
      if (Operation* defining_op = for_return_operand.getDefiningOp()) {
        if (auto call_op = dyn_cast<CallOp>(defining_op)) {
          UpdateUseSetForCallOpUsingOpResult(call_op, for_result, builder);
        } else {
          MeshesWithOrigins use_set = GetUseSet(defining_op);
          UpdateTransitiveUses(for_result, use_set);
          SetUseSet(defining_op, use_set, builder);
        }
      }
    }
  }
}

// This populates the use_set of an op or func arg. If it is an AssignOp, it
// initializes it. Otherwise, if it is an op needing assignment, it generates
// the use_set from its users. Func args are a special case where having the
// use_set helps for further analysis.
//
// Pre-condition: all users of `op` must have their use_sets populated.
//
// Hence, it must be used in conjunction with a post-order traversal of the
// MLIR graph, so that all users are processed before the current op.
WalkResult PopulateUseSet(Operation* op, OpBuilder& builder) {
  if (auto assign_op = dyn_cast<AssignOp>(op)) {
    SetUseSet(assign_op, MeshesWithOrigins(assign_op.getMeshWithOrigin()),
              builder);
  } else if (auto func = dyn_cast<FuncOp>(op)) {
    for (BlockArgument arg : func.getArguments()) {
      MeshesWithOrigins use_set = GetArgUseSet(func, arg.getArgNumber());
      UpdateTransitiveUses(arg, use_set);
      SetArgUseSet(func, arg.getArgNumber(), use_set, builder);
    }
  } else if (auto fragment = dyn_cast<FragmentOp>(op)) {
    // Skip fragment ops and their regions for efficiency,
    // as they are already assigned.
    return WalkResult::skip();
  } else if (auto call_op = dyn_cast<CallOp>(op)) {
    PopulateUseSetForCallOp(call_op, builder);
  } else if (auto for_op = dyn_cast<ForOp>(op)) {
    PopulateUseSetForForOp(for_op, builder);
  } else if (auto return_op = dyn_cast<ReturnOp>(op)) {
    PopulateUseSetForForOpTerminator(return_op, builder);
  } else if (IsMeshlessOp(op) || IsTerminalNodeInAnalysis(op) ||
             isa<UnassignOp>(op)) {
    MeshesWithOrigins use_set = GetUseSet(op);
    UpdateTransitiveUses(op, use_set);
    SetUseSet(op, use_set, builder);
  }

  return WalkResult::advance();
}

class InferMeshPopulateUseSetPass
    : public impl::InferMeshPopulateUseSetPassBase<
          InferMeshPopulateUseSetPass> {
  using InferMeshPopulateUseSetPassBase::InferMeshPopulateUseSetPassBase;

  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    OpBuilder builder(&getContext());

    for (FuncOp func_op : GetMpmdFunctions(module_op)) {
      if (IsEntryPointFunction(func_op)) {
        // Do a post-order traversal.
        func_op.walk<WalkOrder::PostOrder, ReverseIterator>(
            [&builder](Operation* op) { return PopulateUseSet(op, builder); });
      }
    }
  }
};

// If `op` is the concat of a concat-reduce pair, then check that:
// - the concat and reduction are on the same dim,
// - the concat is on a dim of size 1 for all operands, and
// - the concat is only used by the reduce.
//
// Returns the matched ReduceOp.
stablehlo::ReduceOp MatchCrossMeshConcatReduce(Operation* op) {
  if (!llvm::all_equal(op->getOperandTypes())) {
    return nullptr;
  }
  auto concat = dyn_cast<stablehlo::ConcatenateOp>(op);
  stablehlo::ReduceOp reduce_user =
      op->hasOneUse() ? dyn_cast<stablehlo::ReduceOp>(*op->getUsers().begin())
                      : nullptr;

  if (concat && reduce_user && reduce_user.getInputs().size() == 1 &&
      reduce_user.getDimensions().size() == 1) {
    auto concat_operand_type =
        dyn_cast<RankedTensorType>(op->getOperandTypes().front());
    int64_t reduce_dim = reduce_user.getDimensions().front();
    if (concat_operand_type && reduce_dim == concat.getDimension() &&
        concat_operand_type.getDimSize(reduce_dim) == 1) {
      return reduce_user;
    }
  }
  return nullptr;
}

class InferMeshPopulateSrcSetPass
    : public impl::InferMeshPopulateSrcSetPassBase<
          InferMeshPopulateSrcSetPass> {
  using InferMeshPopulateSrcSetPassBase::InferMeshPopulateSrcSetPassBase;

  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    OpBuilder builder(&getContext());

    for (FuncOp func_op : GetMpmdFunctions(module_op)) {
      if (IsEntryPointFunction(func_op)) {
        func_op.walk<WalkOrder::PreOrder>(
            [&](Operation* op) { return PopulateSrcSet(op, builder); });
      }
    }
  }

 private:
  // Copies the use_set to the src_set, updating the origin to inferred_in,
  // rather than copying the old one (e.g. "layer0"). We don't want to copy over
  // the origins for the use_set, we want new origins to indicate that the
  // meshes are inferred.
  void AddUseSetToSrcSet(MLIRContext* context, MeshesWithOrigins& src_set,
                         const MeshesWithOrigins& use_set) {
    for (StringRef mesh_name : use_set.MeshNamesOrEmpty()) {
      src_set.insert(MeshWithOriginsAttr::get(
          context, mesh_name, OriginAttr::get(context, kInferredInputOrigin)));
    }
  }

  // Initializes the src_set to the origin mesh of the UnassignOp and the
  // destination meshes of TransferOp users of the UnassignOp operand.
  //
  // E.g. if we have
  // x = fragment "m1" {...}
  // y = unassign x: mesh_tensor<m1, ...>
  // z = transfer x "m2" {...}
  //
  // Then y has src-set {"m1", "m2"}.
  // And these are the only meshes that y can exist on:
  //    - "m1" is ok because we can replace uses of "y" with "x"
  //    - "m2" is ok because we can replace uses of "y" with "z"
  //    - any other mesh is invalid: say that y is assigned to another mesh
  //    "m3",
  //      and x has no other transfer users, then a new transfer to m3 needs to
  //      be created. But this is not allowed because we do not allow creation
  //      of transfers (see src_set definition above).
  //
  // Note that the transfers must be direct users of x, i.e.
  // `z = transfer x "m"`. Indirect transfers of x do not work because uses of x
  // are either TransferOps, UnassignOps, or other ops. TransferOps and
  // UnassignOps are discussed above. For ops acting on x (e.g. a FragmentOp),
  // the result will be a different tensor (assuming it's non-trivial) and hence
  // we cannot dedup into that op (without introducing logic around the inverse
  // of ops).
  //
  // Note that x is a mesh tensor, so its users also cannot be an AssignOp nor
  // can it be a meshless op. It could also be a FragmentOp but that is
  // addressed above.
  //
  // We register the UnassignOp mesh first, as we prefer that mesh.
  // E.g. in the following program, it is most natural to assign z to mesh m1.
  //
  // x = frag m1 {...}
  // u = unassign(x)
  // t = transfer(x) m1 -> m2
  // y = frag(t) m2 {...}
  // z = u + u
  void InitializeSrcSet(UnassignOp op, OpBuilder& builder) {
    MeshesWithOrigins src_set;
    src_set.insert(op.getMeshWithOrigin());
    for (Operation* user : op.getTensor().getUsers()) {
      if (auto transfer = dyn_cast<TransferOp>(user)) {
        src_set.insert(TransferMeshWithOrigin(transfer));
      }
    }

    // When the unassign is on an entrypoint func arg, then we treat it like a
    // func arg: i.e. we add all its user meshes (the use_set) to the src_set
    // and allow transfers to be created on it.
    if (isa<BlockArgument>(op.getTensor()) &&
        IsEntryPointFunction(op->getParentOfType<FuncOp>())) {
      MeshesWithOrigins use_set;
      UpdateTransitiveUses(op.getResult(), use_set);
      AddUseSetToSrcSet(op->getContext(), src_set, use_set);
    }

    SetSrcSet(op, src_set, builder);
  }

  // Initializes the src_set of a func arg to its use_set. This initialization
  // restricts the src_set to a subset of all meshes. It leaves args with empty
  // use_set untouched. We only want to restrict the mesh assignments of args
  // that have explicit uses in a mesh.
  //
  // Pre-condition: use_set is populated.
  //
  // If func arg is used in mesh m1, then it will exist in mesh m1 and so the
  // func arg does exist there and is part of the src_set. This is special logic
  // for the func args, because we allow introduction of transfers on func args
  // according to the meshes they are used in.
  void InitializeSrcSet(FuncOp func, OpBuilder& builder) {
    for (BlockArgument arg : func.getArguments()) {
      // use_sets aren't populated for block args, so we compute it manually.
      MeshesWithOrigins use_set = GetArgUseSet(func, arg.getArgNumber());
      if (!use_set.empty()) {
        MeshesWithOrigins src_set;
        AddUseSetToSrcSet(func->getContext(), src_set, use_set);
        SetSrcSet(func, arg.getArgNumber(), src_set, builder);
      }
    }
  }

  // Infers reduce ops and propagates src_set as a special case: the src_set of
  // a reduce_op is the union of its operands, rather than the intersection.
  //
  // The inferring algorithm is split in two phases: annotating, and rewriting.
  // This is to preserve our analysis-only phase of mesh inference for easier
  // debugging.
  //
  // We infer a reduce op if:
  // - the intersection of operand src_sets is empty OR one of the operands is a
  // reduce.
  // - the op is a standard element-wise reduction: e.g. add, mul, etc., or is a
  // concat-reduce pair. See `ParseCrossMeshConcatReduce` for details.
  // - all the operands have non-empty src_set.
  //
  // The src_set of the reduce op is the union of its operand src_sets, if the
  // operand src_set is present (i.e. we ignore operands that could exist on any
  // mesh). Such operands with missing src_sets are usually a special case which
  // should not affect the src_set of the reduce.
  //
  // Pre-condition: `op` already has its src_set populated in the usual way.
  //
  // TODO: b/343174113 - support custom reductions.
  // TODO: b/340565987 - consider allowing src_set of reductions to be the whole
  // set, and tracking the srcs as a new annotation `reduce_set` if need be.
  void InferReduceAndPropagateSrcSet(Operation* op, OpBuilder& builder) {
    stablehlo::ReduceOp cross_mesh_reduce = MatchCrossMeshConcatReduce(op);
    std::optional<ReductionType> reduction_type =
        cross_mesh_reduce
            ? ComputeReductionType(cross_mesh_reduce.getBody().front())
            : GetReductionOpType(op);
    if (!reduction_type) {
      return;
    }
    auto reduction_attr = ReductionAttr::get(op->getContext(), *reduction_type);

    MeshesWithOrigins all_srcs;
    bool some_operand_is_reduce = false;
    for (OpOperand& operand : op->getOpOperands()) {
      MeshesWithOrigins operand_src_set = GetSrcSet(operand);
      if (operand_src_set.empty()) {
        return;
      }

      all_srcs.Union(operand_src_set);

      if (Operation* operand_op = operand.get().getDefiningOp();
          operand_op &&
          operand_op->getAttr(kMpmdReduceAnnotation) == reduction_attr) {
        some_operand_is_reduce = true;
      }
    }

    if (GetSrcSet(op).empty() || some_operand_is_reduce) {
      op->setAttr(kMpmdReduceAnnotation, reduction_attr);
      if (cross_mesh_reduce) {
        cross_mesh_reduce->setAttr(kMpmdReduceAnnotation, reduction_attr);
      }
      SetSrcSet(op, all_srcs, builder);
    }
  }

  // Propagates src_sets forward from operands to the op itself, taking the
  // intersection of the operand src_sets.
  //
  // Pre-condition: all operands of `op` must have their src_sets populated.
  //
  // This is indeed the src_set of the op, since the op and its operands (and
  // results) must live on the same mesh.
  //
  // E.g. if we have
  // x = op(w, s)
  //
  // where w has src_set {m1, m2} and s has src_set {m2, m3} then x has src_set
  // {m2}. This is because `w,s,x` and the op must live on the same mesh
  // (because we don't allow transfers to be created on intermediate values) and
  // the only mesh they can all live on is m2.
  //
  // We prove by contradiction that x cannot live on m1. If x were to live on
  // m1, then the op would live on m1 and so s (or a copy of s) must live on m1.
  // But that's not possible since m1 not in src_set(s), so x cannot live on m1.
  // And similarly, x cannot live on m3. So x has src_set {m2}.
  void PropagateSrcSet(Operation* op, OpBuilder& builder) {
    MeshesWithOrigins src_set = GetSrcSet(op);
    for (OpOperand& operand : op->getOpOperands()) {
      src_set.Intersect(GetSrcSet(operand));
    }

    if (src_set) {
      SetSrcSet(op, src_set, builder);

      InferReduceAndPropagateSrcSet(op, builder);
    }
  }

  // This populates the src_set of a CallOp and its callee func. We populate the
  // CallOp as if it were a region op, since the behaviour of mesh inference
  // should be the same whether or not the CallOp is inlined. Treating it as a
  // region op gives similar behaviour as inlining: we populate the
  // src_set of the callee func whenever we see the CallOp. The main difference
  // is that the ops in the callee func are repeatedly processed (instead of
  // processing different ops). This can cause the inference to give problematic
  // results and we validate this in a separate pass. See the tests and
  // the docs of `PopulateUseSetForCallOp` for an example of this.
  void PopulateSrcSetForCallOp(CallOp call_op, OpBuilder& builder) {
    FuncOp callee_func = GetCalleeFunc(call_op);
    // Propagate through to the callee func body.
    for (OpOperand& call_operand : call_op->getOpOperands()) {
      MeshesWithOrigins src_set = GetSrcSet(call_operand);
      src_set.Intersect(
          GetSrcSet(callee_func, call_operand.getOperandNumber()));
      if (src_set) {
        SetSrcSet(callee_func, call_operand.getOperandNumber(), src_set,
                  builder);
      }
    }

    // Populate the func's src_set.
    // Walk only the body, not the func itself. We don't want to set the func
    // args' src_set with the use_set, since the src_set should come from
    // call operands.
    callee_func.getBody().walk(
        [&](Operation* op) { return PopulateSrcSet(op, builder); });
  }

  // This populates the src_set of a ForOp.
  void PopulateSrcSetForForOp(ForOp for_op, OpBuilder& builder) {
    for (OpOperand& for_operand : for_op.getOperation()->getOpOperands()) {
      MeshesWithOrigins src_set = GetSrcSet(for_operand);
      if (src_set) {
        SetSrcSet(for_op, for_operand.getOperandNumber(), src_set, builder);
      }
    }
  }

  // This populates the src_set of an op. See the docs of the individual
  // function calls for details on initialization and propagation.
  //
  // Pre-condition: all operands of `op` must have their src_sets populated.
  //
  // Hence, it must be used in conjunction with a pre-order traversal of the
  // MLIR graph, so that all users are processed before the current op.
  //
  // TODO: b/340565987 - move to class so that we can access `infer_reductions`
  // as a field instead of as an arg.
  WalkResult PopulateSrcSet(Operation* op, OpBuilder& builder) {
    if (auto unassign = dyn_cast<UnassignOp>(op)) {
      InitializeSrcSet(unassign, builder);
    } else if (auto func = dyn_cast<FuncOp>(op)) {
      InitializeSrcSet(func, builder);
    } else if (auto fragment = dyn_cast<FragmentOp>(op)) {
      // Skip fragment ops and their regions for efficiency, as they are already
      // assigned.
      return WalkResult::skip();
    } else if (auto call_op = dyn_cast<CallOp>(op)) {
      PopulateSrcSetForCallOp(call_op, builder);
    } else if (auto for_op = dyn_cast<ForOp>(op)) {
      PopulateSrcSetForForOp(for_op, builder);
    } else if (IsMeshlessOp(op) || isa<AssignOp>(op)) {
      PropagateSrcSet(op, builder);
      // Skip regions of these meshless ops (if they have a region). These
      // regions will have the same mesh as the op itself.
      return WalkResult::skip();
    }

    return WalkResult::advance();
  }
};

// Prints a set of meshes {m1,m2,m3} as "m1,m2,m3".
std::string GetPrintableString(MeshesWithOrigins meshes) {
  return llvm::join(meshes.MeshNames().getArrayRef(), ",");
}

// Returns a mesh for assignment that is valid: i.e. which is in the operand's
// src_set. It tries to pick from `preferred_meshes` where possible, and if not,
// it picks the first mesh in the operand's src_set.
std::optional<StringRef> GetMeshForAssignment(
    OpOperand& operand, StringRef first_mesh_name,
    const MeshesWithOrigins& preferred_meshes) {
  MeshesWithOrigins src_set = GetSrcSet(operand);
  if (src_set.empty()) {
    return std::nullopt;
  }

  if (!src_set) {
    return preferred_meshes.GetPrioritizedMeshName().value_or(first_mesh_name);
  }

  return *src_set.GetPrioritizedMeshName(preferred_meshes.MeshNamesOrEmpty());
}

// Returns the mesh for the main function's return operand which is compatible
// with the `input_use_set`, preferring to pick a mesh from the output's use_set
// over the src_set. Returns std::nullopt if there is no such mesh. We assume
// here that `input_use_set` is the use_set of the input, where the pair (input,
// output) is specified by the user to be constrained to the same mesh.
//
// If the input or output is already a MeshTensor (because of user-provided
// assignment), then we try to use the mesh it is assigned to. If the output is
// assigned, we can always assign the input (because the input can be assigned
// anywhere). If the input is assigned, we may not be able to assign the output
// without introducing a transfer. In this case, we leave it to the
// `EnforceEquishardingPass` pass to introduce this transfer.
//
// If the input or output is already a MeshTensor (because of user-provided
// assignment), then we try to use the mesh it is assigned to. If the output is
// assigned, we can always assign the input (because the input can be assigned
// anywhere). If the input is assigned, we may not be able to assign the output
// without introducing a transfer. In this case, we leave it to the
// `EnforceEquishardingPass` pass to introduce this transfer.
std::optional<StringRef> GetMeshForInputOutputAssignment(
    OpOperand& output_operand, FuncOp func, int64_t input_index) {
  auto output_mesh_type =
      dyn_cast<MeshTensorType>(output_operand.get().getType());
  BlockArgument input_arg = func.getArgument(input_index);
  auto input_mesh_type = dyn_cast<MeshTensorType>(input_arg.getType());

  if (output_mesh_type) {
    if (input_mesh_type) {
      SDY_CHECK_EQ(std::string_view(input_mesh_type.getMeshName()),
                   std::string_view(output_mesh_type.getMeshName()))
          << "Constraint was given to map input " << input_index
          << " to output " << output_operand.getOperandNumber()
          << " but the user provided invalid meshes differ. This should be "
             "caught in python validation.";
    }
    return output_mesh_type.getMeshName();
  }

  MeshesWithOrigins output_src_set = GetSrcSet(output_operand);
  // Returns an empty set if the result is an argument.
  MeshesWithOrigins output_use_set =
      GetUseSet(output_operand.get().getDefiningOp());

  MeshesWithOrigins input_use_set;
  if (input_mesh_type) {
    input_use_set.insert(MeshWithOriginsAttr::get(
        func->getContext(), input_mesh_type.getMeshName()));
  } else {
    input_use_set = GetArgUseSet(func, input_index);
    if (input_use_set.empty()) {
      // This means the input is not used in any mesh. In these cases, the
      // algorithm assigns it to any mesh, but we will have to guarantee that
      // it's assigned to the same mesh as the output. We can do that without
      // restricting the assignment to input's use_set.
      if (!output_use_set.empty()) {
        return *output_use_set.GetPrioritizedMeshName();
      }
      if (!output_src_set.empty()) {
        return *output_src_set.GetPrioritizedMeshName();
      }
      return std::nullopt;
    }
  }

  if (MeshesWithOrigins candidate_meshes =
          GetUseSet(output_operand.get().getDefiningOp());
      !candidate_meshes.empty()) {
    candidate_meshes.Intersect(input_use_set);

    if (!candidate_meshes.empty()) {
      if (candidate_meshes.size() > 1) {
        SDY_LOG(INFO)
            << "Picking first mesh for func output "
            << output_operand.getOperandNumber()
            << " from the intersection of the input and outputs use_set: {"
            << GetPrintableString(candidate_meshes) << "}";
      }

      return *candidate_meshes.GetPrioritizedMeshName();
    }
  }

  MeshesWithOrigins candidate_meshes = input_use_set;
  candidate_meshes.Intersect(GetSrcSet(output_operand));
  if (!candidate_meshes.empty()) {
    if (candidate_meshes.size() > 1) {
      SDY_LOG(INFO)
          << "Picking smallest mesh (in alphanum order) for func output "
          << output_operand.getOperandNumber()
          << " from the intersection of the input use_set and output src_set: {"
          << GetPrintableString(candidate_meshes) << "}";
    }
    return *candidate_meshes.GetPrioritizedMeshName();
  }

  return std::nullopt;
}

void AssignInputAndOutputToMesh(FuncOp func, BlockArgument input_arg,
                                OpOperand& return_operand, StringRef mesh_name,
                                sdy::MeshAttr mesh_attr,
                                RewriterBase& rewriter) {
  // Assign the output to the mesh.
  if (!isa<MeshTensorType>(return_operand.get().getType())) {
    rewriter.setInsertionPoint(return_operand.getOwner());
    return_operand.set(rewriter.create<AssignOp>(
        GetResultInfoLoc(func, return_operand.getOperandNumber())
            .value_or(return_operand.get().getLoc()),
        return_operand.get(), mesh_name, mesh_attr, kIoConstraintOutputOrigin));
  }

  // Assign the input to the mesh.
  if (!isa<MeshTensorType>(input_arg.getType())) {
    rewriter.setInsertionPointAfterValue(input_arg);
    input_arg.setType(MeshTensorType::getFullyReplicated(
        input_arg.getContext(), mesh_name, mesh_attr,
        cast<RankedTensorType>(input_arg.getType())));
    auto unassign = rewriter.create<UnassignOp>(input_arg.getLoc(), input_arg,
                                                kIoConstraintInputOrigin);
    rewriter.replaceAllUsesExcept(input_arg, unassign, unassign);
  }
}

sdy::MeshAttr GetMeshByName(
    const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name, StringRef name) {
  sdy::MeshAttr result = meshes_by_name.lookup(name);
  SDY_CHECK(result) << "Mesh with name '" << name.str() << "' not found.";
  return result;
}

class InferMeshAssignUsingInputOutputConstraintsPass
    : public impl::InferMeshAssignUsingInputOutputConstraintsPassBase<
          InferMeshAssignUsingInputOutputConstraintsPass> {
  using InferMeshAssignUsingInputOutputConstraintsPassBase::
      InferMeshAssignUsingInputOutputConstraintsPassBase;

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    for (FuncOp func_op : GetMpmdFunctions(getOperation())) {
      if (IsEntryPointFunction(func_op)) {
        runOnFunc(func_op, rewriter);
      }
    }
  }

  void runOnFunc(FuncOp func, RewriterBase& rewriter) {
    DenseMap<StringRef, sdy::MeshAttr> meshes_by_name =
        GetMeshesByName(GetTopologyMeshes(func));

    Operation* return_op = func.front().getTerminator();
    for (const InputOutputEquishardingConstraint& constraint : constraints) {
      SDY_CHECK_LT(constraint.input_index, func.getNumArguments());
      SDY_CHECK_LT(constraint.output_index, func.getNumResults());

      OpOperand& return_operand =
          return_op->getOpOperand(constraint.output_index);
      if (std::optional<StringRef> mesh_name = GetMeshForInputOutputAssignment(
              return_operand, func, constraint.input_index)) {
        AssignInputAndOutputToMesh(
            func, func.getArgument(constraint.input_index), return_operand,
            *mesh_name, GetMeshByName(meshes_by_name, *mesh_name), rewriter);
      } else if (verboseLogging) {
        SDY_LOG(INFO)
            << "No suitable mesh found for input-output constraint on input "
            << constraint.input_index << " and output "
            << constraint.output_index;
      }
    }

    UpdateFunctionType(func);
  }
};

class InferMeshAssignMeshForFuncLeavesPass
    : public impl::InferMeshAssignMeshForFuncLeavesPassBase<
          InferMeshAssignMeshForFuncLeavesPass> {
  using InferMeshAssignMeshForFuncLeavesPassBase::
      InferMeshAssignMeshForFuncLeavesPassBase;

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    for (FuncOp func_op : GetMpmdFunctions(getOperation())) {
      ArrayRef<NamedMeshAttr> meshes = GetTopologyMeshes(func_op);
      NamedMeshAttr first_mesh =
          *llvm::min_element(meshes, [](NamedMeshAttr a, NamedMeshAttr b) {
            return a.getName() < b.getName();
          });
      DenseMap<StringRef, sdy::MeshAttr> meshes_by_name =
          GetMeshesByName(meshes);
      if (IsEntryPointFunction(func_op)) {
        runOnEntryPointFunc(func_op, first_mesh, meshes_by_name, rewriter);
      } else {
        runOnNonEntryPointFunc(func_op, first_mesh, meshes_by_name, rewriter);
      }
    }
  }

  void runOnEntryPointFunc(
      FuncOp func, NamedMeshAttr first_mesh,
      const llvm::DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
      RewriterBase& rewriter) {
    // We can assign any mesh to the unused func inputs, and we arbitrarily
    // pick the first mesh.
    AssignUnusedFuncInputs(func, first_mesh);
    AssignFuncOutputs(func, cast<func::ReturnOp>(func.front().getTerminator()),
                      meshes_by_name, first_mesh.getName(), rewriter);
    runOnFunc(func, first_mesh, meshes_by_name, rewriter);
    UpdateFunctionType(func);
  }

  void runOnNonEntryPointFunc(
      FuncOp func, NamedMeshAttr first_mesh,
      const llvm::DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
      RewriterBase& rewriter) {
    runOnFunc(func, first_mesh, meshes_by_name, rewriter);
    // Needs to run after `runOnFunc`, because `runOnFunc` will clear the
    // use_set, but these functions will set the use_set of unused func inputs
    // and outputs.
    AssignUnusedCalleeInputs(func, first_mesh.getName(), rewriter);
    AssignUnusedCalleeOutputs(func, first_mesh.getName(), rewriter);
  }

  void runOnFunc(FuncOp func, NamedMeshAttr first_mesh,
                 const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
                 RewriterBase& rewriter) {
    // Applied to any function in the module.
    func.getFunctionBody().walk<WalkOrder::PreOrder>(
        [this, first_mesh_name = first_mesh.getName(), &meshes_by_name,
         &rewriter](Operation* op) {
          if (auto fragment = dyn_cast<FragmentOp>(op)) {
            // Skip fragment ops and their regions for efficiency, as they are
            // already assigned.
            return WalkResult::skip();
          }

          if (IsMeshlessOp(op)) {
            if (op->getNumResults() == 0) {
              AssignResultlessOp(op, first_mesh_name, rewriter);
            } else if (op->use_empty()) {
              AssignUnusedOp(op, first_mesh_name, meshes_by_name, rewriter);
            } else {
              ClearUseSet(op);
            }
            return WalkResult::skip();
          } else if (IsTerminalNodeInAnalysis(op)) {
            AssignOperandsOfAnalysisTerminalNodes(op, meshes_by_name,
                                                  first_mesh_name, rewriter);
            ClearUseSet(op);
            return WalkResult::skip();
          } else if (sdy::inDialect<MpmdDialect>(op)) {
            // Don't skip the body of other MPMD ops.
            return WalkResult::advance();
          }

          return WalkResult::skip();
        });
    ClearUseSet(func);
  }

  // Picks a mesh assignment for each func output. This assigns an arbitrary
  // mesh from the use_set if it exists, or otherwise from the src_set (note
  // that the outputs may not have a use_set).
  //
  // The use_set could exist since an op result can be used in both a return op
  // and other computations e.g. if we have
  //
  // x = some_op
  // y = assign x "m1"
  // return x, y
  //
  // then x is a func output and has use_set {m1}.
  //
  // We pick a mesh for the return value v by wrapping it in an AssignOp, i.e.
  // we return AssignOp(v) instead of v, to make clear precisely which mesh it
  // is assigned to. This is not possible with attributes on the ops, and while
  // we could annotate the func result attrs, it would be harder to spot when
  // debugging.
  void AssignFuncOutputs(
      FuncOp func, func::ReturnOp op,
      const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
      StringRef first_mesh_name, OpBuilder& builder) {
    builder.setInsertionPoint(op);
    for (OpOperand& return_op_operand : op->getOpOperands()) {
      Value return_operand = return_op_operand.get();
      if (isa<MeshTensorType>(return_operand.getType())) {
        continue;
      }

      std::optional<StringRef> mesh_name =
          GetMeshForAssignment(return_op_operand, first_mesh_name,
                               GetUseSet(return_operand.getDefiningOp()));
      if (!mesh_name) {
        // When --infer-transfers=true, we want to assign it to a
        // mesh anyway. Otherwise, we want to emit an error.
        if (!inferTransfers) {
          op->emitError("Func output ") << return_op_operand.getOperandNumber()
                                        << " has no mesh to be assigned to.";
          IncErrorCountAndMaybeFail();
        }
        mesh_name = first_mesh_name;
      }
      return_op_operand.set(builder.create<AssignOp>(
          GetResultInfoLoc(func, return_op_operand.getOperandNumber())
              .value_or(return_operand.getLoc()),
          return_operand, *mesh_name, GetMeshByName(meshes_by_name, *mesh_name),
          kInferredOutputOrigin));
    }
  }

  // Returns a mesh from the src_set. If the src_set is empty, returns
  // default_mesh_name with a warning. Additionally, it returns a boolean value
  // which is true if the src_set is empty.
  std::pair<StringRef, bool> GetMeshFromSrcSet(MeshesWithOrigins src_set,
                                               StringRef default_mesh_name) {
    if (src_set) {
      if (src_set.empty()) {
        return {default_mesh_name, true};
      }
      return {*src_set.GetPrioritizedMeshName(), false};
    }
    return {default_mesh_name, false};
  }

  // Assigns a mesh to unused func outputs. These outputs could have a src_set,
  // e.g. if they have a func input as operand, but they will not have a use_set
  // since they have no uses.
  //
  // This only sets the use_set, it does not create assign ops. Assign ops will
  // be created later during the rewrite.
  void AssignUnusedCalleeOutputs(FuncOp callee, StringRef first_mesh_name,
                                 OpBuilder& builder) {
    for (MpmdDataflowEdge& edge : GetMpmdDataflowEdgesForFuncResults(callee)) {
      // Filter away outputs with uses.
      if (llvm::any_of(edge.targets, [](Value v) { return !v.use_empty(); })) {
        continue;
      }

      OpOperand& return_op_operand = *edge.sources.front();
      Value return_operand = return_op_operand.get();
      if (isa<MeshTensorType>(return_operand.getType())) {
        continue;
      }

      auto [mesh_name, inferred_name] =
          GetMeshFromSrcSet(GetSrcSet(return_op_operand), first_mesh_name);
      if (inferred_name && !inferTransfers) {
        return_op_operand.getOwner()->emitError("Callee @")
            << callee.getSymName() << " unused output "
            << return_op_operand.getOperandNumber()
            << " has no mesh to be assigned to.";
        return IncErrorCountAndMaybeFail();
      }
      SetResUseSet(callee, return_op_operand.getOperandNumber(),
                   MeshesWithOrigins(UnusedCalleeOutputMeshWithOrigin(
                       callee->getContext(), mesh_name)),
                   builder);
    }
  }

  // Assigns a mesh to unused callee inputs. These inputs could have a src_set
  // but will not have a use_set. So we pick a mesh from the src_set. This only
  // sets the use_set, it does not create assign ops. Assign ops will be created
  // later during the rewrite.
  void AssignUnusedCalleeInputs(FuncOp callee, StringRef first_mesh_name,
                                OpBuilder& builder) {
    for (BlockArgument arg : callee.getArguments()) {
      if (arg.use_empty() && isa<RankedTensorType>(arg.getType())) {
        auto [mesh_name, inferred_name] = GetMeshFromSrcSet(
            GetSrcSet(callee, arg.getArgNumber()), first_mesh_name);
        if (inferred_name && !inferTransfers) {
          emitError(arg.getLoc(), "Callee @")
              << callee.getSymName() << " unused input " << arg.getArgNumber()
              << " has no mesh to be assigned to.";
          return IncErrorCountAndMaybeFail();
        }
        SetArgUseSet(callee, arg.getArgNumber(),
                     MeshesWithOrigins(UnusedCalleeInputMeshWithOrigin(
                         callee->getContext(), mesh_name)),
                     builder);
      }
    }
  }

  // Assigns a mesh to unused intermediate ops, by creating an assign user.
  // These ops could have a src_set, e.g. if they have a func input as operand.
  // But they will not have a use_set since they have no uses.
  //
  // The ops may have no results, in which case we wrap them in a fragment since
  // we cannot create an assign user for an op with no results.
  //
  // This relies on the pass pipeline not doing DCE, since the newly created
  // assign ops will be unused and would otherwise be DCE-ed.
  void AssignUnusedOp(Operation* op, StringRef default_mesh_name,
                      const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
                      RewriterBase& rewriter) {
    StringRef mesh_name;
    if (MeshesWithOrigins src_set = GetSrcSet(op)) {
      if (src_set.empty()) {
        op->emitError("src_set must not be empty for this op.");
        // In this case, we have to stop here, or otherwise we would crash
        // below.
        return signalPassFailure();
      }
      mesh_name = *src_set.GetPrioritizedMeshName();
    } else {
      // If there is no src_set attr, then the src_set is all meshes and so we
      // pick an arbitrary mesh.
      mesh_name = default_mesh_name;
    }
    rewriter.setInsertionPointAfter(op);
    sdy::MeshAttr mesh = GetMeshByName(meshes_by_name, mesh_name);
    for (Value res : op->getResults()) {
      rewriter.create<AssignOp>(op->getLoc(), res, mesh_name, mesh,
                                kInferredUnusedOrigin);
    }

    ClearUseSet(op);
  }

  // Like `AssignUnusedOp` but for ops that have no results. In this case, we
  // wrap the op in a fragment, since the op has no results to add.
  //
  // If the op is not pure, then we clone it per mesh. This is important because
  // non-pure ops may have side effects, and we want to keep them per-mesh. E.g.
  // the sdy.sharding_group op needs to be cloned per mesh.
  //
  // TODO: b/397895929 - Improve how we handle non-pure meshless ops. Ideally,
  // we should clone them according to the actual mesh assignment, rather than
  // the src_set, which may not actually be the assignment.
  void AssignResultlessOp(Operation* op, StringRef default_mesh_name,
                          RewriterBase& rewriter) {
    std::optional<SetVector<StringRef>> src_set = GetSrcMeshes(op);
    if (src_set) {
      if (src_set->empty()) {
        op->emitError("src_set must not be empty for resultless ops.");
        return signalPassFailure();
      }
    } else {
      // If there is no src_set attr, then we at least assign it to a default
      // mesh.
      src_set = SetVector<StringRef>();
      src_set->insert(default_mesh_name);
    }
    ClearUseSetAndSrcSet(op);
    rewriter.setInsertionPointAfter(op);

    for (StringRef mesh_name : src_set->getArrayRef()) {
      WrapOpWithFragment(op, mesh_name, rewriter);
      if (isPure(op)) {
        // For pure ops, we only need to wrap it in a fragment once. But for
        // non-pure ops, we need to keep them associated with each src.
        break;
      }
    }
    rewriter.eraseOp(op);
  };

  // Assigns a mesh to unused func inputs. These inputs will not have a src_set
  // nor a use_set as they are unused. So we can assign any mesh.
  void AssignUnusedFuncInputs(FuncOp func, NamedMeshAttr first_mesh) {
    for (BlockArgument arg : func.getArguments()) {
      if (auto type = dyn_cast<RankedTensorType>(arg.getType());
          type && arg.use_empty()) {
        SDY_LOG(INFO) << "Picking first mesh for unused func input "
                      << arg.getArgNumber() << " from the topology, i.e. "
                      << std::string_view(first_mesh.getName());
        arg.setType(MeshTensorType::getFullyReplicated(
            func->getContext(), first_mesh.getName(), first_mesh.getMesh(),
            type));
      }
    }
  }

  // Treats the operands of analysis-terminal-nodes (e.g. mpmd.broadcast and
  // mpmd.reduce) as leaves of the tree, and assigns them to a specific mesh.
  void AssignOperandsOfAnalysisTerminalNodes(
      Operation* op, const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
      StringRef first_mesh_name, OpBuilder& builder) {
    builder.setInsertionPoint(op);

    // By picking a mesh in the use_set, we can save transfers. In particular,
    // if there are multiple meshes in the src_set and we know it will be needed
    // in one of the meshes in the use_set, we might as well assign it to one of
    // the meshes in the src_set that also appears in the use_set.
    MeshesWithOrigins preferred_meshes = GetUseSet(op);
    for (OpOperand& operand : op->getOpOperands()) {
      std::optional<StringRef> mesh_name =
          GetMeshForAssignment(operand, first_mesh_name, preferred_meshes);
      if (!mesh_name) {
        SDY_CHECK(inferTransfers)
            << "This should only happen if we are inferring "
               "transfers.";
        // This happens if the src_set is empty. In this case, we will have to
        // introduce transfers in the later pass, or error out.
        mesh_name =
            preferred_meshes.GetPrioritizedMeshName().value_or(first_mesh_name);
      }
      Value operand_val = operand.get();
      AssignOp assign = builder.create<AssignOp>(
          operand_val.getLoc(), operand_val, *mesh_name,
          GetMeshByName(meshes_by_name, *mesh_name), TerminalNodesOrigin(op));
      operand.set(builder.create<UnassignOp>(operand_val.getLoc(), assign));
    }
  }

  void IncErrorCountAndMaybeFail() {
    error_count_++;
    if (error_count_ == max_errors_) {
      return signalPassFailure();
    }
  }

  LogicalResult initialize(MLIRContext* context) final {
    error_count_ = 0;
    max_errors_ = GetValidatedMaxErrors(errorLimit);
    return success();
  }

  int error_count_ = 0;
  int max_errors_ = 1;
};

// If `op` is the reduce of a cross-mesh concat-reduce pair, then replace the
// concat-reduce pair with an mpmd.reduce and return true.
void ConvertConcatReduceOp(Operation* op, RewriterBase& rewriter) {
  if (isa<stablehlo::ConcatenateOp>(op)) {
    // Conversion of the concat will happen together with the stablehlo.reduce.
    return;
  }
  auto reduce = dyn_cast<stablehlo::ReduceOp>(op);
  if (!reduce) {
    return;
  }
  auto concat =
      reduce.getInputs().front().getDefiningOp<stablehlo::ConcatenateOp>();
  if (!concat) {
    return;
  }
  rewriter.setInsertionPoint(reduce);

  // Collapse the concat and reduce dim of the operands.
  SmallVector<Value> reshaped_operands;
  reshaped_operands.reserve(concat.getOperands().size());
  for (Value operand : concat.getOperands()) {
    auto reshape = rewriter.create<stablehlo::ReshapeOp>(
        operand.getLoc(), reduce->getResultTypes().front(), operand);
    if (operand.getDefiningOp()) {
      reshape->setDiscardableAttrs(
          operand.getDefiningOp()->getDiscardableAttrDictionary());
    }
    reshaped_operands.push_back(reshape);
  }

  rewriter.setInsertionPoint(reduce);
  MeshesWithOrigins src_set = GetSrcSet(op);
  auto mpmd_reduce = rewriter.replaceOpWithNewOp<ReduceOp>(
      reduce, reshaped_operands,
      op->getAttrOfType<ReductionAttr>(kMpmdReduceAnnotation));
  if (src_set) {
    SetSrcSet(mpmd_reduce, src_set, rewriter);
  }
  rewriter.eraseOp(concat);
}

// Converts the annotated op to an actual mpmd.reduce op, and also flattens any
// reduce chains.
//
// Pre-condition: op is annotated to be an MPMD reduce.
void ConvertAnnotatedReduceOp(Operation* op, RewriterBase& rewriter) {
  if (isa<stablehlo::ConcatenateOp, stablehlo::ReduceOp>(op)) {
    ConvertConcatReduceOp(op, rewriter);
    return;
  }

  auto reduction_attr = op->getAttrOfType<ReductionAttr>(kMpmdReduceAnnotation);

  SmallVector<Value> operands;
  SmallVector<ReduceOp> operands_to_erase;
  for (Value operand : op->getOperands()) {
    if (auto operand_reduce = operand.getDefiningOp<ReduceOp>();
        operand_reduce && reduction_attr == operand_reduce.getReduction()) {
      operands.append(operand_reduce->getOperands().begin(),
                      operand_reduce->getOperands().end());
      if (operand_reduce->hasOneUse()) {
        operands_to_erase.push_back(operand_reduce);
      }
    } else {
      operands.push_back(operand);
    }
  }

  rewriter.setInsertionPoint(op);
  MeshesWithOrigins src_set = GetSrcSet(op);
  auto reduce =
      rewriter.replaceOpWithNewOp<ReduceOp>(op, operands, reduction_attr);
  if (src_set) {
    SetSrcSet(reduce, src_set, rewriter);
  }
  for (ReduceOp operand : operands_to_erase) {
    rewriter.eraseOp(operand);
  }
}

// Annotates the binary op with the attribute that indicates the reduction type
// if the binary op's reduction type matches the target reduction type.
// Rewrites
//
// x = add(w0, w1)
//
// ~~>
//
// x = add(w0, w1) {mpmd.reduce = #mpmd.reduce<add>}
// The allowed binary ops are add, mul, max, min, and or.
void AnnotateBinaryOp(Operation* op, ReductionType target_reduction_type) {
  if (!op) {
    return;
  }

  if (GetReductionOpType(op) == target_reduction_type) {
    op->setAttr(kCanConvertToReduce, UnitAttr::get(op->getContext()));
    // If the op has a valid reduction type, it's a binary op.
    SDY_CHECK(op->getNumOperands() == 2 && op->getNumResults() == 1);
    AnnotateBinaryOp(op->getOperand(0).getDefiningOp(), target_reduction_type);
    AnnotateBinaryOp(op->getOperand(1).getDefiningOp(), target_reduction_type);
  }
}

// Returns true if `op` is a unary operation.
bool IsUnaryOperation(Operation* op) {
  return op->getNumOperands() == 1 && op->getNumResults() == 1;
}

// Annotates `op` with the attribute that indicates the reduction type if it is
// a binary op that is used by a mpmd.reduce<none> op.
// Rewrites
//
// x = add(w0, w1)
// y = mpmd.reduce<none>(x)
//
// ~~>
//
// x = add(w0, w1) {mpmd.can_convert_to_reduce}
// y = mpmd.reduce<none>(x)
// If any of w0 or w1 are from add ops, they will also be annotated recursively.
//
// This also annotates a concat-reduce pair if the concat is a cross-mesh
// concat.
//
// x = stablehlo.concat(w0, w1)
// y = stablehlo.reduce<add>(x)
// z = mpmd.reduce<none>(y)
//
// ~~>
// x = stablehlo.concat(w0, w1)
// y = stablehlo.reduce<add>(x) {mpmd.can_convert_to_reduce}
// z = mpmd.reduce<none>(y)
//
// This annotation skips any unary ops that uses the result of the binary op.
// For example,
//
// x = add(w0, w1)
// y = abs(x)
// z = mpmd.reduce<none>(y)
//
// ~~>
//
// x = add(w0, w1) {mpmd.can_convert_to_reduce}
// y = abs(x)
// z = mpmd.reduce<none>(y)
void AnnotateProducerOfNoneReduce(ReduceOp none_reduce) {
  if (none_reduce.getReductionType() != ReductionType::kNone) {
    return;
  }
  Operation* defining_op = none_reduce.getTensors().front().getDefiningOp();
  if (!defining_op) {
    return;
  }

  // Skip all the unary ops until we reach a binary op.
  Operation* current_op = defining_op;
  while (IsUnaryOperation(current_op)) {
    current_op = current_op->getOperands().front().getDefiningOp();
    if (!current_op) {
      return;
    }
  }

  std::optional<ReductionType> reduction_type = GetReductionOpType(current_op);
  if (reduction_type.has_value()) {
    AnnotateBinaryOp(current_op, *reduction_type);
    return;
  }

  // Match the `mpmd.reduce<none>(stablehlo.reduce<max>(x, y))` pattern. If the
  // concat is a cross-mesh concat, then we add the annotation to the reduce.
  if (isa<stablehlo::ReduceOp>(current_op)) {
    auto concat =
        current_op->getOperand(0).getDefiningOp<stablehlo::ConcatenateOp>();
    if (!concat) {
      return;
    }

    stablehlo::ReduceOp cross_mesh_reduce = MatchCrossMeshConcatReduce(concat);
    std::optional<ReductionType> reduction_type =
        cross_mesh_reduce
            ? ComputeReductionType(cross_mesh_reduce.getBody().front())
            : GetReductionOpType(current_op);
    if (!reduction_type) {
      return;
    }
    current_op->setAttr(kCanConvertToReduce,
                        UnitAttr::get(current_op->getContext()));
  }
}

class InferMeshConvertReduceOpsPass
    : public impl::InferMeshConvertReduceOpsPassBase<
          InferMeshConvertReduceOpsPass> {
  using InferMeshConvertReduceOpsPassBase::InferMeshConvertReduceOpsPassBase;

  void runOnOperation() final {
    FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    // Annotate the binary ops that are used by a reduce<none> op.
    // This annotation will allow us to introduce reduce ops. For example, an
    // add op with the annotation will be converted to a reduce<add> op.
    func.walk(
        [&](ReduceOp reduce_op) { AnnotateProducerOfNoneReduce(reduce_op); });

    // Walk the op before its region, so that we can delete any reduce-ops
    // without needing to walk its body. Also, we can skip the body of fragment
    // ops.
    func.getFunctionBody().walk<WalkOrder::PreOrder>(
        [&rewriter, this](Operation* op) {
          if (auto fragment = dyn_cast<FragmentOp>(op)) {
            return WalkResult::skip();
          } else if (IsMeshlessOp(op)) {
            if (IsMpmdReduceAnnotated(op) &&
                (inferCrossMeshReductions || CanConvertToReduce(op))) {
              ConvertAnnotatedReduceOp(op, rewriter);
              return WalkResult::skip();
            }
          }
          return WalkResult::advance();
        });
    // We no longer need the temporary can_convert_to_reduce attribute.
    ClearCanConvertAttr(func);
  }
};

// Attribute name to annotate where the current value was copied to, when
// copying a value across meshes.
inline constexpr StringRef kMpmdCopied = "mpmd.copies";

// Assigns the results of `callee` to the use_set, duplicating results which are
// assigned to multiple meshes. The original result will be annotated with an
// attribute indicating the position of the clones. E.g.
//
// func.func mpmdf(...) -> (T {use_set=[m1,m2,m3]})
// ~~>
// func.func mpmdf(...) -> (T_m1 {mpmd.copies=[1,2]}, T_m2, T_m3)
//
// Note that this does not update the type of the func op itself. That is done
// separately.
void AssignCalleeFuncResultsUsingAnalysis(
    FuncOp callee, const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
    RewriterBase& rewriter) {
  auto return_op =
      cast<func::ReturnOp>(callee.getBody().front().getTerminator());
  rewriter.setInsertionPoint(return_op);

  SmallVector<Value> new_operands(return_op.getOperands());
  // Process the existing return operands, copying them as needed and adding
  // them to the set of `new_operands`.
  for (auto [res_num, return_val] : llvm::enumerate(return_op.getOperands())) {
    if (isa<MeshTensorType>(return_val.getType())) {
      continue;
    }

    SetVector<StringRef> mesh_names =
        GetResUseSet(callee, res_num).MeshNamesOrEmpty();

    // This can happen when a callee result is unused. We only handle this
    // during the rewrite, because this requires the use set analysis to
    // determine that it is in fact unused.
    if (mesh_names.empty()) {
      SDY_CHECK(return_val.getDefiningOp())
          << "Callee should not have value chained through from arg to result "
             "directly.";
      StringRef mesh_name;
      if (MeshesWithOrigins src_meshes = GetSrcSet(return_val.getDefiningOp());
          src_meshes && !src_meshes.empty()) {
        mesh_name = *src_meshes.GetPrioritizedMeshName();
      } else {
        mesh_name = *llvm::min_element(llvm::map_range(
            meshes_by_name, [](const auto& it) { return it.first; }));
      }
      mesh_names.insert(mesh_name);
      SDY_LOG(INFO) << "Callee @" << std::string_view(callee.getSymName())
                    << " output " << res_num << " is unused. Assigning to mesh "
                    << std::string_view(mesh_name);
    }

    if (mesh_names.size() > 1) {
      // If the result is assigned to multiple meshes, annotate the result with
      // the positions where it is copied to.
      callee.setResultAttr(
          res_num, kMpmdCopied,
          rewriter.getDenseI64ArrayAttr(llvm::to_vector(llvm::seq<int64_t>(
              new_operands.size(),
              new_operands.size() + mesh_names.size() - 1))));
    }

    // Assign the results to the corresponding meshes. If a result has multiple
    // meshes, we copy it such that each result corresponds to a single mesh.
    for (auto [i, mesh_name] : llvm::enumerate(mesh_names.getArrayRef())) {
      auto assign =
          rewriter.create<AssignOp>(return_val.getLoc(), return_val, mesh_name,
                                    GetMeshByName(meshes_by_name, mesh_name));
      if (i == 0) {
        new_operands[res_num] = assign;
      } else {
        new_operands.push_back(assign);
      }
    }
  }

  // Finally, update the return_op with the new operands.
  return_op->setOperands(new_operands);
}

// Assigns the arguments of `callee` to according to the meshes of assign users,
// duplicating arguments which are assigned to multiple meshes. The original
// argument will be annotated with an attribute indicating the position of the
// clones. E.g.
//
// func.func mpmdf(%arg0: T {use_set=[m1,m2,m3]}) -> ...
// ~~>
// func.func mpmdf(%arg0: T_m1 {mpmd.copies=[1,2]}, %arg1: T_m2, %arg2: T_m3) ->
// ...
//
// Note that this does not update the type of the func op itself. That is done
// separately.
void AssignCalleeFuncArgsToAssignUsers(
    FuncOp callee, const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
    RewriterBase& rewriter) {
  Block& body = callee.front();
  rewriter.setInsertionPointToStart(&body);
  // Process the existing args, copying them as needed.
  // Save the argument list to a vector, because we'll be rewriting the
  // arguments.
  for (auto [arg_num, arg] :
       llvm::enumerate(llvm::to_vector(callee.getArguments()))) {
    if (isa<MeshTensorType>(arg.getType())) {
      continue;
    }

    SDY_CHECK(!arg.getUsers().empty())
        << "Callee argument is unused. It should have been removed in the "
           "mpmd-erase-unused-callee-block-arguments pass.";

    // Group users by mesh_name, to avoid having to loop through the user list
    // multiple times, which may be problematic since we will be updating the
    // user list.
    DenseMap<StringRef, SmallVector<AssignOp>> assign_users_by_mesh_name;
    for (auto user : arg.getUsers()) {
      if (auto assign_user = dyn_cast<AssignOp>(user)) {
        assign_users_by_mesh_name[assign_user.getType().getMeshName()]
            .push_back(assign_user);
      }
    }
    SmallVector<StringRef> mesh_names = llvm::map_to_vector(
        assign_users_by_mesh_name, [](const auto& it) { return it.first; });
    llvm::sort(mesh_names);

    SDY_CHECK(!mesh_names.empty()) << "No mesh names found.";
    if (mesh_names.size() > 1) {
      // If the arg is assigned to multiple meshes, annotate the arg with
      // the positions where it is copied to.
      callee.setArgAttr(
          arg_num, kMpmdCopied,
          rewriter.getDenseI64ArrayAttr(llvm::to_vector(llvm::seq<int64_t>(
              body.getNumArguments(),
              body.getNumArguments() + mesh_names.size() - 1))));
    }

    // Assign the args to the corresponding meshes. If a arg has multiple
    // meshes, we copy it such that each arg corresponds to a single mesh.
    // To preserve the arg type, we replace it users with an unassign.
    auto local_type = cast<RankedTensorType>(arg.getType());
    for (auto [i, mesh_name] : llvm::enumerate(mesh_names)) {
      auto mesh_type = MeshTensorType::getFullyReplicated(
          arg.getContext(), mesh_name, GetMeshByName(meshes_by_name, mesh_name),
          local_type);
      UnassignOp unassign_op;
      if (i == 0) {
        arg.setType(mesh_type);
        unassign_op = rewriter.create<UnassignOp>(arg.getLoc(), arg);
      } else {
        unassign_op = rewriter.create<UnassignOp>(
            arg.getLoc(), body.addArgument(mesh_type, arg.getLoc()));
      }

      if (auto users_it = assign_users_by_mesh_name.find(mesh_name);
          users_it != assign_users_by_mesh_name.end()) {
        for (auto assign_user : users_it->second) {
          assign_user.setOperand(unassign_op);
        }
      }
    }
  }
}

// Rewrites the `call_op` using the updated information from its callee func.
// I.e. it takes into account the results and arguments which have been copied
// because they were used in multiple meshes, and creates assigns into the call,
// and unassigns out of the call accordingly.
//
// Pre-condition: The callee should already be rewritten to have mesh types.
// Also, the users of the callee should already be rewritten, i.e. assigned to a
// mesh.
void RewriteAccordingToUpdatedCallee(CallOp call_op, RewriterBase& rewriter) {
  auto callee = GetCalleeFunc(call_op);
  Block& call_body = callee.front();
  rewriter.setInsertionPointAfter(call_op);

  // Get assigned operands for the new call.
  std::vector<Value> new_operands(call_body.getNumArguments());
  for (auto [arg_num, operand] : llvm::enumerate(call_op.getOperands())) {
    if (isa<MeshTensorType>(operand.getType())) {
      new_operands[arg_num] = operand;
      continue;
    }
    SDY_CHECK(isa<MeshTensorType>(call_body.getArgument(arg_num).getType()));
    new_operands[arg_num] = rewriter.create<AssignOp>(
        operand.getLoc(), call_body.getArgument(arg_num).getType(), operand);

    if (auto copies =
            callee.getArgAttrOfType<DenseI64ArrayAttr>(arg_num, kMpmdCopied)) {
      for (int64_t cloned_arg_index : copies.asArrayRef()) {
        SDY_CHECK(isa<MeshTensorType>(
            call_body.getArgument(cloned_arg_index).getType()));
        new_operands[cloned_arg_index] = rewriter.create<AssignOp>(
            operand.getLoc(), call_body.getArgument(cloned_arg_index).getType(),
            operand);
      }
    }
  }

  // Create the new call and copy attrs over.
  auto new_call_op = rewriter.create<CallOp>(
      call_op.getLoc(), call_body.getTerminator()->getOperandTypes(),
      new_operands, call_op.getCalleeAttr());
  new_call_op->setDiscardableAttrs(call_op->getDiscardableAttrDictionary());

  // Replace uses of the original call with the new call's results.
  for (OpResult res : call_op.getResults()) {
    DenseMap<Type, int64_t> type_to_arg_num;
    type_to_arg_num.try_emplace(
        new_call_op.getResult(res.getResultNumber()).getType(),
        res.getResultNumber());

    if (auto copies = callee.getResultAttrOfType<DenseI64ArrayAttr>(
            res.getResultNumber(), kMpmdCopied)) {
      for (int64_t cloned_arg_num : copies.asArrayRef()) {
        type_to_arg_num.try_emplace(
            new_call_op.getResult(cloned_arg_num).getType(), cloned_arg_num);
      }
    }

    // We need to save the user list, because we'll be rewriting the users.
    for (Operation* user : llvm::to_vector(res.getUsers())) {
      if (auto assign_user = dyn_cast<AssignOp>(user)) {
        auto arg_num_it = type_to_arg_num.find(assign_user.getType());
        SDY_CHECK(arg_num_it != type_to_arg_num.end())
            << "Argument number for type " << debugString(assign_user.getType())
            << " not found";
        assign_user.setOperand(rewriter.create<UnassignOp>(
            assign_user.getLoc(), new_call_op.getResult(arg_num_it->second)));
      }
    }
  }
  rewriter.eraseOp(call_op);
}

// Assigns the arguments of the entrypoint func to the first mesh of its
// assign users in lexicographic order (i.e. default string sorting order).
//
// func.func main(%arg0: T) -> ...
// {
//    %0 = assign %arg0 -> m2
//    %1 = assign %arg0 -> m1
//    %2 = assign %arg0 -> m3
//    return %0, %1, %2
// }
// ~~>
// func.func main(%arg0: T_m1) ->
// {
//    %0 = unassign %arg0 -> T
//    %1 = assign %0 -> m1
//    %2 = assign %0 -> m3
//    return %arg0, %1, %2
// }
//
// Note that this does not update the type of the func op itself. That is done
// separately.
//
// If an argument isn't a mesh tensor type yet, and a mesh tensor type needs to
// be built, then we check for consistency in the memory kinds of the assign
// users. If it succeeds, we return true. Otherwise, we return false.
//
// Pre-condition: The users of the args should either be mesh tensors, or only
// used by AssignOps.
bool AssignEntrypointFuncArgsToAssignUsers(FuncOp entrypoint_func,
                                           RewriterBase& rewriter) {
  Block& body = entrypoint_func.front();
  rewriter.setInsertionPointToStart(&body);
  for (auto arg : body.getArguments()) {
    if (isa<MeshTensorType>(arg.getType())) {
      continue;
    }

    // Find the memory kind, if any, that should be used for the mesh tensor.
    llvm::SetVector<StringAttr> memory_kinds;
    bool has_user_with_undefined_memory_kind = false;
    for (auto user : arg.getUsers()) {
      if (auto assign_user = dyn_cast<AssignOp>(user)) {
        StringAttr user_memory_kind = assign_user.getType().getMemoryKind();
        if (user_memory_kind) {
          memory_kinds.insert(user_memory_kind);
        } else {
          has_user_with_undefined_memory_kind = true;
        }
      }
    }
    if (has_user_with_undefined_memory_kind && !memory_kinds.empty()) {
      entrypoint_func.emitError()
          << "Argument " << arg.getArgNumber()
          << " has different memory kinds assigned to it. Found at least one "
             "user with undefined memory kind and at least one user with a "
             "memory kind.";
      return false;
    }

    if (memory_kinds.size() > 1) {
      // TODO: b/374994155 - Consider a different verification pass for memory
      // kinds.
      entrypoint_func.emitError()
          << "Argument " << arg.getArgNumber()
          << " has different memory kinds assigned to it.";
      return false;
    }

    StringAttr memory_kind =
        has_user_with_undefined_memory_kind ? nullptr : *memory_kinds.begin();

    StringRef mesh_name =
        *llvm::min_element(llvm::map_range(arg.getUsers(), [](Operation* user) {
          return cast<AssignOp>(user).getType().getMeshName();
        }));

    arg.setType(MeshTensorType::get(arg.getContext(), mesh_name,
                                    cast<RankedTensorType>(arg.getType()),
                                    memory_kind));

    UnassignOp unassign_op = rewriter.create<UnassignOp>(arg.getLoc(), arg);
    rewriter.replaceAllUsesExcept(arg, unassign_op, unassign_op);
  }
  return true;
}

// Clones `op` into `consumer` and returns the cloned results.
ValueRange AbsorbOpByCloning(FragmentOp consumer, Operation* op,
                             IRMapping operand_mapping,
                             RewriterBase& rewriter) {
  rewriter.setInsertionPointToStart(consumer.getBody());
  Operation* cloned_op = rewriter.clone(*op, operand_mapping);
  return cloned_op->getResults();
}

// Inlines `op` into `consumer` and returns the inlined results, given a mapping
// between operands/free variables of `op` and block arguments of consumer.
// `has_free_vars` indicates whether `op` contains any free variables (which
// would be included in `operand_mapping`).
ValueRange AbsorbOpByInlining(FragmentOp consumer, Operation* op,
                              IRMapping operand_or_freevar_mapping,
                              bool has_free_vars, RewriterBase& rewriter) {
  for (auto& [operand_or_freevar, new_value] :
       operand_or_freevar_mapping.getValueMap()) {
    // As we move the op to the body of a fragment and because fragments do
    // not have free variables, we need to guarantee that op's free variables
    // and operands will be replaced with the respective arguments of the
    // fragment.
    rewriter.replaceUsesWithIf(operand_or_freevar, new_value,
                               [&](OpOperand& operand) {
                                 if (!has_free_vars) {
                                   // In this case, no need to visit the op's
                                   // ancestor, which could go all the way to
                                   // the module.
                                   return op == operand.getOwner();
                                 }
                                 return op->isAncestor(operand.getOwner());
                               });
  }
  rewriter.moveOpBefore(op, consumer.getBody(), consumer.getBody()->begin());
  return op->getResults();
}

// Absorbs a meshless `op` into a consumer fragment.
// This means the op will be cloned (if it has multiple users) or inlined (if it
// has a single user) into the consumer fragment. Operands of the absorbed op
// are appended to the consumer's operands.
//
// For example,
//  ```
//    x = op1(a, b)
//    a = assign x
//    fragment(a) (%arg) {
//      y = op2(%arg)
//      return y
//    }
//      ~~>
//    a' = assign a
//    b' = assign b
//    fragment(a', b') (%arg0, %arg1) {
//      y1 = op1(%arg0, %arg1)
//      y2 = op2(y1)
//      return y2
//    }
//  ```
//
// TODO: jupvfranco - we could optimize this code by checking if the op's
// operands already exist in the consumer's operands. For now, we avoid this for
// the sake of code simplicity.
void AbsorbMeshlessProducer(FragmentOp consumer, Operation* op,
                            bool op_used_by_consumer_only,
                            RewriterBase& rewriter) {
  rewriter.setInsertionPoint(consumer);
  StringRef mesh_name = consumer.getMeshName();
  sdy::MeshAttr mesh_attr = GetMeshOrFail(consumer);
  Block* body = consumer.getBody();

  // When inlining the op in a fragment, get its operands before inlining as
  // they'll change during inlining and we still need them later on to extend
  // the consumer fragment's operands.
  // It isn't enough to use the operands of the op, as some control-flow ops
  // (e.g., stablehlo.case) may use free variables.
  llvm::SetVector<Value> op_operands_and_free_vars;
  getUsedValuesDefinedAbove(op->getRegions(), op_operands_and_free_vars);
  bool has_free_vars = !op_operands_and_free_vars.empty();
  op_operands_and_free_vars.insert(op->operand_begin(), op->operand_end());

  IRMapping mapping;
  for (Value operand : op_operands_and_free_vars) {
    mapping.map(operand,
                body->insertArgument(body->getNumArguments(), operand.getType(),
                                     operand.getLoc()));
  }
  ValueRange new_results =
      op_used_by_consumer_only
          ? AbsorbOpByInlining(consumer, op, mapping, has_free_vars, rewriter)
          : AbsorbOpByCloning(consumer, op, mapping, rewriter);
  BitVector erase_args(body->getNumArguments());
  // Make sure that the body of consumer uses the op results directly instead
  // of using the arguments respective of their uses.
  // Note: when the op is inlined, old_result and new_result are the same.
  for (auto [old_result, new_result] :
       llvm::zip(op->getResults(), new_results)) {
    for (Operation* user : old_result.getUsers()) {
      AssignOp assign_op = cast<AssignOp>(user);
      for (OpOperand& assign_use : assign_op->getUses()) {
        if (assign_use.getOwner() == consumer) {
          rewriter.replaceAllUsesWith(
              body->getArgument(assign_use.getOperandNumber()), new_result);
          erase_args.set(assign_use.getOperandNumber());
        }
      }
    }
  }

  body->eraseArguments(erase_args);
  std::vector<Value> new_consumer_operands;
  for (auto index : erase_args.flip().set_bits()) {
    // The number of arguments is the same or higher as we may have appended
    // more arguments to the block of the fragment.
    if (index < consumer.getNumOperands()) {
      new_consumer_operands.push_back(consumer->getOperand(index));
    }
  }
  rewriter.setInsertionPoint(consumer);
  for (Value operand : op_operands_and_free_vars) {
    new_consumer_operands.push_back(rewriter.create<AssignOp>(
        operand.getLoc(),
        MeshTensorType::getFullyReplicated(
            operand.getContext(), mesh_name, mesh_attr,
            cast<RankedTensorType>(operand.getType())),
        operand));
  }
  consumer->setOperands(new_consumer_operands);
}

// Wraps the op in fragments based on the mesh types of its assign users. At
// most we get one fragment per mesh type.
void WrapBasedOnAssignUsers(Operation* op, RewriterBase& rewriter) {
  // Use a set, instead of a vector, to avoid creating redundant fragments.
  DenseSet<StringRef> user_mesh_types;
  for (Operation* user : op->getUsers()) {
    AssignOp assign_op = cast<AssignOp>(user);
    user_mesh_types.insert(assign_op.getType().getMeshName());
  }
  for (StringRef mesh_name : user_mesh_types) {
    WrapOpWithFragment(
        op, mesh_name, rewriter,
        /*should_replace_use=*/[&mesh_name](OpOperand& use) {
          if (auto assign_user = dyn_cast<AssignOp>(use.getOwner())) {
            return assign_user.getType().getMeshName() == mesh_name;
          }
          return false;
        });
  }
}

// Assigns the meshless `op` to a mesh either by being absorbed (by cloning or
// inlining) or wrapped in a new fragment. In particular, if `op`:
// - is used by a single fragment, then it is inlined into the fragment;
// - is used by an op that is not a fragment (e.g., a transfer), then it is
// wrapped in a fragment;
// - is used by N fragments and N <= `max_clones`, or
// the op has no operands, then it is cloned into each of its fragment users.
//
// Pre-condition: The meshless `op` is used by AssignOps only.
void AssignOpBasedOnConsumers(Operation* op, const int max_clones,
                              RewriterBase& rewriter) {
  SDY_CHECK_GT(op->getNumResults(), 0)
      << "All ops with no results should have been assigned by "
         "now.";
  ClearUseSetAndSrcSet(op);
  DenseSet<FragmentOp> fragment_users;
  bool has_non_fragment_user = false;
  for (Operation* user : op->getUsers()) {
    auto assign_op = dyn_cast<AssignOp>(user);
    SDY_CHECK(assign_op) << "Expected user to be AssignOp, got: "
                         << user->getName().getStringRef().str();
    for (Operation* assign_user : assign_op->getUsers()) {
      if (auto fragment_op = dyn_cast<FragmentOp>(assign_user)) {
        fragment_users.insert(fragment_op);
      } else {
        has_non_fragment_user = true;
      }
    }
  }

  if (fragment_users.size() == 1 && !has_non_fragment_user) {
    // Absorb by the op by inlining it.
    DenseSet<Operation*> users_pre_inlining(op->getUsers().begin(),
                                            op->getUsers().end());
    AbsorbMeshlessProducer(*fragment_users.begin(), op,
                           /*op_used_by_consumer_only=*/true, rewriter);
    for (Operation* user : users_pre_inlining) {
      rewriter.eraseOp(user);
    }
    return;
  }

  // When `fragment_users.empty` then there's no fragment to absorb meshless
  // ops.
  // When `has_non_fragment_user=true` we cannot clone as any non-fragment
  // user of the assign ops will not be erased, meaning `op` cannot be
  // erased either, as well as all its predecessors.
  // When `op` has many users (> `kMaxClones`) we do not want it to be
  // cloned as it could be the root of a large tree, i.e., we would
  // replicate too much code, potentially slowing down this pass (and maybe
  // others) significantly.
  // In any other case, we clone `op` into each of its fragment users.
  if (fragment_users.empty() || has_non_fragment_user ||
      (fragment_users.size() > max_clones && op->getNumOperands() > 0)) {
    WrapBasedOnAssignUsers(op, rewriter);
  } else {
    for (FragmentOp fragment_user : fragment_users) {
      AbsorbMeshlessProducer(fragment_user, op,
                             /*op_used_by_consumer_only=*/false, rewriter);
    }
  }
  // All users of op must be unused at this point. The op was either: (a)
  // wrapped in one or more fragments, and its assign users replaced with
  // the respective fragments, or (b) cloned into each of its consumer
  // fragments. This means that all the assign users may be safely erased.
  llvm::SmallDenseSet<Operation*> users_to_erase(op->getUsers().begin(),
                                                 op->getUsers().end());
  for (Operation* user : users_to_erase) {
    rewriter.eraseOp(user);
  }
  // And because we erase all its users, then the op is also unused.
  // Note we need to get the previous op before erasing the current one.
  rewriter.eraseOp(op);
}

// Rewrites the terminator of the given `for_op` to return mesh tensor types.
// For each non-mesh tensor result, we create an AssignOp that assigns the
// result to the mesh of the result's use_set.
void RewriteForOpTerminator(
    ForOp for_op, const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
    RewriterBase& rewriter) {
  auto return_op =
      dyn_cast<ReturnOp>(for_op.getRegion().front().getTerminator());
  rewriter.setInsertionPoint(return_op);

  SmallVector<Value> new_operands;
  for (auto [res_num, return_val] : llvm::enumerate(return_op.getOperands())) {
    if (isa<MeshTensorType>(return_val.getType())) {
      continue;
    }

    SetVector<StringRef> mesh_names = GetUseMeshes(return_val.getDefiningOp());
    // TODO(petebu): Handle unused result.
    SDY_CHECK(!mesh_names.empty()) << "No mesh names found for return value";
    // TODO(b/401476674): Handle multiple meshes.
    SDY_CHECK_LE(mesh_names.size(), 1)
        << "Multiple mesh names found for return value";

    new_operands.push_back(rewriter.create<AssignOp>(
        return_val.getLoc(), return_val, mesh_names[0],
        GetMeshByName(meshes_by_name, mesh_names[0])));
  }

  return_op->setOperands(new_operands);
}

// Iterates over all ops of the function and (1) assigns every meshless op to a
// mesh (by wrapping it or absorbing it into a fragment) and (2) rewrites any
// call_op so that it can handle inputs/outputs used in multiple meshes and so
// that it returns mesh tensor types. This is analogous to
// WalkAndAbsorbMeshlessProducers, but for ForOp.
void RewriteForOpBody(ForOp for_op,
                      const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
                      const int max_clones, RewriterBase& rewriter) {
  Block& block = *for_op.getBody();
  for (Operation& operation :
       llvm::make_early_inc_range(llvm::reverse(block.getOperations()))) {
    Operation* op = &operation;
    if (auto call_op = dyn_cast<CallOp>(op)) {
      RewriteAccordingToUpdatedCallee(call_op, rewriter);
    } else if (auto for_op = dyn_cast<ForOp>(op)) {
      SDY_CHECK(false) << "Nested ForOp is not supported";
    } else if (IsMeshlessOp(op)) {
      AssignOpBasedOnConsumers(op, max_clones, rewriter);
    }
  }
}

// Rewrites the arguments of the given `for_op` to be mesh tensor types. For
// each non-mesh tensor argument, we create an UnAssignOp on entry to the loop
// body that converts the mesh tensor type to a local tensor type.
void RewriteForOpArgsAndTypes(
    ForOp for_op, const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
    RewriterBase& rewriter) {
  Block& body = for_op.getRegion().front();
  rewriter.setInsertionPointToStart(&body);

  for (auto [arg_num, arg] :
       llvm::enumerate(llvm::to_vector(for_op.getRegion().getArguments()))) {
    if (isa<MeshTensorType>(arg.getType())) {
      continue;
    }

    DenseMap<StringRef, SmallVector<AssignOp>> assign_users_by_mesh_name;
    for (Operation* user : arg.getUsers()) {
      if (auto assign_user = dyn_cast<AssignOp>(user)) {
        assign_users_by_mesh_name[assign_user.getType().getMeshName()]
            .push_back(assign_user);
      }
    }
    SmallVector<StringRef> mesh_names = llvm::map_to_vector(
        assign_users_by_mesh_name, [](const auto& it) { return it.first; });
    llvm::sort(mesh_names);

    // TODO(b/401476674): Handle multiple meshes.
    SDY_CHECK_LE(mesh_names.size(), 1)
        << "Multiple mesh names found for return value";

    if (mesh_names.size() == 1) {
      auto local_type = cast<RankedTensorType>(arg.getType());
      auto mesh_type = MeshTensorType::getFullyReplicated(
          arg.getContext(), mesh_names[0],
          GetMeshByName(meshes_by_name, mesh_names[0]), local_type);
      arg.setType(mesh_type);
      UnassignOp unassign_op = rewriter.create<UnassignOp>(arg.getLoc(), arg);

      if (auto users_it = assign_users_by_mesh_name.find(mesh_names[0]);
          users_it != assign_users_by_mesh_name.end()) {
        for (auto assign_user : users_it->second) {
          assign_user.setOperand(unassign_op);
        }
      }
    }
  }

  // Update ForOp result type.
  for (auto [res, op_type] : llvm::zip(
           for_op.getResults(),
           for_op.getRegion().front().getTerminator()->getOperandTypes())) {
    res.setType(op_type);
  }
}

// Rewrites the operands of the given `for_op` to be mesh tensor types. For
// each non-mesh tensor operand, we create an AssignOp that assigns the operand
// to the mesh of the operand's use_set.
void RewriteForOpOperands(ForOp for_op, RewriterBase& rewriter) {
  Block& for_body = for_op.getRegion().front();
  rewriter.setInsertionPoint(for_op);

  std::vector<Value> new_operands(for_op.getNumOperands());
  for (auto [arg_num, operand] : llvm::enumerate(for_op.getOperands())) {
    if (isa<MeshTensorType>(operand.getType())) {
      new_operands[arg_num] = operand;
      continue;
    }
    new_operands[arg_num] = rewriter.create<AssignOp>(
        operand.getLoc(), for_body.getArgument(arg_num).getType(), operand);
  }

  for_op->setOperands(new_operands);
}

// Rewrites the results of the given `for_op`. For each result, we create an
// UnassignOp that converts the result to a local tensor type.
void RewriteForOpResults(ForOp for_op, RewriterBase& rewriter) {
  rewriter.setInsertionPointAfter(for_op);
  for (OpResult res : for_op.getResults()) {
    for (Operation* user : res.getUsers()) {
      if (auto assign_user = dyn_cast<AssignOp>(user)) {
        assign_user.setOperand(
            rewriter.create<UnassignOp>(assign_user.getLoc(), res));
      }
    }
  }
  ClearUseSetAndSrcSet(for_op);
}

void RewriteForOp(ForOp for_op,
                  const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
                  const int max_clones, RewriterBase& rewriter) {
  RewriteForOpTerminator(for_op, meshes_by_name, rewriter);
  RewriteForOpBody(for_op, meshes_by_name, max_clones, rewriter);
  RewriteForOpArgsAndTypes(for_op, meshes_by_name, rewriter);
  RewriteForOpOperands(for_op, rewriter);
  RewriteForOpResults(for_op, rewriter);
}

// Iterates over all ops of the function and (1) assigns every meshless op to a
// mesh (by wrapping it or absorbing it into a fragment) and (2) rewrites any
// call_op so that it can handle inputs/outputs used in multiple meshes and so
// that it returns mesh tensor types.
void WalkAndAbsorbMeshlessProducers(
    FuncOp func_op, const DenseMap<StringRef, sdy::MeshAttr>& meshes_by_name,
    const int max_clones, RewriterBase& rewriter) {
  Block& block = func_op.getBody().front();
  for (Operation& operation :
       llvm::make_early_inc_range(llvm::reverse(block.getOperations()))) {
    Operation* op = &operation;
    if (auto call_op = dyn_cast<CallOp>(op)) {
      RewriteAccordingToUpdatedCallee(call_op, rewriter);
    } else if (auto for_op = dyn_cast<ForOp>(op)) {
      RewriteForOp(for_op, meshes_by_name, max_clones, rewriter);
    } else if (IsMeshlessOp(op)) {
      AssignOpBasedOnConsumers(op, max_clones, rewriter);
    }
  }
}

class BroadcastToTransfersPattern : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern<BroadcastOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter& rewriter) const override {
    auto unassign_op = op.getTensor().getDefiningOp<UnassignOp>();
    SDY_CHECK(unassign_op)
        << "Expected the operand of the BroadcastOp to be the "
           "result of an UnassignOp";
    bool module_rewritten = false;
    // We copy the users to a vector because we modify the list of users when
    // replacing the op, which would cause a segfault otherwise.
    SmallVector<Operation*> users(op->getUsers().begin(), op->getUsers().end());
    for (Operation* user : users) {
      if (auto assign_op = dyn_cast<AssignOp>(user)) {
        if (assign_op.getType() != unassign_op.getTensor().getType()) {
          rewriter.replaceOpWithNewOp<TransferOp>(
              assign_op, assign_op.getType(), unassign_op.getTensor());
        } else {
          assign_op->setOperand(0, unassign_op.getResult());
        }
        module_rewritten = true;
      }
    }
    return success(module_rewritten);
  }
};

class InferMeshRewriteUsingAnalysisPass
    : public impl::InferMeshRewriteUsingAnalysisPassBase<
          InferMeshRewriteUsingAnalysisPass> {
  using InferMeshRewriteUsingAnalysisPassBase::
      InferMeshRewriteUsingAnalysisPassBase;

  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    MLIRContext* context = module_op->getContext();
    IRRewriter rewriter(context);

    llvm::SmallVector<func::FuncOp> mpmd_functions =
        GetMpmdFunctions(module_op);

    // Rewrite all the callees before rewriting the entrypoint functions, so
    // that we can use the updated callee func info when we propagate through
    // the mpmd.calls in the entrypoint function.
    for (FuncOp func_op : mpmd_functions) {
      if (!IsEntryPointFunction(func_op)) {
        // Assigns meshless ops in `callee`s. This does not update the type
        // of the func. We reserve that to after the calls to the callee have
        // been updated. This only changes the `callee` and does not change
        // calls to the callee.
        DenseMap<StringRef, sdy::MeshAttr> meshes_by_name =
            GetMeshesByName(GetTopologyMeshes(func_op));

        // We assign results first, so that they can absorb meshless ops.
        AssignCalleeFuncResultsUsingAnalysis(func_op, meshes_by_name, rewriter);
        WalkAndAbsorbMeshlessProducers(func_op, meshes_by_name, maxClones,
                                       rewriter);
        AssignCalleeFuncArgsToAssignUsers(func_op, meshes_by_name, rewriter);
      }
    }
    for (FuncOp func_op : mpmd_functions) {
      if (IsEntryPointFunction(func_op)) {
        DenseMap<StringRef, sdy::MeshAttr> meshes_by_name =
            GetMeshesByName(GetTopologyMeshes(func_op));
        WalkAndAbsorbMeshlessProducers(func_op, meshes_by_name, maxClones,
                                       rewriter);
      }
    }
    for (FuncOp func_op : mpmd_functions) {
      UpdateFunctionType(func_op);
    }
  }
};

// TODO: b/359832656 - Use single walk instead of greedy rewriter.
class InferMeshFinalizePass
    : public impl::InferMeshFinalizePassBase<InferMeshFinalizePass> {
  using InferMeshFinalizePassBase::InferMeshFinalizePassBase;

  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    MLIRContext* context = module_op->getContext();

    IRRewriter rewriter(context);
    for (FuncOp func_op : GetMpmdFunctions(module_op)) {
      if (IsEntryPointFunction(func_op)) {
        // TODO: joelwee - Move this into the RewriteUsingAnalysisPass, as that
        // is the more natural place to handle the assignments.
        if (!AssignEntrypointFuncArgsToAssignUsers(func_op, rewriter)) {
          return signalPassFailure();
        }
      }
    }

    RewritePatternSet patterns(context);
    patterns.add<AssignOfUnassignSameMeshPattern,
                 DedupAssignOfUnassignAndTransferPattern,
                 AssignOfUnassignFuncArgPattern, BroadcastToTransfersPattern,
                 LowerMpmdReducePattern>(context);
    if (inferTransfers) {
      // Note that because we're doing this after running
      // `RewriteUsingAnalysis`, the introduction of transfers is naive: they
      // are created as early as possible: i.e. in the first place where
      // `src_set` is not contained in `use_set`. E.g.
      //
      // x = unassign x_m0
      // y = x + x
      // z = y + y
      // z_m1 = assign z -> m1
      //
      // ~~>
      //
      // x_m1 = transfer x_m0 -> m1
      // z_m1 = frag m1 {
      //   y = x + x
      //   z = y + y
      //   return z
      // }
      //
      // To improve this further, we would need to do some form of min-cut
      // analysis before the rewrite itself.
      // TODO: b/329221688 - Use heuristics or a min-cut analysis to improve the
      // creation of transfers.
      patterns.add<AssignOfUnassignPattern>(context);
    }
    // This converges in 2 passes, where each pass completes the following:
    // 1. Clean up AssignOps and UnassignOps.
    // 2. Nothing (i.e. detect convergence).
    //
    // So in theory, we could do this with an MLIR walk if we refactored the
    // cleanup patterns out of the RewritePattern classes.
    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(module_op, std::move(patterns), config))) {
      return signalPassFailure();
    }

    for (FuncOp func_op : GetMpmdFunctions(module_op)) {
      // Update function types once after all function arguments and return
      // operands have been replaced.
      UpdateFunctionType(func_op);
      if (failed(VerifyMeshAssignment(func_op))) {
        signalPassFailure();
      }
      ClearUseSetAndSrcSet(func_op);
    }
  }
};

}  // namespace

void addInferMeshPipeline(
    OpPassManager& pm,
    SmallVector<InputOutputEquishardingConstraint> inputOutputConstraints,
    InferMeshOptions options) {
  // If infer_transfers is true, then we always infer cross_mesh reductions too,
  // since infer_transfers is more permissive. Otherwise we may have dangling
  // annotated mpmd.reduce ops.
  if (options.inferTransfers) {
    options.inferCrossMeshReductions = true;
  }

  pm.addPass(createInferMeshPopulateUseSetPass());
  // We populate the src_set after we have the use_set info, as we want to use
  // the use_set to constrain where the func inputs can be transferred to. Func
  // inputs with a use_set will have their src_set constrained to the use_set.
  // I.e. they are only assigned to meshes they are used in, as defined by the
  // user (i.e. before any inference). If they are not used in any mesh, then
  // they remain unconstrained, i.e. they can be assigned to any mesh. We
  // populate the src_set to know which meshes we can assign our outputs to.
  pm.addPass(createInferMeshPopulateSrcSetPass());

  if (!inputOutputConstraints.empty()) {
    // Use input_output_constraints to assign the func outputs and inputs,
    // before making any other assignments.
    InferMeshAssignUsingInputOutputConstraintsPassOptions constraints_options;
    constraints_options.constraints = std::move(inputOutputConstraints);
    constraints_options.verboseLogging = options.errorLimit == -1;
    pm.addPass(createInferMeshAssignUsingInputOutputConstraintsPass(
        std::move(constraints_options)));
  }

  // Convert annotated ops before assignment, which uses the converted reduce
  // ops.
  // If `options.inferCrossMeshReductions` is off and the user didn't explicitly
  // tag an op as being a reduction (via mpmd.reduce), then this pass will be a
  // noop, as no operation should be annotated. However, if the user did tag
  // operations for reductions, then we want to apply this pass, even if the
  // flag is off.
  pm.addNestedPass<FuncOp>(createInferMeshConvertReduceOpsPass(
      InferMeshConvertReduceOpsPassOptions{options.inferCrossMeshReductions}));

  // Validate that every op can be assigned to some mesh.
  if (!options.inferTransfers) {
    pm.addPass(createInferMeshValidateSrcSetNotEmptyPass(
        InferMeshValidateSrcSetNotEmptyPassOptions{options.errorLimit}));
  }

  // Assigning meshes to FuncOp leaves is enough, as every op in a FuncOp flows
  // into some FuncOp leaf or AssignOp (which already has a mesh). After this
  // pass, the leaf will have a use_set, which will be propagated when we run
  // the PopulateUseSet pass again.
  InferMeshAssignMeshForFuncLeavesPassOptions assign_mesh_for_leaves_options;
  assign_mesh_for_leaves_options.inferTransfers = options.inferTransfers;
  assign_mesh_for_leaves_options.errorLimit = options.errorLimit;
  pm.addPass(createInferMeshAssignMeshForFuncLeavesPass(
      assign_mesh_for_leaves_options));
  pm.addPass(createInferMeshPopulateUseSetPass());

  // Validate that no additional transfers are needed.
  if (!options.inferTransfers) {
    pm.addPass(createInferMeshValidateNoAdditionalTransfersNeededPass(
        InferMeshValidateNoAdditionalTransfersNeededPassOptions{
            options.errorLimit}));
  }
  // Finally, we rewrite the DAG to assign meshes using the analyses above.
  //
  // If `inferTransfers=true`, this will introduce new cross-mesh transfers.
  // Note: currently the transfers are created as early as possible, which is
  // likely suboptimal.
  pm.addPass(createInferMeshRewriteUsingAnalysisPass(
      InferMeshRewriteUsingAnalysisPassOptions{options.maxClones}));

  pm.addPass(createInferMeshFinalizePass(
      InferMeshFinalizePassOptions{options.inferTransfers}));
}

namespace {

struct InferMeshPipelineOptions
    : public PassPipelineOptions<InferMeshPipelineOptions> {
  Option<bool> inferTransfers{
      *this, "infer-transfers",
      llvm::cl::desc(
          "Whether to create transfers when needed, instead of erroring."),
      llvm::cl::init(false)};

  Option<bool> inferCrossMeshReductions{
      *this, "infer-cross-mesh-reductions",
      llvm::cl::desc("Whether to infer cross-mesh reductions."),
      llvm::cl::init(false)};
};

}  // namespace

void registerInferMeshPipeline() {
  PassPipelineRegistration<InferMeshPipelineOptions>(
      "mpmd-infer-mesh-pipeline", "Run the passes for mesh inference",
      [](OpPassManager& pm, const InferMeshPipelineOptions& pipelineOptions) {
        InferMeshOptions options;
        options.inferTransfers = pipelineOptions.inferTransfers;
        options.inferCrossMeshReductions =
            pipelineOptions.inferCrossMeshReductions;
        addInferMeshPipeline(pm, /*inputOutputConstraints=*/{}, options);
      });
}

}  // namespace mlir::mpmd
