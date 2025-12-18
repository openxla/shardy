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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"

using ::mlir::func::FuncOp;

namespace mlir::mpmd {

#define GEN_PASS_DEF_INTRODUCETRANSFERSPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

// Replaces the AssignOp of an UnassignOp with a TransferOp, or a noop if the
// transfer is not needed. It reuses existing transfers if possible. For a given
// value, this create at most one transfer of that value to a given mesh.
class AssignOfUnassignPattern : public OpRewritePattern<AssignOp> {
  using OpRewritePattern<AssignOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(AssignOp op,
                                PatternRewriter& rewriter) const override {
    auto unassign_op = op.getTensor().getDefiningOp<UnassignOp>();
    if (!unassign_op) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "Expected the operand of the AssignOp to be the result of an "
                "UnassignOp";
      });
    }

    Type target_type = op.getType();
    TypedValue<MeshTensorType> op_to_transfer = unassign_op.getTensor();
    if (op_to_transfer.getType() == target_type) {
      // Noop transfer, so just replace with tensor.
      rewriter.replaceOp(op, op_to_transfer);
      return success();
    }

    auto existing_transfer_it =
        llvm::find_if(op_to_transfer.getUsers(), [target_type](Operation* op) {
          if (auto trf = DynCastInterMeshTransfer(op)) {
            return trf.getType() == target_type;
          }
          return false;
        });
    if (existing_transfer_it != op_to_transfer.getUsers().end()) {
      // We create the transfer at the user location, but we don't necessarily
      // iterate the users in program order. So the transfer might be created on
      // a later user, and comes after the current user, and we need to move it
      // before the current user. Note that we could also create the transfer
      // right after the transfer operand's defining op, but we don't for now
      // to preserve existing semantics.
      if (op->isBeforeInBlock(*existing_transfer_it)) {
        existing_transfer_it->moveBefore(op);
      }
      rewriter.replaceOp(op, existing_transfer_it->getResults());
    } else {
      rewriter.replaceOpWithNewOp<TransferOp>(op, op.getType(),
                                              unassign_op.getTensor());
    }
    return success();
  }
};

// This checks if all `stablehlo.add` operands are UnassignOps, for additions of
// two or more values. Because AddOp is a binary op, this means that we have to
// check recursive adds. E.g., handle something like
//
// u_{i} = unassign(...)
// x = stablehlo.add u_0, u_1
// y = stablehlo.add x, u_2
// z = stablehlo.add y, u_3
//
// which can also be expressed as (u_0 + u_1 + u_2 + u_3).
//
// For simplicity, we only handle the case where the entire addition chain has a
// single user.
//
// This method is recursive, but the depth shouldn't be excessive since it is
// unlikely that we have a cross-mesh reduction with many operands.
LogicalResult IsAddOfUnassigns(stablehlo::AddOp add, Operation* user,
                               PatternRewriter& rewriter) {
  if (!add->hasOneUse()) {
    return rewriter.notifyMatchFailure(user, [&](Diagnostic& diag) {
      diag << "Expected AddOp to have only one user";
    });
  }

  for (Value operand : add->getOperands()) {
    if (auto unassign = operand.getDefiningOp<UnassignOp>()) {
      continue;
    }
    if (auto nested_add = operand.getDefiningOp<stablehlo::AddOp>()) {
      if (failed(IsAddOfUnassigns(nested_add, add, rewriter))) {
        return failure();
      }
      continue;
    }
    return rewriter.notifyMatchFailure(user, [&](Diagnostic& diag) {
      diag << "Expected all operands of " << user
           << " to be an UnassignOp or AddOp";
    });
  }
  return success();
}

// When we have a meshless addition between fragments, we want to assign the
// addition to the consuming mesh and introduce a transfer. E.g.,
//
// x = unassign(x') from some_mesh1
// y = unassign(y') from some_mesh2
// z = x + y
// z' = assign(z) to m1
//
// ~~>
//
// x = unassign(x')
// y = unassign(y')
// x'' = assign(x) to m1
// y'' = assign(y) to m1
// z' = fragment m1 {
//    return x'' + y''
// }
//
// More generally, we need to handle the case of adding more than two values:
// assign(\sum{0..n} unassign(x_i)) ~~> \sum{0..n}(assign(unassign(x_i))).
//
// This pass does the assignment of the addition, and the transfer pattern will
// create the transfer.
//
// We do this because in the backward pass of a model (e.g. via jax.grad),
// a meshless add may appear in this way, and we want to transfer the operands
// to the consuming mesh for the addition to be done.
//
// For simplicity, we handle only the case where the addition has a single user.
//
// Note: we could support other operators, but at present there's only a use
// case for the AddOp.
class PushAssignBackwardThroughAdd : public OpRewritePattern<AssignOp> {
  using OpRewritePattern<AssignOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(AssignOp assign,
                                PatternRewriter& rewriter) const override {
    auto meshless_add = assign.getTensor().getDefiningOp<stablehlo::AddOp>();
    if (!meshless_add) {
      return rewriter.notifyMatchFailure(assign, [&](Diagnostic& diag) {
        diag << "Expected the operand of AssignOp to be an AddOp";
      });
    }
    if (failed(IsAddOfUnassigns(meshless_add, assign, rewriter))) {
      return failure();
    }
    WrapOpWithFragment(meshless_add, assign.getType().getMeshName(), rewriter);
    return success();
  }
};

// Returns the UnassignOp that defines the given value, or nullptr if the value
// is not defined by an UnassignOp. Walks through CallOps to find the UnassignOp
// in the callee, if necessary.
UnassignOp FindUnassignOp(Value value) {
  if (auto call = value.getDefiningOp<CallOp>()) {
    FuncOp callee = GetCalleeFunc(call);
    auto value_in_callee = callee.front().getTerminator()->getOperand(
        cast<OpResult>(value).getResultNumber());
    return value_in_callee.getDefiningOp<UnassignOp>();
  }

  return value.getDefiningOp<UnassignOp>();
}

// Returns true if the callee function with sources as its callers that use
// arg can push an unassign op in the callee.
bool ShouldPushInUnassignOp(ArrayRef<OpOperand*> sources, Value arg) {
  SDY_CHECK(!sources.empty());

  // If not all of the sources are from unassign op or the meshes are
  // different we won't push in unassign op.
  UnassignOp first_source_defining_op = FindUnassignOp(sources.front()->get());
  if (!first_source_defining_op) {
    return false;
  }
  StringRef first_source_mesh_name =
      first_source_defining_op.getTensor().getType().getMeshName();

  for (OpOperand* source : ArrayRef<OpOperand*>(sources).drop_front()) {
    UnassignOp defining_op = FindUnassignOp(source->get());
    if (!defining_op) {
      return false;
    }
    if (defining_op.getTensor().getType().getMeshName() !=
        first_source_mesh_name) {
      return false;
    }
  }

  // If the argument is used by any assign op in the callee, push in unassign
  // op.
  return llvm::any_of(arg.getUses(), [](const OpOperand& use) {
    return isa<AssignOp>(use.getOwner());
  });
}

// Pushes in unassign ops in the callee if possible.
void MaybePushInUnassignOp(FuncOp callee, IRRewriter& rewriter) {
  SmallVector<MpmdDataflowEdge> edges = GetMpmdDataflowEdgesForFuncArgs(callee);
  // We go through each edge and check if the unassign op can be pushed into
  // the callee. If so, we push them in and remove the unassign op in the
  // caller if it is not used anymore.
  // Each edge (sources, targets) pair is for one of the callee arguments.
  // sources represents the inputs to the callee from different calls and
  // targets represents the outputs of the callee.
  for (auto& [sources, targets] : edges) {
    BlockArgument callee_arg = cast<BlockArgument>(targets.front());
    if (!ShouldPushInUnassignOp(sources, callee_arg)) {
      continue;
    }

    rewriter.setInsertionPointAfterValue(callee_arg);
    auto callee_assign = AssignOp::create(
        rewriter, callee_arg.getLoc(),
        FindUnassignOp(sources.front()->get()).getTensor().getType(),
        callee_arg);
    auto callee_unassign =
        UnassignOp::create(rewriter, callee_arg.getLoc(), callee_assign);
    rewriter.replaceAllUsesExcept(callee_arg, callee_unassign.getResult(),
                                  callee_assign);
  }
}

class IntroduceTransfersPass
    : public impl::IntroduceTransfersPassBase<IntroduceTransfersPass> {
  using IntroduceTransfersPassBase::IntroduceTransfersPassBase;

  LogicalResult initialize(MLIRContext* context) final {
    RewritePatternSet patternsInternal(context);
    patternsInternal.add<AssignOfUnassignPattern, PushAssignBackwardThroughAdd>(
        context);
    patterns = std::move(patternsInternal);

    return success();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    for (FuncOp func : GetMpmdFunctions(module)) {
      if (!IsEntryPointFunction(func)) {
        IRRewriter rewriter(func->getContext());
        MaybePushInUnassignOp(func, rewriter);
      }
    }
    GreedyRewriteConfig config =
        GreedyRewriteConfig()
            .setRegionSimplificationLevel(
                mlir::GreedySimplifyRegionLevel::Disabled)
            .enableFolding(false)
            .enableConstantCSE(false);
    if (failed(applyPatternsGreedily(module, patterns, config))) {
      return signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
};

}  // namespace

}  // namespace mlir::mpmd
