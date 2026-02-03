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

#include <cstddef>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Threading.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/sharding_propagation/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_ENFORCEUSERSHARDINGSPASS
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h.inc"

namespace {

// If the sharding has sub axis in a dimension, replace it with the prefix of
// the sharding axes until the first sub axis. This is necessary because we use
// jax.sharding.NamedSharding for input/output shardings, which doesn't support
// sub axes.
// TODO: b/414535722 - Remove this temporary solution once we support sub axes
// shardings.
sdy::TensorShardingAttr MaybeDropSubAxesShardingSuffix(
    sdy::TensorShardingAttr sharding) {
  SmallVector<sdy::DimensionShardingAttr> dim_shardings;
  dim_shardings.reserve(sharding.getRank());
  for (sdy::DimensionShardingAttr dim_sharding : sharding.getDimShardings()) {
    size_t prefix_to_keep = 0;
    for (const auto& [i, axis] : llvm::enumerate(dim_sharding.getAxes())) {
      if (axis.getSubAxisInfo()) {
        break;
      }
      ++prefix_to_keep;
    }
    if (prefix_to_keep != dim_sharding.getAxes().size()) {
      dim_shardings.push_back(sdy::DimensionShardingAttr::get(
          sharding.getContext(),
          dim_sharding.getAxes().take_front(prefix_to_keep),
          /*is_closed=*/true));
    } else {
      dim_shardings.push_back(dim_sharding);
    }
  }
  return sdy::TensorShardingAttr::get(
      sharding.getContext(), sharding.getMeshOrRef(), dim_shardings,
      sharding.getReplicatedAxes(), sharding.getUnreducedAxes());
}

void emitSubAxisWarning(llvm::once_flag& once_flag, StringRef value_name,
                        unsigned value_index,
                        sdy::TensorShardingAttr old_sharding,
                        sdy::TensorShardingAttr user_sharding, Location loc) {
  llvm::call_once(once_flag, [=]() {
    emitWarning(loc, "Sub-axes sharding found for ")
        << value_name << " " << value_index
        << ". This is not supported, and sharding will be tuncated. "
           "Original sharding: "
        << old_sharding << "; truncated sharding: " << user_sharding
        << " Please contact the MPMD team.";
  });
}

// Enforces the user specified sharding for the given block argument.
// - If the block argument is used by a fragment: e.g.,
// fragment<mesh="m", origin=[], in_shardings=[<@mesh, [{"x"},
// {?}]>]> (%arg0)
// If the user specified sharding is different from the in_shardings for the
// argument, we update the sharding of the fragment input to be the user
// specified one.
// - If the block argument is used by a transfer op, we set the sharding of
// the result of the transfer op to be the user specified one.
void EnforceUserInputSharding(BlockArgument arg) {
  sdy::TensorShardingAttr old_sharding = sdy::getSharding(arg);
  if (!old_sharding) {
    return;
  }

  sdy::TensorShardingAttr user_sharding =
      MaybeDropSubAxesShardingSuffix(old_sharding);
  if (user_sharding != old_sharding) {
    sdy::setSharding(arg, user_sharding);
    static llvm::once_flag log_arg_with_sub_axes_once;
    emitSubAxisWarning(log_arg_with_sub_axes_once, "arg", arg.getArgNumber(),
                       old_sharding, user_sharding, arg.getLoc());
  }

  for (OpOperand& use : arg.getUses()) {
    if (auto fragment_user = dyn_cast<FragmentOp>(use.getOwner())) {
      unsigned operand_num = use.getOperandNumber();
      SmallVector<sdy::TensorShardingAttr> all_in_shardings =
          fragment_user.getBlockArgumentEdgeOwnerShardings();
      if (all_in_shardings.empty()) {
        all_in_shardings = SmallVector<sdy::TensorShardingAttr>(
            fragment_user->getNumOperands(), sdy::TensorShardingAttr());
      }
      sdy::TensorShardingAttr fragment_in_sharding =
          all_in_shardings[operand_num];
      if (fragment_in_sharding != user_sharding) {
        all_in_shardings[operand_num] = user_sharding;
        fragment_user.setBlockArgumentEdgeOwnerShardings(all_in_shardings);
      }
    }
    // We don't need to handle the case where the argument is used by a transfer
    // op because the later `ExtractReshardFromInterMeshTransfersPass` will
    // introduce a reshard fragment to handle the case where the user-specified
    // argument sharding is different from the transfer op result sharding.
  }
}

// Enforces the user specified sharding for the given return operand.
void EnforceUserResultSharding(OpOperand& return_operand, func::FuncOp func) {
  sdy::TensorShardingAttr old_sharding =
      sdy::getFuncResultSharding(func, return_operand.getOperandNumber());
  if (!old_sharding) {
    return;
  }
  sdy::TensorShardingAttr user_sharding =
      MaybeDropSubAxesShardingSuffix(old_sharding);
  if (user_sharding != old_sharding) {
    sdy::setSharding(return_operand.get(), user_sharding);

    static llvm::once_flag log_result_with_sub_axes_once;
    emitSubAxisWarning(log_result_with_sub_axes_once, "result",
                       return_operand.getOperandNumber(), old_sharding,
                       user_sharding, sdy::getBodyTerminator(func)->getLoc());
  }

  Value result = return_operand.get();
  if (FragmentOp defining_fragment = result.getDefiningOp<FragmentOp>()) {
    // They could be the same type but note but with different open/closed dims.
    // For now, we call the pass to close shardings, but this needs cleaning up.
    // There is no need to update fragment users shardings of this
    // `defining_fragment` like we do for transfers below because propagation
    // have taken care of it.
    defining_fragment.setUserSpecifiedResultSharding(
        cast<OpResult>(return_operand.get()).getResultNumber(), user_sharding);
    UpdateValueUserInShardings(result, user_sharding);
  } else if (auto transfer_op = result.getDefiningOp<TransferOp>()) {
    sdy::setSharding(transfer_op.getResult(), user_sharding);
    UpdateValueUserInShardings(result, user_sharding);
  }
}

class EnforceUserShardingsPass
    : public impl::EnforceUserShardingsPassBase<EnforceUserShardingsPass> {
  using EnforceUserShardingsPassBase::EnforceUserShardingsPassBase;

 private:
  void runOnFunc(func::FuncOp func) final {
    if (!IsEntryPointFunction(func) || !IsMpmdFunction(func)) {
      return;
    }

    for (BlockArgument arg : func.getArguments()) {
      EnforceUserInputSharding(arg);
    }

    for (OpOperand& return_operand :
         func.front().getTerminator()->getOpOperands()) {
      EnforceUserResultSharding(return_operand, func);
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
