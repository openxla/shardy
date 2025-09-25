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

#include <optional>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/explicit_reshards_util.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_CONVERTSDYSHARDINGSTOMPMDTYPESPASS
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h.inc"

namespace {

sdy::TensorShardingAttr GetShardingFromMeshTensorValue(Value value) {
  auto mesh_tensor_type = dyn_cast<MeshTensorType>(value.getType());
  SDY_CHECK(mesh_tensor_type);
  std::optional<sdy::TensorShardingAttr> sharding =
      mesh_tensor_type.getSharding();
  return sharding.value_or(sdy::TensorShardingAttr());
}

class ConvertSdyShardingsToMpmdTypesPass
    : public impl::ConvertSdyShardingsToMpmdTypesPassBase<
          ConvertSdyShardingsToMpmdTypesPass> {
  using ConvertSdyShardingsToMpmdTypesPassBase::
      ConvertSdyShardingsToMpmdTypesPassBase;

 private:
  void runOnFunc(func::FuncOp func_op) override {
    func_op->walk([&](Operation* op) {
      TypeSwitch<Operation*, void>(op)
          .Case<FragmentOp>([](FragmentOp fragment) {
            if (std::optional<sdy::TensorShardingPerValueAttr> in_shardings =
                    fragment.getInShardings()) {
              for (OpOperand& operand : fragment->getOpOperands()) {
                UpdateValueTypeWithSharding(
                    operand.get(),
                    in_shardings->getSharding(operand.getOperandNumber()));
              }
              // Remove the inshardings attr since it's been moved to the type.
              fragment.removeInShardingsAttr();
            }
            if (std::optional<sdy::TensorShardingPerValueAttr> out_shardings =
                    fragment.getOutShardings()) {
              for (OpResult result : fragment->getResults()) {
                UpdateValueTypeWithSharding(
                    result,
                    out_shardings->getSharding(result.getResultNumber()));
              }
              // Remove the outshardings attr since it's been moved to the type.
              fragment.removeOutShardingsAttr();
            }
          })
          .Case<TransferOp>([](TransferOp transfer) {
            UpdateValueTypeWithSharding(transfer.getTensor(),
                                        sdy::getSharding(transfer.getTensor()));
            UpdateValueTypeWithSharding(transfer.getResult(),
                                        sdy::getSharding(transfer.getResult()));
          })
          .Case<ForOp>([](ForOp for_op) {
            for (OpOperand& operand : for_op->getOpOperands()) {
              sdy::TensorShardingAttr sharding =
                  sdy::getSharding(operand.get());
              UpdateValueTypeWithSharding(operand.get(), sharding);
              UpdateValueTypeWithSharding(
                  for_op.getRegion().getArgument(operand.getOperandNumber()),
                  sharding);
            }
            for (OpResult op_result : for_op.getResults()) {
              UpdateValueTypeWithSharding(op_result,
                                          sdy::getSharding(op_result));
            }
          });
      op->removeAttr(sdy::kShardingAttr);
      return WalkResult::advance();
    });
    // Remove the sharding attribute in args and results and update the function
    // signature.
    for (unsigned arg_num = 0; arg_num < func_op.getNumArguments(); ++arg_num) {
      UpdateValueTypeWithSharding(
          func_op.getArgument(arg_num),
          sdy::getSharding(func_op.getArgument(arg_num)));
      func_op.removeArgAttr(arg_num, sdy::kShardingAttr);
    }

    for (unsigned result_num = 0; result_num < func_op.getNumResults();
         ++result_num) {
      func_op.removeResultAttr(result_num, sdy::kShardingAttr);
    }
    func_op->removeAttr(sdy::kShardingAttr);
    UpdateFunctionType(func_op);

    // Verify that all transfers have the same operand and result sharding.
    func_op->walk([](TransferOp transfer) {
      sdy::TensorShardingAttr operand_sharding =
          GetShardingFromMeshTensorValue(transfer.getTensor());
      sdy::TensorShardingAttr result_sharding =
          GetShardingFromMeshTensorValue(transfer.getResult());

      // TODO(petebu): Add check for TransferOp between heterogeneous meshes.
      if (operand_sharding && result_sharding &&
          operand_sharding.getMeshName() == result_sharding.getMeshName() &&
          !sdy::isEquivalent(operand_sharding, result_sharding)) {
        transfer->emitError()
            << "Transfer op has different shardings for the operand and result "
               "on the same mesh, operand sharding: "
            << operand_sharding << ", result sharding: " << result_sharding;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};

}  // namespace
}  // namespace mlir::mpmd
