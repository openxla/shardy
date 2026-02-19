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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_VALIDATENOBACKWARDDEPSPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

bool IsAllForward(func::FuncOp func) {
  bool is_all_forward = true;
  func.walk([&](FragmentOp fragment) {
    for (Attribute attr : fragment.getOrigin().getValue()) {
      if (cast<UserOriginAttr>(attr).getTransposeCount() != 0) {
        is_all_forward = false;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return is_all_forward;
}

FragmentOp FindProducerFragment(Value operand) {
  if (mlir::isa<BlockArgument>(operand)) {
    return nullptr;
  }
  Operation* op = operand.getDefiningOp();
  if (!op) {
    return nullptr;
  }
  if (auto fragment = mlir::dyn_cast<FragmentOp>(op)) {
    return fragment;
  }
  if (auto transfer = mlir::dyn_cast<TransferOp>(op)) {
    return FindProducerFragment(transfer.getOperand());
  }
  return nullptr;
}

class ValidateNoBackwardDepsPass
    : public impl::ValidateNoBackwardDepsPassBase<ValidateNoBackwardDepsPass> {
  using ValidateNoBackwardDepsPassBase::ValidateNoBackwardDepsPassBase;

 protected:
  void runOnFunc(func::FuncOp func) override {
    if (!IsAllForward(func)) {
      return;
    }

    func.walk([&](FragmentOp consumer) {
      std::optional<int64_t> consumer_stage = consumer.getStageId();
      if (!consumer_stage.has_value()) {
        return;
      }

      for (Value operand : consumer.getOperands()) {
        FragmentOp producer = FindProducerFragment(operand);
        if (!producer) {
          continue;
        }

        std::optional<int64_t> producer_stage = producer.getStageId();
        if (!producer_stage.has_value()) {
          continue;
        }

        if (*producer_stage >= *consumer_stage) {
          consumer.emitError()
              << "Detected backward dependency in forward-only program: "
              << "fragment on mesh \"" << producer.getMeshName() << "\" (stage "
              << *producer_stage
              << ") produces a value consumed by fragment on mesh \""
              << consumer.getMeshName() << "\" (stage " << *consumer_stage
              << "). In a forward-only pipeline, dependencies must go from "
              << "lower stages to higher stages (stage i -> stage j where "
              << "j > i).";
          signalPassFailure();
        }
      }
    });
  }
};

}  // namespace

}  // namespace mlir::mpmd
