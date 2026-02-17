/* Copyright 2026 The MPMD Authors.

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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_POPULATEUNREDUCEDOUTSHARDINGPASS
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h.inc"

namespace {

// Return the sharding of the value if it exists, otherwise return fully closed
// shardings for the value.
sdy::TensorShardingAttr getOrCreateFullyClosedSharding(Value value,
                                                       StringRef mesh_name) {
  if (sdy::TensorShardingAttr sharding = sdy::getSharding(value)) {
    return sharding;
  }
  // Shardy seems to set closed shardings by default if no sharding is specified
  // on a Value, so we do the same here.
  return sdy::getFullyClosedShardings(value.getContext(), value.getType(),
                                      mesh_name)
      .front();
}

class PopulateUnreducedOutShardingPass
    : public impl::PopulateUnreducedOutShardingPassBase<
          PopulateUnreducedOutShardingPass> {
  using PopulateUnreducedOutShardingPassBase::
      PopulateUnreducedOutShardingPassBase;

 protected:
  void runOnFunc(func::FuncOp func) override {
    func.walk([&](FragmentOp fragment_op) {
      auto terminator = cast<ReturnOp>(fragment_op.getBody()->getTerminator());
      ValueRange yielded_values = terminator.getOperands();

      std::optional<sdy::TensorShardingPerValueAttr> out_shardings_attr =
          fragment_op.getOutShardings();
      SmallVector<sdy::TensorShardingAttr> new_out_shardings;
      bool modified = false;

      if (!out_shardings_attr) {
        new_out_shardings.reserve(fragment_op.getNumResults());
        for (Value result : fragment_op.getResults()) {
          new_out_shardings.push_back(getOrCreateFullyClosedSharding(
              result, fragment_op.getMeshName()));
        }
      } else {
        new_out_shardings = llvm::to_vector(out_shardings_attr->getShardings());
      }

      for (auto [i, yielded_value] : llvm::enumerate(yielded_values)) {
        if (auto constraint_op =
                yielded_value.getDefiningOp<sdy::ShardingConstraintOp>()) {
          sdy::TensorShardingAttr constraint_sharding =
              constraint_op.getSharding();
          if (!constraint_sharding.getUnreducedAxes().empty()) {
            sdy::TensorShardingAttr current_sharding = new_out_shardings[i];
            if (!current_sharding) {
              current_sharding = getOrCreateFullyClosedSharding(
                  fragment_op.getResult(i), fragment_op.getMeshName());
            }
            if (current_sharding.getUnreducedAxes() !=
                constraint_sharding.getUnreducedAxes()) {
              new_out_shardings[i] = current_sharding.replaceUnreducedAxes(
                  constraint_sharding.getUnreducedAxes());
              modified = true;
            }
          }
        }
      }

      if (modified) {
        fragment_op.setOutShardingsAttr(sdy::TensorShardingPerValueAttr::get(
            fragment_op.getContext(), new_out_shardings));
      }
    });
  }
};

}  // namespace
}  // namespace mlir::mpmd
