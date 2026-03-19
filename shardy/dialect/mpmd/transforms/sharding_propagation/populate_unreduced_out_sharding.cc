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
#include "mlir/IR/Attributes.h"
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

      if (!out_shardings_attr) {
        new_out_shardings.reserve(fragment_op.getNumResults());
        for (Value result : fragment_op.getResults()) {
          // fragment_op.getMeshName() is just a placeholder here, as these
          // shardings will have their mesh rewritten to the SDY mesh before
          // they are set as attributes on the fragment op - we just don't have
          // the SDY mesh at this point yet.
          new_out_shardings.push_back(getOrCreateFullyClosedSharding(
              result, fragment_op.getMeshName()));
        }
      } else {
        new_out_shardings = llvm::to_vector(out_shardings_attr->getShardings());
      }

      // Set when out_shardings are updated with unreduced axes from a
      // constraint. Carries the SDY mesh ref (e.g. @mesh), which may differ
      // from the MPMD mesh name (e.g. "mesh1"), and is used to ensure all
      // out_shardings reference the correct SDY mesh before writing.
      Attribute updated_mesh_ref;

      for (auto [i, yielded_value] : llvm::enumerate(yielded_values)) {
        auto constraint_op =
            yielded_value.getDefiningOp<sdy::ShardingConstraintOp>();
        if (!constraint_op) {
          continue;
        }

        sdy::TensorShardingAttr constraint_sharding =
            constraint_op.getSharding();
        if (constraint_sharding.getUnreducedAxes().empty()) {
          continue;
        }

        sdy::TensorShardingAttr current_sharding = new_out_shardings[i];
        if (current_sharding && current_sharding.getUnreducedAxes() ==
                                    constraint_sharding.getUnreducedAxes()) {
          continue;
        }

        sdy::TensorShardingAttr base_sharding = current_sharding;
        if (!base_sharding) {
          base_sharding = getOrCreateFullyClosedSharding(
              fragment_op.getResult(i), constraint_sharding.getMeshName());
        }
        // Build the sharding with the SDY mesh name from the
        // constraint, preserving dim shardings and replicated axes
        // from the existing or a new fully-closed sharding.
        updated_mesh_ref = constraint_sharding.getMeshOrRef();
        new_out_shardings[i] = sdy::TensorShardingAttr::get(
            base_sharding.getContext(), updated_mesh_ref,
            base_sharding.getDimShardings(), base_sharding.getReplicatedAxes(),
            constraint_sharding.getUnreducedAxes());
      }

      if (updated_mesh_ref) {
        // Ensure all out_shardings reference the SDY mesh, not the MPMD
        // mesh which may have been used when creating default shardings.
        for (sdy::TensorShardingAttr& sharding : new_out_shardings) {
          if (sharding && sharding.getMeshOrRef() != updated_mesh_ref) {
            sharding = sdy::TensorShardingAttr::get(
                sharding.getContext(), updated_mesh_ref,
                sharding.getDimShardings(), sharding.getReplicatedAxes(),
                sharding.getUnreducedAxes());
          }
        }
        fragment_op.setOutShardingsAttr(sdy::TensorShardingPerValueAttr::get(
            fragment_op.getContext(), new_out_shardings));
      }
    });
  }
};

}  // namespace
}  // namespace mlir::mpmd
