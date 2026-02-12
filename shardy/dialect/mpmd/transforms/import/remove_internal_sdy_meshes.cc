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

#include <variant>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"                // IWYU pragma: keep
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_REMOVEINTERNALSDYMESHESPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

// Returns the mesh name from a MeshTensorType value, or empty string if the
// value is not a MeshTensorType.
StringRef getMeshNameFromValue(Value value) {
  if (auto mesh_tensor = dyn_cast<MeshTensorType>(value.getType())) {
    return mesh_tensor.getMeshName();
  }
  return {};
}

// Returns the mesh name from the enclosing FragmentOp, or empty string if
// the op is not inside a fragment.
StringRef getEnclosingFragmentMeshName(Value value) {
  if (Operation* defining_op = value.getDefiningOp()) {
    if (auto fragment = sdy::getEnclosingOfType<FragmentOp>(defining_op)) {
      return fragment.getMeshName();
    }
  } else if (auto block_arg = dyn_cast<BlockArgument>(value)) {
    if (auto fragment =
            dyn_cast<FragmentOp>(block_arg.getOwner()->getParentOp())) {
      return fragment.getMeshName();
    }
  }
  return {};
}

// Determines the topology mesh name for a value. First tries MeshTensorType,
// then falls back to the enclosing fragment's mesh name.
StringRef getTopologyMeshName(const sdy::ValueOrFuncResult& value_or_result) {
  if (auto* value = std::get_if<Value>(&value_or_result)) {
    StringRef mesh_name = getMeshNameFromValue(*value);
    if (!mesh_name.empty()) return mesh_name;
    return getEnclosingFragmentMeshName(*value);
  }
  // For function results, get the mesh name from the result type.
  auto [func_op, res_num] = std::get<sdy::FuncResult>(value_or_result);
  if (auto mesh_tensor =
          dyn_cast<MeshTensorType>(func_op.getResultTypes()[res_num])) {
    return mesh_tensor.getMeshName();
  }
  return {};
}

struct RemoveInternalSdyMeshesPass
    : public impl::RemoveInternalSdyMeshesPassBase<
          RemoveInternalSdyMeshesPass> {
  using RemoveInternalSdyMeshesPassBase::RemoveInternalSdyMeshesPassBase;

 protected:
  void runOnOperation() final {
    ModuleOp module_op = getOperation();

    // Rewrite shardings referencing __-prefixed meshes to reference the
    // corresponding topology mesh, determined from the MeshTensorType of the
    // associated value or the enclosing fragment's mesh attribute.
    sdy::transformShardings(
        module_op,
        [&](sdy::TensorShardingAttr sharding,
            const sdy::ValueOrFuncResult& value_or_result) {
          StringRef mesh_name = sharding.getMeshName();
          if (!mesh_name.starts_with("__")) {
            return sharding;
          }
          StringRef topology_mesh = getTopologyMeshName(value_or_result);
          if (topology_mesh.empty()) {
            return sharding;
          }
          return sdy::TensorShardingAttr::get(
              sharding.getContext(), topology_mesh,
              sharding.getDimShardings(), sharding.getReplicatedAxes(),
              sharding.getUnreducedAxes());
        });

    // Remove __-prefixed meshes.
    SmallVector<sdy::MeshOp> meshes_to_remove;
    for (auto mesh_op : module_op.getOps<sdy::MeshOp>()) {
      if (mesh_op.getName().starts_with("__")) {
        meshes_to_remove.push_back(mesh_op);
      }
    }

    for (sdy::MeshOp mesh_op : meshes_to_remove) {
      mesh_op.erase();
    }
  }
};

}  // namespace

}  // namespace mlir::mpmd
