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

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"                // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_RENAMEMESHESPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

class RenameMeshesPass : public impl::RenameMeshesPassBase<RenameMeshesPass> {
  using RenameMeshesPassBase::RenameMeshesPassBase;

 protected:
  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    SymbolTable symbol_table(module_op);
    OpBuilder builder(module_op);
    builder.setInsertionPointToStart(module_op.getBody());

    SmallVector<sdy::MeshOp> old_meshes;
    for (auto mesh_op : module_op.getOps<sdy::MeshOp>()) {
      old_meshes.push_back(mesh_op);
    }

    for (sdy::MeshOp mesh_op : old_meshes) {
      if (mesh_op.getMesh().empty() || mesh_op.getMesh().isMaximal()) {
        continue;
      }
      StringRef old_name = mesh_op.getName();
      std::string new_name = ("__" + old_name).str();
      sdy::MeshOp::create(builder, mesh_op.getLoc(), new_name,
                          mesh_op.getMesh());
    }

    sdy::transformShardings(module_op, [&](sdy::TensorShardingAttr sharding) {
      StringRef old_mesh_name = sharding.getMeshName();
      if (old_mesh_name.empty()) {
        return sharding;
      }
      auto mesh_op = symbol_table.lookup<sdy::MeshOp>(old_mesh_name);
      if (mesh_op.getMesh().empty() || mesh_op.getMesh().isMaximal()) {
        return sharding;
      }
      std::string new_mesh_name = ("__" + old_mesh_name).str();
      return sdy::TensorShardingAttr::get(
          sharding.getContext(), new_mesh_name, sharding.getDimShardings(),
          sharding.getReplicatedAxes(), sharding.getUnreducedAxes());
    });

    for (sdy::MeshOp mesh_op : old_meshes) {
      if (mesh_op.getMesh().empty() || mesh_op.getMesh().isMaximal()) {
        continue;
      }
      mesh_op.erase();
    }
  }
};

}  // namespace

}  // namespace mlir::mpmd
