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

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"                // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_REMOVEINTERNALSDYMESHESPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

struct RemoveInternalSdyMeshesPass
    : public impl::RemoveInternalSdyMeshesPassBase<
          RemoveInternalSdyMeshesPass> {
  using RemoveInternalSdyMeshesPassBase::RemoveInternalSdyMeshesPassBase;

 protected:
  void runOnOperation() final {
    ModuleOp module_op = getOperation();
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
