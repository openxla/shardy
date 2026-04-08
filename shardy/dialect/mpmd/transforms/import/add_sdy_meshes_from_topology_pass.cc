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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_ADDSDYMESHESFROMTOPOLOGYPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

class AddSdyMeshesFromTopologyPass
    : public impl::AddSdyMeshesFromTopologyPassBase<
          AddSdyMeshesFromTopologyPass> {
  using AddSdyMeshesFromTopologyPassBase::AddSdyMeshesFromTopologyPassBase;

 protected:
  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    TopologyAttr topology = GetTopology(module_op);
    SDY_CHECK(topology) << "No topology attribute found";

    OpBuilder builder(module_op);
    builder.setInsertionPointToStart(module_op.getBody());
    for (NamedMeshAttr named_mesh_attr : topology.getMeshes()) {
      SDY_CHECK(!module_op.lookupSymbol<sdy::MeshOp>(named_mesh_attr.getName()))
          << "Mesh " << named_mesh_attr.getName().str() << " already exists.";
      sdy::MeshOp::create(builder, module_op.getLoc(),
                          named_mesh_attr.getName(), named_mesh_attr.getMesh());
    }
  }
};

}  // namespace

}  // namespace mlir::mpmd
