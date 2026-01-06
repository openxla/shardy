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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"

using ::mlir::func::FuncOp;

namespace mlir::mpmd {

#define GEN_PASS_DEF_COPYTOPOLOGYFROMMAINPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

// If the module doesn't have an `sdy.mesh`, constructs a global mesh containing
// all the axes in the main function's topology. Assumes all meshes in the
// topology are homogeneous.
void MaybeConstructSdyMesh(ModuleOp module_op) {
  if (!module_op.getOps<sdy::MeshOp>().empty()) {
    return;
  }

  FuncOp main_func = GetMainFunction(module_op);
  SDY_CHECK(HasHomogeneousTopology(main_func));

  // Create a global mesh containing all the axes.
  auto current_topology_attr =
      main_func->getAttrOfType<TopologyAttr>(kTopologyAttr);
  SDY_CHECK(current_topology_attr);
  sdy::MeshAttr named_mesh =
      current_topology_attr.getMeshes().front().getMesh();

  MLIRContext* ctx = module_op->getContext();
  SmallVector<sdy::MeshAxisAttr> sdy_axes;
  sdy_axes.reserve(named_mesh.getAxes().size());
  for (sdy::MeshAxisAttr mesh_axis : named_mesh.getAxes()) {
    sdy_axes.push_back(
        sdy::MeshAxisAttr::get(ctx, mesh_axis.getName(), mesh_axis.getSize()));
  }

  OpBuilder::atBlockBegin(module_op.getBody())
      .create<sdy::MeshOp>(module_op.getLoc(), kGlobalMeshName,
                           sdy::MeshAttr::get(ctx, sdy_axes));
}

class CopyTopologyFromMainPass
    : public impl::CopyTopologyFromMainPassBase<CopyTopologyFromMainPass> {
  using CopyTopologyFromMainPassBase::CopyTopologyFromMainPassBase;

  void runOnOperation() final {
    ModuleOp module_op = getOperation();

    // TODO(b/428336749): remove gspmd specific logic when no longer needed.
    // MaybeConstructSdyMesh(module_op);
    SDY_CHECK(!module_op.getOps<sdy::MeshOp>().empty());

    for (FuncOp func_op : module_op.getOps<FuncOp>()) {
      auto main_topology_attr =
          func_op->getAttrOfType<TopologyAttr>(kTopologyAttr);
      if (!main_topology_attr) continue;

      func_op.walk([main_topology_attr](CallOp call_op) {
        FuncOp callee = GetCalleeFunc(call_op);
        callee->setAttr(kTopologyAttr, main_topology_attr);
        // Set the callee to private to mark that it's not an entry point
        // function.
        callee.setPrivate();
      });
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
