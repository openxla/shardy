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

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"

namespace mlir::mpmd {

namespace {

sdy::MeshOp GetOrCreateEmptyMesh(ModuleOp module_op) {
  if (sdy::MeshOp empty_mesh =
          module_op.lookupSymbol<sdy::MeshOp>(sdy::kEmptyMeshSymbol)) {
    SDY_CHECK(empty_mesh.getMesh().empty())
        << "Mesh named '" << sdy::kEmptyMeshSymbol
        << "' already exists but is not empty.";
    return empty_mesh;
  }
  OpBuilder builder = OpBuilder::atBlockBegin(module_op.getBody());
  sdy::MeshOp empty_mesh =
      sdy::MeshOp::create(builder, module_op.getLoc(), sdy::kEmptyMeshSymbol,
                          sdy::MeshAttr::get(builder.getContext(), {}));

  // TODO(b/441487083): Look up the mesh in the global mesh registry.
  // Insert empty mesh into the topology attribute.
  SmallVector<func::FuncOp> mpmd_funcs = GetMpmdFunctions(module_op);
  SDY_CHECK(!mpmd_funcs.empty());
  TopologyAttr topology = GetTopology(mpmd_funcs.front());
  SmallVector<NamedMeshAttr> meshes(topology.getMeshes());
  meshes.push_back(
      NamedMeshAttr::get(module_op.getContext(), sdy::kEmptyMeshSymbol,
                         sdy::MeshAttr::get(module_op.getContext(), {})));
  mpmd_funcs.front()->removeAttr(kTopologyAttr);
  mpmd_funcs.front()->setAttr(
      kTopologyAttr, mpmd::TopologyAttr::get(module_op.getContext(), meshes));

  return empty_mesh;
}

}  // namespace

constexpr char kMeshAxisSeparator = '_';

#define GEN_PASS_DEF_GENERATESDYMESHESFROMTOPOLOGYPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

class GenerateSdyMeshesFromTopologyPass
    : public impl::GenerateSdyMeshesFromTopologyPassBase<
          GenerateSdyMeshesFromTopologyPass> {
  using GenerateSdyMeshesFromTopologyPassBase::
      GenerateSdyMeshesFromTopologyPassBase;

 private:
  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    SymbolTable symbol_table(module_op);
    TopologyAttr topology = GetTopology(module_op);
    SDY_CHECK(topology) << "No topology attribute found";

    SmallVector<StringRef> old_meshes;
    for (sdy::MeshOp mesh_op : module_op.getOps<sdy::MeshOp>()) {
      old_meshes.push_back(mesh_op.getName());
    }

    OpBuilder builder(module_op);
    builder.setInsertionPointToStart(module_op.getBody());
    for (NamedMeshAttr named_mesh_attr : topology.getMeshes()) {
      sdy::MeshOp::create(builder, module_op.getLoc(),
                          named_mesh_attr.getName(), named_mesh_attr.getMesh());
    }

    sdy::transformShardings(module_op, [&](sdy::TensorShardingAttr sharding) {
      if (sharding.isFullyReplicated()) {
        sdy::MeshOp mesh_op =
            symbol_table.lookup<sdy::MeshOp>(sharding.getMeshName());
        if (mesh_op.getMesh().empty() || mesh_op.getMesh().isMaximal()) {
          return sharding;
        }
        return sdy::TensorShardingAttr::get(
            sharding.getContext(), GetOrCreateEmptyMesh(module_op).getName(),
            sharding.getDimShardings(), sharding.getReplicatedAxes(),
            sharding.getUnreducedAxes());
      }
      StringRef mesh_name;
      SmallVector<sdy::DimensionShardingAttr> dim_shardings;
      for (auto dim_sharding : sharding.getDimShardings()) {
        SmallVector<sdy::AxisRefAttr> axes;
        for (sdy::AxisRefAttr axis : dim_sharding.getAxes()) {
          auto [prefix, axis_name] = axis.getName().split(kMeshAxisSeparator);
          SDY_CHECK(!axis_name.empty())
              << "Axis name does not contain '" << kMeshAxisSeparator << "'";
          mesh_name = prefix;
          axes.push_back(sdy::AxisRefAttr::get(
              module_op.getContext(), axis_name, axis.getSubAxisInfo()));
        }
        dim_shardings.push_back(sdy::DimensionShardingAttr::get(
            module_op.getContext(), axes, dim_sharding.getIsClosed(),
            dim_sharding.getPriority()));
      }
      SDY_CHECK(!llvm::is_contained(old_meshes, mesh_name))
          << "Invalid mesh name: " << mesh_name.str();
      // TODO(b/440336690): Add support for replicated axes and unreduced axes.
      return sdy::TensorShardingAttr::get(
          sharding.getContext(), mesh_name, dim_shardings,
          sharding.getReplicatedAxes(), sharding.getUnreducedAxes());
    });

    for (StringRef mesh_name : old_meshes) {
      // TODO(petebu): Find better way to handle empty/maximal meshes.
      auto mesh_op = symbol_table.lookup<sdy::MeshOp>(mesh_name);
      if (mesh_op.getMesh().empty() || mesh_op.getMesh().isMaximal()) {
        continue;
      }
      symbol_table.erase(symbol_table.lookup<sdy::MeshOp>(mesh_name));
    }
  }
};

}  // namespace mlir::mpmd
