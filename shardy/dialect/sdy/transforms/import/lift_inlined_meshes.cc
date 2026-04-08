/* Copyright 2024 The Shardy Authors.

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

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_LIFTINLINEDMESHESPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

MeshOp createNewMeshOp(Location loc, MeshAttr mesh, OpBuilder& builder) {
  auto createMesh = [&](StringRef meshName) {
    return MeshOp::create(builder, loc, meshName, mesh);
  };
  if (std::optional<int64_t> deviceId = mesh.getMaximalDeviceId()) {
    std::string meshName = llvm::formatv("maximal_mesh_{0}", deviceId);
    return createMesh(meshName);
  }
  if (mesh.empty()) {
    return createMesh(sdy::kEmptyMeshSymbol);
  }
  return createMesh("mesh");
}

TensorShardingAttr replaceMesh(TensorShardingAttr sharding,
                               StringAttr meshName) {
  return TensorShardingAttr::get(
      sharding.getContext(), meshName, sharding.getDimShardings(),
      sharding.getReplicatedAxes(), sharding.getUnreducedAxes());
}

struct LiftInlinedMeshesPass
    : public impl::LiftInlinedMeshesPassBase<LiftInlinedMeshesPass> {
  using LiftInlinedMeshesPassBase::LiftInlinedMeshesPassBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // A map from a `MeshAttr` in an existing `MeshOp` or inlined in a
    // `TensorShardingAttr`, to the name of an existing or new `MeshOp` that
    // should now be referenced instead.
    llvm::SmallDenseMap<Attribute, StringAttr> meshOrRefToNewName;

    MeshOp lastMeshOp;
    for (auto meshOp : llvm::make_early_inc_range(moduleOp.getOps<MeshOp>())) {
      meshOrRefToNewName.try_emplace(meshOp.getMesh(), meshOp.getSymNameAttr());
      lastMeshOp = meshOp;
    }

    OpBuilder builder(moduleOp);
    if (lastMeshOp) {
      builder.setInsertionPointAfter(lastMeshOp);
    } else {
      builder.setInsertionPointToStart(moduleOp.getBody());
    }

    transformShardings(moduleOp, [&](TensorShardingAttr sharding) {
      // Taking `newMeshName` by reference so we can update it if `sharding`
      // has an inlined mesh that isn't already in the map.
      StringAttr& newMeshName = meshOrRefToNewName[sharding.getMeshOrRef()];
      if (newMeshName) {
        return replaceMesh(sharding, newMeshName);
      }
      if (auto mesh = dyn_cast<MeshAttr>(sharding.getMeshOrRef())) {
        // Inlined mesh with a new `MeshAttr`.
        // TODO(tomnatan): give better names for meshes with device IDs, e.g.,
        // `@some_mesh_arbitrary_device_order` when there is an identical
        // `@some_mesh` without device IDs.
        newMeshName = symbolTable.insert(
            createNewMeshOp(moduleOp.getLoc(), mesh, builder));
        return replaceMesh(sharding, newMeshName);
      }
      return sharding;
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
