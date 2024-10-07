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
#include <memory>  // IWYU pragma: keep
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
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_LIFTINLINEDMESHESPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

MeshOp createNewMeshOp(Location loc, MeshAttr mesh, OpBuilder& builder) {
  auto createMesh = [&](StringRef meshName) {
    return builder.create<MeshOp>(loc, meshName, mesh);
  };
  if (std::optional<int64_t> deviceId = mesh.getMaximalDeviceId()) {
    std::string meshName = llvm::formatv("maximal_mesh_{0}", deviceId);
    return createMesh(meshName);
  }
  return createMesh("mesh");
}

TensorShardingAttr replaceMesh(TensorShardingAttr sharding,
                               StringAttr meshName) {
  return TensorShardingAttr::get(sharding.getContext(), meshName,
                                 sharding.getDimShardings(),
                                 sharding.getReplicatedAxes());
}

struct LiftInlinedMeshesPass
    : public impl::LiftInlinedMeshesPassBase<LiftInlinedMeshesPass> {
  using LiftInlinedMeshesPassBase::LiftInlinedMeshesPassBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // A map from both:
    // 1. `FlatSymbolRefAttr` referencing a `MeshOp` that was deduped.
    // 2. `MeshAttr` in an existing `MeshOp` or inlined in a
    //    `TensorShardingAttr`.
    // To the name of an existing or new `MeshOp` that should now be referenced
    // instead.
    llvm::SmallDenseMap<Attribute, StringAttr> meshOrRefToNewName;

    MeshOp lastMeshOp;
    for (auto meshOp : llvm::make_early_inc_range(moduleOp.getOps<MeshOp>())) {
      if (auto insertedIt = meshOrRefToNewName.try_emplace(
              meshOp.getMesh(), meshOp.getSymNameAttr());
          !insertedIt.second) {
        // This is a duplicate mesh, we map its name (as a FlatSymbolRefAttr) to
        // the name of the identical mesh that was already inserted, so we can
        // replace the former with the latter in any sharding in the module that
        // referenced it. We can also erase the mesh because we know it won't be
        // used after this pass.
        // NOTE: assigning to a map entry a value that is read from the same map
        // can lead to use-after-free (if rehash is triggered).
        StringAttr newMeshName = insertedIt.first->second;
        meshOrRefToNewName[FlatSymbolRefAttr::get(meshOp.getSymNameAttr())] =
            newMeshName;
        symbolTable.erase(meshOp);
      } else {
        lastMeshOp = meshOp;
      }
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
