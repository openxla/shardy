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
#include "llvm/Support/Signals.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"

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

MeshOp createNewMeshOp(Location loc, mlir::stablehlo::MeshAttr mesh,
                       OpBuilder& builder) {
  auto createMesh = [&](StringRef meshName, MeshAttr sdyMesh) {
    return MeshOp::create(builder, loc, meshName, sdyMesh);
  };
  SmallVector<MeshAxisAttr> sdyAxes;
  for (auto axisAttr : mesh.getAxes()) {
    auto shloMeshAxis = axisAttr;
    if (!shloMeshAxis) continue;
    sdyAxes.push_back(MeshAxisAttr::get(
        mesh.getContext(), shloMeshAxis.getName(), shloMeshAxis.getSize()));
  }
  auto sdyMeshAttr = MeshAttr::get(mesh.getContext(), sdyAxes);
  return createMesh("mesh", sdyMeshAttr);
}

DictionaryAttr getStablehloMeshAttrAsDict(MeshAttr sdyMeshAttr) {
  MLIRContext* ctx = sdyMeshAttr.getContext();
  Builder builder(ctx);

  SmallVector<Attribute> axesAttrs;
  for (MeshAxisAttr axisAttr : sdyMeshAttr.getAxes()) {
    NamedAttribute nameAttr =
        builder.getNamedAttr("name", builder.getStringAttr(axisAttr.getName()));
    NamedAttribute sizeAttr = builder.getNamedAttr(
        "size", builder.getI64IntegerAttr(axisAttr.getSize()));
    axesAttrs.push_back(builder.getDictionaryAttr({nameAttr, sizeAttr}));
  }
  ArrayAttr axesArrayAttr = builder.getArrayAttr(axesAttrs);

  SmallVector<NamedAttribute> dictFields;
  dictFields.push_back(builder.getNamedAttr("axes", axesArrayAttr));

  if (!sdyMeshAttr.getDeviceIds().empty()) {
    auto type = RankedTensorType::get(
        {static_cast<int64_t>(sdyMeshAttr.getDeviceIds().size())},
        builder.getI64Type());
    auto deviceIds =
        DenseIntElementsAttr::get(type, sdyMeshAttr.getDeviceIds());
    dictFields.push_back(builder.getNamedAttr("device_ids", deviceIds));
  }

  return builder.getDictionaryAttr(dictFields);
}

TensorShardingAttr replaceMesh(TensorShardingAttr sharding,
                               StringAttr meshName) {
  return TensorShardingAttr::get(
      sharding.getContext(), FlatSymbolRefAttr::get(meshName),
      sharding.getDimShardings(), sharding.getReplicatedAxes(),
      sharding.getUnreducedAxes());
}

template <typename ReplicaGroupMeshAxesAttrTy>
Attribute liftMeshInReplicaGroups(
    Attribute attr, SymbolTable& symbolTable,
    llvm::SmallDenseMap<Attribute, StringAttr>& meshOrRefToNewName,
    OpBuilder& builder, Location loc) {
  if (auto replicaGroupsAttr =
          mlir::dyn_cast<ReplicaGroupMeshAxesAttrTy>(attr)) {
    Attribute meshOrRef = replicaGroupsAttr.getMesh();
    StringAttr& newMeshName = meshOrRefToNewName[meshOrRef];
    if (newMeshName) {
      return ReplicaGroupMeshAxesAttrTy::get(
          replicaGroupsAttr.getContext(), FlatSymbolRefAttr::get(newMeshName),
          replicaGroupsAttr.getAxes());
    }
    if (auto mesh = mlir::dyn_cast<MeshAttr>(meshOrRef)) {
      newMeshName = symbolTable.insert(createNewMeshOp(loc, mesh, builder));
      return ReplicaGroupMeshAxesAttrTy::get(
          replicaGroupsAttr.getContext(), FlatSymbolRefAttr::get(newMeshName),
          replicaGroupsAttr.getAxes());
    }
    if (auto shloMesh = mlir::dyn_cast<mlir::stablehlo::MeshAttr>(meshOrRef)) {
      newMeshName = symbolTable.insert(createNewMeshOp(loc, shloMesh, builder));
      return ReplicaGroupMeshAxesAttrTy::get(
          replicaGroupsAttr.getContext(), FlatSymbolRefAttr::get(newMeshName),
          replicaGroupsAttr.getAxes());
    }
  }
  return attr;
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

    auto processMeshInReplicaGroups = [&](auto op) {
      if (auto attr = op->getAttr("replica_groups")) {
        auto newAttr =
            liftMeshInReplicaGroups<mlir::stablehlo::ReplicaGroupMeshAxesAttr>(
                attr, symbolTable, meshOrRefToNewName, builder, op->getLoc());

        if (newAttr != attr) {
          op->setAttr("replica_groups", newAttr);
        }
      }
    };

    moduleOp.walk(
        [&](stablehlo::AllGatherOp op) { processMeshInReplicaGroups(op); });
    moduleOp.walk(
        [&](stablehlo::AllReduceOp op) { processMeshInReplicaGroups(op); });
    moduleOp.walk(
        [&](stablehlo::ReduceScatterOp op) { processMeshInReplicaGroups(op); });
    moduleOp.walk(
        [&](stablehlo::AllToAllOp op) { processMeshInReplicaGroups(op); });
    moduleOp.walk([&](stablehlo::CollectiveBroadcastOp op) {
      processMeshInReplicaGroups(op);
    });

    // Attach discardable `stablehlo.mesh` attributes to all named meshes.
    for (auto meshOp : llvm::make_early_inc_range(moduleOp.getOps<MeshOp>())) {
      if (!meshOp->hasAttr("stablehlo.mesh")) {
        meshOp->setAttr("stablehlo.mesh",
                        getStablehloMeshAttrAsDict(meshOp.getMesh()));
      }
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
