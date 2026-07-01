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

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_INLINEMESHESPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

struct InlineMeshesPass : public impl::InlineMeshesPassBase<InlineMeshesPass> {
  using InlineMeshesPassBase::InlineMeshesPassBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    transformShardings(moduleOp, [&](TensorShardingAttr sharding) {
      return inlineMesh(symbolTable, sharding);
    });

    moduleOp.walk([&](Operation* op) {
      if (auto attr = op->getAttr("replica_groups")) {
        if (auto replicaGroupsAttr =
                mlir::dyn_cast<mlir::stablehlo::ReplicaGroupMeshAxesAttr>(
                    attr)) {
          if (auto symbolRef = mlir::dyn_cast<FlatSymbolRefAttr>(
                  replicaGroupsAttr.getMesh())) {
            MeshAttr sdyMesh = getMeshAttr(symbolTable, symbolRef);
            if (sdyMesh) {
              SmallVector<mlir::stablehlo::MeshAxisAttr> shloAxes;
              for (MeshAxisAttr axisAttr : sdyMesh.getAxes()) {
                shloAxes.push_back(mlir::stablehlo::MeshAxisAttr::get(
                    axisAttr.getContext(), axisAttr.getName(),
                    axisAttr.getSize()));
              }
              DenseIntElementsAttr deviceIds;
              if (!sdyMesh.getDeviceIds().empty()) {
                auto type = RankedTensorType::get(
                    {static_cast<int64_t>(sdyMesh.getDeviceIds().size())},
                    Builder(sdyMesh.getContext()).getI64Type());
                deviceIds =
                    DenseIntElementsAttr::get(type, sdyMesh.getDeviceIds());
              }
              auto shloMeshAttr = mlir::stablehlo::MeshAttr::get(
                  sdyMesh.getContext(), shloAxes, deviceIds);
              op->setAttr("replica_groups",
                          mlir::stablehlo::ReplicaGroupMeshAxesAttr::get(
                              replicaGroupsAttr.getContext(), shloMeshAttr,
                              replicaGroupsAttr.getAxes()));
            }
          }
        }
      }
    });

    // Remove all MeshOps.
    for (auto meshOp : llvm::make_early_inc_range(moduleOp.getOps<MeshOp>())) {
      symbolTable.erase(meshOp);
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
