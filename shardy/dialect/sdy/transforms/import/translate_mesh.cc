/* Copyright 2025 The OpenXLA Authors.

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

#include <iterator>
#include <memory>  // IWYU pragma: keep
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_TRANSLATEMESHPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

LogicalResult translateMesh(ModuleOp moduleOp,
                            StringRef oldMeshName,
                            ArrayRef<std::string> newMeshAxisNames) {
  MLIRContext* context = moduleOp.getContext();
  auto oldMeshOp = SymbolTable::lookupNearestSymbolFrom<sdy::MeshOp>(
      moduleOp, SymbolRefAttr::get(context, oldMeshName));
  if (!oldMeshOp) {
    return moduleOp.emitError()
           << "Mesh " << oldMeshName << " not found in module.";
  }
  sdy::MeshAttr oldMesh = oldMeshOp.getMesh();
  if (oldMesh.getAxes().size() != newMeshAxisNames.size()) {
    return moduleOp.emitError()
           << "Both meshes must have the same number of axes.";
  }
  llvm::StringMap<StringRef> oldToNewAxis;
  bool sameMesh = true;
  for (auto [oldAxis, newAxisName] :
       llvm::zip_equal(oldMesh.getAxes(), newMeshAxisNames)) {
    oldToNewAxis[oldAxis.getName()] = newAxisName;
    if (oldAxis.getName() != newAxisName) {
      sameMesh = false;
    }
  }
  // Exit early if the meshes are the exact same.
  if (sameMesh) {
    return success();
  }
  StringAttr meshName = StringAttr::get(context, oldMeshName);
  sdy::transformShardings(
      moduleOp,
      [&](sdy::TensorShardingAttr oldSharding) -> sdy::TensorShardingAttr {
        SmallVector<sdy::DimensionShardingAttr> newDimShardings;
        for (auto oldDimSharding : oldSharding.getDimShardings()) {
          SmallVector<sdy::AxisRefAttr> newAxisRefs;
          llvm::transform(oldDimSharding.getAxes(),
                          std::back_inserter(newAxisRefs),
                          [&](sdy::AxisRefAttr oldAxisRef) {
                            return sdy::AxisRefAttr::get(
                                context, oldToNewAxis[oldAxisRef.getName()],
                                oldAxisRef.getSubAxisInfo());
                          });
          newDimShardings.push_back(sdy::DimensionShardingAttr::get(
              context, newAxisRefs, oldDimSharding.getIsClosed(),
              oldDimSharding.getPriority()));
        }
        SmallVector<sdy::AxisRefAttr> newReplicatedAxes;
        llvm::transform(oldSharding.getReplicatedAxes(),
                        std::back_inserter(newReplicatedAxes),
                        [&](sdy::AxisRefAttr oldAxisRef) {
                          return sdy::AxisRefAttr::get(
                              context, oldToNewAxis[oldAxisRef.getName()],
                              oldAxisRef.getSubAxisInfo());
                        });
        return sdy::TensorShardingAttr::get(context, meshName, newDimShardings,
                                            newReplicatedAxes);
      });
  SmallVector<MeshAxisAttr> newAxes;
  newAxes.reserve(newMeshAxisNames.size());
  for (const auto& [axisName, oldAxis] :
       llvm::zip_equal(newMeshAxisNames, oldMesh.getAxes())) {
    newAxes.push_back(MeshAxisAttr::get(context, axisName, oldAxis.getSize()));
  }
  IRRewriter rewriter(moduleOp);
  rewriter.setInsertionPoint(oldMeshOp);
  SymbolTable symbolTable(moduleOp);
  auto newMeshOp = rewriter.create<MeshOp>(
      moduleOp.getLoc(), oldMeshName,
      MeshAttr::get(context, newAxes, oldMesh.getDeviceIds()));
  symbolTable.erase(oldMeshOp);
  symbolTable.insert(newMeshOp);
  return success();
}

struct TranslateMeshPass
    : public impl::TranslateMeshPassBase<TranslateMeshPass> {
  using TranslateMeshPassBase::TranslateMeshPassBase;

  void runOnOperation() final {
    if (translateMesh(getOperation(), oldMeshName, llvm::to_vector(axisNames))
            .failed()) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
