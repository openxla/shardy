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
#include <string>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_IMPORTMAXIMALSHARDINGPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

struct ImportMaximalShardingPass
    : public impl::ImportMaximalShardingPassBase<ImportMaximalShardingPass> {
  using ImportMaximalShardingPassBase::ImportMaximalShardingPassBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(moduleOp);
    MLIRContext* context = moduleOp.getContext();
    OpBuilder builder(moduleOp);
    builder.setInsertionPointToStart(moduleOp.getBody());

    moduleOp.walk([&](Operation* op) {
      IntegerAttr attr = op->getAttrOfType<IntegerAttr>(kShardingAttr);
      if (!attr) {
        return;
      }
      int64_t deviceId = attr.getInt();
      std::string meshName = llvm::formatv("maximal_mesh_{0}", deviceId);
      if (auto meshOp = symbolTable.lookup<MeshOp>(meshName)) {
        assert(meshOp.getMesh().isMaximal(deviceId));
      } else {
        symbolTable.insert(builder.create<mlir::sdy::MeshOp>(
            moduleOp.getLoc(), meshName, MeshAttr::get(context, deviceId)));
      }
      for (Value result : op->getResults()) {
        int64_t rank = 0;
        if (auto shaped_type = dynCastStaticShapedType(result.getType())) {
          rank = shaped_type.getRank();
        }
        setSharding(result, sdy::TensorShardingAttr::getFullyClosed(
                                context, rank, meshName));
      }
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
