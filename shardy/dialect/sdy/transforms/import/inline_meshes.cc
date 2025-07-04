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
      if (auto name = dyn_cast<FlatSymbolRefAttr>(sharding.getMeshOrRef())) {
        MeshAttr mesh = getMeshAttr(symbolTable, name);
        assert(mesh && "unknown mesh");
        return TensorShardingAttr::get(sharding.getContext(), mesh,
                                       sharding.getDimShardings(),
                                       sharding.getReplicatedAxes(),
                                       sharding.getUnreducedAxes());
      }
      return sharding;
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
