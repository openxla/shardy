/* Copyright 2025 The Shardy Authors.

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
#include <iterator>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_REMOVESIZEONEAXESPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

bool hasSizeOneAxes(MeshOp meshOp) {
  return llvm::any_of(meshOp.getMesh().getAxes(),
                      [](MeshAxisAttr axis) { return axis.getSize() == 1; });
}

TensorShardingAttr removeSizeOneAxes(TensorShardingAttr sharding,
                                     const SymbolTable& symbolTable) {
  MeshAttr mesh = sharding.getMesh(symbolTable);
  assert(mesh && "unknown mesh");

  auto isNotSizeOne = [&](AxisRefAttr axis) { return axis.getSize(mesh) != 1; };

  // Remove from dimension shardings.
  SmallVector<DimensionShardingAttr> dimShardings;
  dimShardings.reserve(sharding.getRank());
  for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
    SmallVector<AxisRefAttr> newAxes;
    newAxes.reserve(dimSharding.getAxes().size());
    llvm::copy_if(dimSharding.getAxes(), std::back_inserter(newAxes),
                  isNotSizeOne);
    // Remove priority if there are no sharding axes and the dimension is
    // closed, since this isn't allowed by verification (would have no effect on
    // propagation).
    std::optional<int64_t> priority =
        newAxes.empty() && dimSharding.getIsClosed()
            ? std::nullopt
            : dimSharding.getPriority();
    dimShardings.push_back(
        DimensionShardingAttr::get(dimSharding.getContext(), newAxes,
                                   dimSharding.getIsClosed(), priority));
  }

  // Remove from replicated axes.
  SmallVector<AxisRefAttr> replicatedAxes;
  llvm::copy_if(sharding.getReplicatedAxes(),
                std::back_inserter(replicatedAxes), isNotSizeOne);

  // Remove from unreduced axes.
  SmallVector<AxisRefAttr> unreducedAxes;
  llvm::copy_if(sharding.getUnreducedAxes(), std::back_inserter(unreducedAxes),
                isNotSizeOne);

  return TensorShardingAttr::get(sharding.getContext(), sharding.getMeshOrRef(),
                                 dimShardings, replicatedAxes, unreducedAxes);
}

struct RemoveSizeOneAxesPass
    : public impl::RemoveSizeOneAxesPassBase<RemoveSizeOneAxesPass> {
  using RemoveSizeOneAxesPassBase::RemoveSizeOneAxesPassBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    if (llvm::none_of(moduleOp.getOps<MeshOp>(), hasSizeOneAxes)) {
      // Nothing to do.
      return;
    }

    transformShardings(moduleOp, [&](TensorShardingAttr sharding) {
      return removeSizeOneAxes(sharding, symbolTable);
    });
    // The meshes still have size one axes, but they are not used in the
    // shardings anymore.
  }
};

}  // namespace
}  // namespace sdy
}  // namespace mlir
