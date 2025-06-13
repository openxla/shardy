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

#include <memory>  // IWYU pragma: keep
#include <tuple>
#include <utility>

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define DEBUG_TYPE "sdy-export"

#define GEN_PASS_DEF_CONSTANTORSCALARMERGERPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

using ConstantKey = std::pair<DictionaryAttr, Region*>;
using BroadcastKey = std::tuple<DictionaryAttr, Region*, Type, Value>;

Operation* maybeGetCachedOp(
    Operation* op, llvm::DenseMap<ConstantKey, Operation*>& constantsCache,
    llvm::DenseMap<BroadcastKey, Operation*>& broadcastsCache) {
  if (llvm::isa<sdy::ConstantOp>(op) || op->hasTrait<OpTrait::ConstantLike>()) {
    auto key = std::make_pair(op->getAttrDictionary(), op->getParentRegion());
    auto [cachedIt, inserted] = constantsCache.try_emplace(key, op);
    // If insert was successful, then this is a new constant.
    return inserted ? nullptr : cachedIt->second;
  }
  if (auto broadcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(op);
      broadcastOp && isScalar(broadcastOp.getOperand())) {
    auto key = std::make_tuple(broadcastOp->getAttrDictionary(),
                               broadcastOp->getParentRegion(),
                               broadcastOp.getType(), broadcastOp.getOperand());
    auto [cachedIt, inserted] = broadcastsCache.try_emplace(key, op);
    // If insert was successful, then this is a new broadcast.
    return inserted ? nullptr : cachedIt->second;
  }
  return nullptr;
}

struct ConstantOrScalarMergerPass
    : public impl::ConstantOrScalarMergerPassBase<ConstantOrScalarMergerPass> {
  using ConstantOrScalarMergerPassBase::ConstantOrScalarMergerPassBase;

  void runOnOperation() final {
    // Store a map of <AttrDictionary,ParentRegion>
    // This will account for all sharding annotations, as well as ensure that
    // dedup does not cause operations or computations to be moved between
    // regions and potentially invalidate sharding annotations.
    llvm::DenseMap<ConstantKey, Operation*> constantsCache;
    llvm::DenseMap<BroadcastKey, Operation*> broadcastsCache;
    getOperation().walk([&](Operation* op) {
      if (Operation* cachedOp =
              maybeGetCachedOp(op, constantsCache, broadcastsCache);
          cachedOp) {
        LLVM_DEBUG(llvm::dbgs() << "Deduplicating op: " << *op << "\n"
                                << "With: " << *cachedOp << "\n");
        op->replaceAllUsesWith(cachedOp);
        op->erase();
      }
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
