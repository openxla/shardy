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
#include <utility>

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define DEBUG_TYPE "sdy-export"

#define GEN_PASS_DEF_CONSTANTMERGERPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

struct ConstantMergerPass
    : public impl::ConstantMergerPassBase<ConstantMergerPass> {
  using ConstantMergerPassBase::ConstantMergerPassBase;

  void runOnOperation() final {
    // Store a map of <AttrDictionary,ParentRegion>
    // This will account for all sharding annotations, as well as ensure that
    // dedup does not cause operations or computations to be moved between
    // regions and potentially invalidate sharding annotations.
    llvm::DenseMap<std::pair<DictionaryAttr, Region*>, Operation*>
        constantsCache;

    getOperation().walk([&](Operation* op) {
      if (!llvm::isa<sdy::ConstantOp>(op) &&
          !op->hasTrait<OpTrait::ConstantLike>()) {
        return;
      }
      auto key = std::make_pair(op->getAttrDictionary(), op->getParentRegion());
      auto [cachedIt, inserted] = constantsCache.try_emplace(key, op);
      if (inserted) {
        // If insert was successful, then this is a new constant.
        return;
      }

      Operation* cachedConstant = cachedIt->second;
      LLVM_DEBUG(llvm::dbgs() << "Deduplicating constant: " << *op << "\n"
                              << "With: " << *cachedConstant << "\n");
      op->replaceAllUsesWith(cachedConstant);
      op->erase();
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
