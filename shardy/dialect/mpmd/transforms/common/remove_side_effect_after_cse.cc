/* Copyright 2025 The MPMD Authors.

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

#include <memory>
#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::mpmd {

namespace {

#define GEN_PASS_DEF_REMOVESIDEEFFECTAFTERCSEPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

struct RemoveSideEffectAfterCSEPass
    : public impl::RemoveSideEffectAfterCSEPassBase<
          RemoveSideEffectAfterCSEPass> {
  using impl::RemoveSideEffectAfterCSEPassBase<
      RemoveSideEffectAfterCSEPass>::RemoveSideEffectAfterCSEPassBase;

  void runOnOperation() override {
    getOperation().walk([&](stablehlo::CustomCallOp customCallOp) {
      if (customCallOp->hasAttr(kMhloNoCseAttr)) {
        customCallOp.setHasSideEffect(std::nullopt);
      }
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createRemoveSideEffectAfterCSEPass() {
  return std::make_unique<RemoveSideEffectAfterCSEPass>();
}

}  // namespace mlir::mpmd
