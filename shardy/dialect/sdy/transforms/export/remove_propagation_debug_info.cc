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

#include <array>
#include <cassert>
#include <cstdint>
#include <memory>  // IWYU pragma: keep

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_REMOVEPROPAGATIONDEBUGINFOPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

struct RemovePropagationDebugInfoPass
    : public impl::RemovePropagationDebugInfoPassBase<
          RemovePropagationDebugInfoPass> {
  using RemovePropagationDebugInfoPassBase::RemovePropagationDebugInfoPassBase;

  void runOnOperation() final {
    constexpr std::array<StringRef, 6> propagationDebugAttrs = {
        mlir::sdy::kPropagationEdgesAttr,
        mlir::sdy::kBlockArgPropagationEdgesAttr,
        mlir::sdy::kResultPropagationEdgesAttr,
        mlir::sdy::kShardingOriginsAttr,
        mlir::sdy::kBlockArgShardingOriginsAttr,
        mlir::sdy::kResultShardingOriginsAttr};

    mlir::ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      TypeSwitch<Operation *, void>(op)
          .Case<func::FuncOp>([&](func::FuncOp funcOp) {
            for (const mlir::StringRef &debugAttr : propagationDebugAttrs) {
              for (int64_t argNum = 0; argNum < funcOp.getNumArguments();
                   ++argNum) {
                funcOp.removeArgAttr(argNum, debugAttr);
              }
              for (int64_t resNum = 0; resNum < funcOp.getNumResults();
                   ++resNum) {
                funcOp.removeResultAttr(resNum, debugAttr);
              }
            }
          })
          .Default([&](Operation *op) {
            for (const mlir::StringRef &debugAttr : propagationDebugAttrs) {
              op->removeAttr(debugAttr);
            }
          });
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
