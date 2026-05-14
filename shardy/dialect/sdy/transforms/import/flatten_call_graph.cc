/* Copyright 2026 The OpenXLA Authors.
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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_FLATTENCALLGRAPHPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

using func::CallOp;
using func::FuncOp;

struct FlattenCallGraphPass
    : public impl::FlattenCallGraphPassBase<FlattenCallGraphPass> {
  using FlattenCallGraphPassBase::FlattenCallGraphPassBase;

  // A nullptr for `flattenCallGraphUnder` implies a full flattening.
  FlattenCallGraphPass(llvm::function_ref<bool(func::CallOp)> predicate)
      : flattenCallGraphUnder(predicate) {}
  llvm::function_ref<bool(func::CallOp)> flattenCallGraphUnder = nullptr;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    llvm::SmallDenseSet<StringRef> funcNames;

    walkCalls(moduleOp, [&](CallOp callOp) {
      if (flattenCallGraphUnder && !flattenCallGraphUnder(callOp)) {
        return WalkResult::advance();
      }
      // TODO(enver): Should we special handle loops and conditionals?
      FuncOp funcOp = getFuncOpOrDie(callOp.getCallee(), symbolTable);
      if (auto [_, inserted] = funcNames.insert(funcOp.getName()); inserted) {
        if (flattenCallGraphUnder) {
          // This is the first call to the function. Keep the function itself
          // uncloned (it will be cloned on a second call though) but clone the
          // call graphs under it and during which flatten thos sub call graphs.
          // Because, unlike the case of full flattening it is *not* guaranteed
          // that any other calls to the called functions to be cloned.
          funcOp->walk([&](CallOp callOp) {
            FuncOp calledFuncOp =
                getFuncOpOrDie(callOp.getCallee(), symbolTable);
            callOp.setCallee(symbolTable.insert(
                cloneFuncRecursively(calledFuncOp, symbolTable)));
          });
        }
        // In the case of full flatenning, it does not need to clone the called
        // functions as it is guaranteed that any second call later to the
        // called functions is guaranteed to be cloned.
        return WalkResult::advance();
      }
      // A second call the the function. Clone fully and flatten.
      callOp.setCallee(
          symbolTable.insert(cloneFuncRecursively(funcOp, symbolTable)));
      return WalkResult::advance();
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createFlattenCallGraphPass(
    llvm::function_ref<bool(func::CallOp)> predicate) {
  return std::make_unique<FlattenCallGraphPass>(predicate);
}

}  // namespace sdy
}  // namespace mlir
