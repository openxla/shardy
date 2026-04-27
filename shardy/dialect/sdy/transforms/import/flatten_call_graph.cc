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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
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

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    llvm::SmallDenseSet<StringRef> funcNames;

    walkCalls(moduleOp, [&](CallOp callOp) {
      // TODO(enver): Should we special handle loops and conditionals?
      FuncOp funcOp = getFuncOpOrDie(callOp.getCallee(), symbolTable);
      if (auto [_, inserted] = funcNames.insert(funcOp.getName()); inserted) {
        return WalkResult::advance();
      }
      callOp.setCallee(
          symbolTable.insert(cloneFuncRecursively(funcOp, symbolTable)));
      return WalkResult::advance();
    });

    walkCalls(moduleOp, [&](CallOp callOp) {
      FuncOp funcOp = getFuncOpOrDie(callOp.getCallee(), symbolTable);
      if (auto callOpResultShardings = getShardingPerValue(callOp)) {
        setFuncResultShardings(funcOp, callOpResultShardings);
      }
      return WalkResult::advance();
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
