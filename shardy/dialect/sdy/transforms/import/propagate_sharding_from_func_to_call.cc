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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_PROPAGATESHARDINGFROMFUNCTOCALLPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

using func::CallOp;
using func::FuncOp;

struct PropagateShardingFromFuncToCallPass
    : public impl::PropagateShardingFromFuncToCallPassBase<
          PropagateShardingFromFuncToCallPass> {
  using PropagateShardingFromFuncToCallPassBase::
      PropagateShardingFromFuncToCallPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // Propagate shardings from func results to call results if call does not
    // have them and func does.
    moduleOp.walk([&](CallOp callOp) {
      FuncOp funcOp = getFuncOpOrDie(callOp.getCallee(), symbolTable);
      if (!getShardingPerValue(callOp)) {
        if (TensorShardingPerValueAttr funcResultShardings =
                getFuncResultShardings(funcOp, symbolTable);
            funcResultShardings) {
          setShardings(callOp, funcResultShardings);
        }
      }
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
