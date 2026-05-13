/* Copyright 2026 The Shardy Authors.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_PROPAGATETOFUNCRESULTSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

void setFuncResultShardingOrClear(func::FuncOp funcOp,
                                  const OpOperand& opOperand) {
  int64_t resNum = opOperand.getOperandNumber();
  if (auto sharding = getSharding(opOperand.get())) {
    setFuncResultSharding(funcOp, resNum, sharding);
    return;
  }
  funcOp.removeResultAttr(resNum, kShardingAttr);
}

struct PropagateToFuncResultsPass
    : public impl::PropagateToFuncResultsPassBase<PropagateToFuncResultsPass> {
  using PropagateToFuncResultsPassBase::PropagateToFuncResultsPassBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    func::FuncOp mainFuncOp =
        getMainFuncOrDie(moduleOp, symbolTable, /*useSingleFunc=*/true);
    for (func::FuncOp funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (funcOp == mainFuncOp) {
        continue;
      }
      auto returnOp =
          dyn_cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());
      if (!returnOp) continue;
      for (const OpOperand& opOperand : returnOp->getOpOperands()) {
        setFuncResultShardingOrClear(funcOp, opOperand);
      }
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
