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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_VALIDATENORESHARDSPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

class ValidateNoReshardsPass
    : public impl::ValidateNoReshardsPassBase<ValidateNoReshardsPass> {
  using ValidateNoReshardsPassBase::ValidateNoReshardsPassBase;

 protected:
  void runOnFunc(func::FuncOp func) override {
    func.walk([&](FragmentCallOp callOp) {
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          callOp, callOp.getCalleeAttr());
      if (!callee || !isReshardOnly(callee)) {
        return;
      }

      InFlightDiagnostic diag = failOnReshardOnlyFragments
                                    ? callOp.emitError()
                                    : callOp.emitWarning();
      diag << "Detected reshard-only fragment '" << callOp.getCallee()
           << "'. This usually indicates an "
              "unexpected reshard. Operands: ";

      auto printSharding = [](Type type, InFlightDiagnostic& diag) {
        auto meshType = mlir::dyn_cast<MeshTensorType>(type);
        if (auto sharding = meshType ? meshType.getSharding() : nullptr) {
          diag << sharding;
        } else {
          diag << "Replicated";
        }
      };

      llvm::interleaveComma(
          llvm::zip(callOp.getOperands(), callOp.getOperandTypes()), diag,
          [&](auto pair) {
            auto [operand, type] = pair;
            printSharding(type, diag);
            diag << " " << operand.getLoc();
          });
      diag << ". Results: ";
      bool first = true;
      for (auto [result, type] :
           llvm::zip(callOp.getResults(), callOp.getResultTypes())) {
        if (!first) {
          diag << ", ";
        }
        first = false;
        printSharding(type, diag);
        for (OpOperand& use : result.getUses()) {
          auto returnOp = mlir::dyn_cast<func::ReturnOp>(use.getOwner());
          if (!returnOp) {
            continue;
          }
          if (auto loc = GetResultInfoLoc(func, use.getOperandNumber())) {
            diag << " " << *loc;
          }
        }
      }

      if (failOnReshardOnlyFragments) {
        signalPassFailure();
      }
    });
  }

  static bool isReshardOnly(func::FuncOp callee) {
    Block& body = callee.getBody().front();
    if (!llvm::hasSingleElement(body)) {
      return false;
    }
    auto returnOp = mlir::dyn_cast<func::ReturnOp>(body.front());
    if (!returnOp) {
      return false;
    }
    return llvm::all_of(returnOp.getOperands(),
                        [](Value v) { return mlir::isa<BlockArgument>(v); });
  }
};

}  // namespace

}  // namespace mlir::mpmd
