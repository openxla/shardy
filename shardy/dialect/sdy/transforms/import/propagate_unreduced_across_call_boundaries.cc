/* Copyright 2024 The Shardy Authors.

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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_PROPAGATEUNREDUCEDACROSSCALLBOUNDARIESPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

using func::CallOp;
using func::FuncOp;

// Removes any axis in `axesToRemove` from `dimSharding`.
DimensionShardingAttr removeAxes(DimensionShardingAttr dimSharding,
                                 ArrayRef<AxisRefAttr> axesToRemove) {
  SmallVector<AxisRefAttr> newAxes;
  for (AxisRefAttr axis : dimSharding.getAxes()) {
    bool remove = false;
    for (AxisRefAttr removeAxis : axesToRemove) {
      if (axis.getName() == removeAxis.getName()) {
        remove = true;
        break;
      }
    }
    if (!remove) {
      newAxes.push_back(axis);
    }
  }
  if (newAxes.size() == dimSharding.getAxes().size()) {
    return dimSharding;
  }
  return DimensionShardingAttr::get(dimSharding.getContext(), newAxes,
                                    dimSharding.getIsClosed(),
                                    dimSharding.getPriority());
}

// Copies unreduced shardings from caller operands to callee arguments if
// present, as unreduced axes don't propagate.
// If the argument already has a sharding, we remove any conflicting axes (axes
// that are now unreduced) from its dimension shardings.
void copyUnreducedShardingsToCallees(ModuleOp moduleOp,
                                     const SymbolTable& symbolTable) {
  moduleOp.walk([&](CallOp callOp) {
    FuncOp funcOp = symbolTable.lookup<FuncOp>(callOp.getCallee());
    if (!funcOp) {
      return;
    }
    for (auto [i, operand] : llvm::enumerate(callOp.getOperands())) {
      TensorShardingAttr operandSharding = getSharding(operand);
      if (operandSharding && !operandSharding.getUnreducedAxes().empty()) {
        Value argument = funcOp.getArgument(i);
        TensorShardingAttr argumentSharding = getSharding(argument);
        TensorShardingAttr newSharding;
        if (argumentSharding) {
          SmallVector<DimensionShardingAttr> newDimShardings;
          for (DimensionShardingAttr dimSharding :
               argumentSharding.getDimShardings()) {
            newDimShardings.push_back(
                removeAxes(dimSharding, operandSharding.getUnreducedAxes()));
          }
          newSharding = TensorShardingAttr::get(
              argumentSharding.getContext(), argumentSharding.getMeshOrRef(),
              newDimShardings, argumentSharding.getReplicatedAxes(),
              operandSharding.getUnreducedAxes());
        } else {
          newSharding =
              TensorShardingAttr::getFullyOpen(callOp.getContext(),
                                               getTensorRank(operand),
                                               operandSharding.getMeshName())
                  .replaceUnreducedAxes(operandSharding.getUnreducedAxes());
        }
        setSharding(argument, newSharding);
      }
    }
  });
}

void copyUnreducedShardingsToFuncResults(FuncOp funcOp) {
  // Find the return op of the callee.
  func::ReturnOp returnOp;
  funcOp.walk([&](func::ReturnOp op) {
    returnOp = op;
    return WalkResult::interrupt();
  });
  if (!returnOp) {
    return;
  }
  for (auto [i, returnVal] : llvm::enumerate(returnOp.getOperands())) {
    TensorShardingAttr returnValSharding = getSharding(returnVal);
    if (returnValSharding && !returnValSharding.getUnreducedAxes().empty()) {
      TensorShardingAttr funcResultSharding = getFuncResultSharding(funcOp, i);
      TensorShardingAttr newSharding;
      if (funcResultSharding) {
        SmallVector<DimensionShardingAttr> newDimShardings;
        for (DimensionShardingAttr dimSharding :
             funcResultSharding.getDimShardings()) {
          newDimShardings.push_back(
              removeAxes(dimSharding, returnValSharding.getUnreducedAxes()));
        }
        newSharding = TensorShardingAttr::get(
            funcResultSharding.getContext(), funcResultSharding.getMeshOrRef(),
            newDimShardings, funcResultSharding.getReplicatedAxes(),
            returnValSharding.getUnreducedAxes());
      } else {
        newSharding =
            TensorShardingAttr::getFullyOpen(funcOp.getContext(),
                                             getTensorRank(returnVal),
                                             returnValSharding.getMeshName())
                .replaceUnreducedAxes(returnValSharding.getUnreducedAxes());
      }
      setFuncResultSharding(funcOp, i, newSharding);
    }
  }
}

void copyUnreducedShardingsToCallers(ModuleOp moduleOp,
                                     const SymbolTable& symbolTable) {
  moduleOp.walk([&](CallOp callOp) {
    FuncOp funcOp = symbolTable.lookup<FuncOp>(callOp.getCallee());
    if (!funcOp) {
      return;
    }
    for (auto [i, result] : llvm::enumerate(callOp.getResults())) {
      TensorShardingAttr returnSharding = getFuncResultSharding(funcOp, i);
      if (returnSharding && !returnSharding.getUnreducedAxes().empty()) {
        TensorShardingAttr resultSharding = getSharding(result);
        TensorShardingAttr newSharding;
        if (resultSharding) {
          SmallVector<DimensionShardingAttr> newDimShardings;
          for (DimensionShardingAttr dimSharding :
               resultSharding.getDimShardings()) {
            newDimShardings.push_back(
                removeAxes(dimSharding, returnSharding.getUnreducedAxes()));
          }
          newSharding = TensorShardingAttr::get(
              resultSharding.getContext(), resultSharding.getMeshOrRef(),
              newDimShardings, resultSharding.getReplicatedAxes(),
              returnSharding.getUnreducedAxes());
        } else {
          newSharding =
              TensorShardingAttr::getFullyOpen(callOp.getContext(),
                                               getTensorRank(result),
                                               returnSharding.getMeshName())
                  .replaceUnreducedAxes(returnSharding.getUnreducedAxes());
        }
        setSharding(result, newSharding);
      }
    }
  });
}

struct PropagateUnreducedAcrossCallBoundariesPass
    : public impl::PropagateUnreducedAcrossCallBoundariesPassBase<
          PropagateUnreducedAcrossCallBoundariesPass> {
  using PropagateUnreducedAcrossCallBoundariesPassBase::
      PropagateUnreducedAcrossCallBoundariesPassBase;

 protected:
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      copyUnreducedShardingsToFuncResults(funcOp);
    }
    copyUnreducedShardingsToCallees(moduleOp, symbolTable);
    copyUnreducedShardingsToCallers(moduleOp, symbolTable);
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
