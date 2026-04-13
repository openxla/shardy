/* Copyright 2024 The OpenXLA Authors.

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
#include <iterator>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_IMPORTFUNCCALLSPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

using func::CallOp;
using func::FuncOp;

void importCallOp(
    CallOp callOp,
    llvm::SmallDenseMap<StringRef, Region*>& calleeNameToMovedRegion,
    IRRewriter& rewriter, SymbolTable& symbolTable) {
  SmallVector<NamedAttribute> namedCompAttrs;
  llvm::copy_if(callOp->getDiscardableAttrs(),
                std::back_inserter(namedCompAttrs),
                [](const NamedAttribute& attr) {
                  return attr.getName() != kShardingAttr;
                });

  StringRef calleeName = callOp.getCallee();
  FuncOp funcOp = symbolTable.lookup<FuncOp>(calleeName);
  SDY_CHECK(funcOp) << "Failed to lookup function: " << calleeName.str();

  rewriter.setInsertionPoint(callOp);
  auto namedCompOp = NamedComputationOp::create(
      rewriter, callOp->getLoc(), callOp->getResultTypes(),
      getOriginalFuncName(funcOp), callOp.getOperands(),
      /*inShardings=*/getFuncArgShardings(funcOp, symbolTable),
      // TODO(b/439018088): Take func result shardings if call op result
      // shardings are empty.
      /*outShardings=*/getShardingPerValue(callOp));
  namedCompOp->setAttrs(namedCompAttrs);

  Region& namedCompRegion = namedCompOp.getRegion();
  if (auto movedRegionIt = calleeNameToMovedRegion.find(calleeName);
      movedRegionIt != calleeNameToMovedRegion.end()) {
    static llvm::once_flag onceFlag;
    emitOpWarningOnce(
        onceFlag, callOp,
        llvm::formatv("function @{0} has multiple call ops, we "
                      "need to clone the function body for each call",
                      calleeName)
            .str());
    rewriter.cloneRegionBefore(*movedRegionIt->second, namedCompRegion,
                               namedCompRegion.begin());
  } else {
    inlineRegionAndConvertTerminatorOp<ReturnOp>(funcOp.getBody(),
                                                 namedCompRegion);
    calleeNameToMovedRegion[calleeName] = &namedCompRegion;
  }

  rewriter.replaceOp(callOp, namedCompOp);
}

struct ImportFuncCallsPass
    : public impl::ImportFuncCallsPassBase<ImportFuncCallsPass> {
  using ImportFuncCallsPassBase::ImportFuncCallsPassBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    IRRewriter rewriter(moduleOp.getContext());
    SymbolTable symbolTable(moduleOp);
    // For every callee name, the first CallOp encountered with that symbol will
    // move the body of the callee into the created NamedComputationOp, and map
    // the symbol name to the moved region. Subsequent CallOps with that symbol
    // will clone the mapped region.
    llvm::SmallDenseMap<StringRef, Region*> calleeNameToMovedRegion;

    walkCalls(moduleOp, [&](CallOp callOp) {
      importCallOp(callOp, calleeNameToMovedRegion, rewriter, symbolTable);
      return WalkResult::advance();
    });

    // Erase all func ops that now have no call ops.
    for (auto [calleeName, _] : calleeNameToMovedRegion) {
      symbolTable.erase(symbolTable.lookup(calleeName));
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
