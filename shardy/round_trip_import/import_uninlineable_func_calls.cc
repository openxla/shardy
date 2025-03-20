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

#include "shardy/round_trip_import/import_uninlineable_func_calls.h"

#include <cassert>
#include <iterator>
#include <memory>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/round_trip_import/constants.h"
#include "shardy/round_trip_import/utils.h"

namespace mlir {
namespace sdy {

namespace {

using func::CallOp;
using func::FuncOp;

bool isInlineableCallOp(CallOp callOp) {
  if (hasFrontendAttr(callOp, kXlaBackendConfigAttr)) {
    return false;
  }
  auto inlineableAttr =
      tryGetFrontendAttr<BoolAttr>(callOp, kXlaInlineableAttr);
  return !inlineableAttr || inlineableAttr->getValue();
}

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
  rewriter.setInsertionPoint(callOp);
  auto namedCompOp = rewriter.create<NamedComputationOp>(
      callOp->getLoc(), callOp->getResultTypes(), calleeName,
      callOp.getOperands(),
      /*inShardings=*/nullptr,
      /*outShardings=*/getShardingPerValue(callOp));
  namedCompOp->setAttrs(namedCompAttrs);

  Region& namedCompRegion = namedCompOp.getRegion();
  if (auto movedRegionIt = calleeNameToMovedRegion.find(calleeName);
      movedRegionIt != calleeNameToMovedRegion.end()) {
    static llvm::once_flag onceFlag;
    emitOpWarningOnce(
        onceFlag, callOp,
        llvm::formatv("uninlineable function @{0} has multiple call ops, we "
                      "need to clone the function body for each call",
                      calleeName)
            .str());
    rewriter.cloneRegionBefore(*movedRegionIt->second, namedCompRegion,
                               namedCompRegion.begin());
  } else {
    FuncOp funcOp = symbolTable.lookup<FuncOp>(calleeName);
    assert(funcOp &&
           ("Failed to lookup function: " + std::string(calleeName)).c_str());
    inlineRegionAndConvertTerminatorOp<ReturnOp>(funcOp.getBody(),
                                                 namedCompRegion);
    calleeNameToMovedRegion[calleeName] = &namedCompRegion;
  }

  rewriter.replaceOp(callOp, namedCompOp);
}

class ImportUninlineableFuncCallsPass
    : public PassWrapper<ImportUninlineableFuncCallsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportUninlineableFuncCallsPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(moduleOp.getContext());
    SymbolTable symbolTable(moduleOp);
    // For every callee name, the first CallOp encountered with that symbol will
    // move the body of the callee into the created NamedComputationOp, and map
    // the symbol name to the moved region. Subsequent CallOps with that symbol
    // will clone the mapped region.
    llvm::SmallDenseMap<StringRef, Region*> calleeNameToMovedRegion;

    moduleOp->walk([&](CallOp op) {
      if (isInlineableCallOp(op)) {
        return;
      }
      importCallOp(op, calleeNameToMovedRegion, rewriter, symbolTable);
    });

    // Erase all func ops that now have no call ops.
    for (auto [calleeName, _] : calleeNameToMovedRegion) {
      symbolTable.erase(symbolTable.lookup(calleeName));
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-import-uninlineable-func-calls";
  }

  StringRef getDescription() const override {
    return "Creates a pass that converts a `CallOp` with a `backend_config` "
           "or `inlineable=false` frontend attr to a `NamedComputationOp` with "
           "the function body inlined and the name of the callee.";
  }
};

}  // namespace

std::unique_ptr<Pass> createImportUninlineableFuncCallsPass() {
  return std::make_unique<ImportUninlineableFuncCallsPass>();
}

void registerImportUninlineableFuncCallsPass() {
  registerPass(createImportUninlineableFuncCallsPass);
}

}  // namespace sdy
}  // namespace mlir
