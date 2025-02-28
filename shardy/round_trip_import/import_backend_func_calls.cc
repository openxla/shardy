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

#include "shardy/round_trip_import/import_backend_func_calls.h"

#include <cassert>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
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

using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;

class BackendFuncCallPattern : public OpConversionPattern<CallOp> {
 public:
  explicit BackendFuncCallPattern(MLIRContext* context,
                                  const SymbolTable& symbolTable)
      : OpConversionPattern<CallOp>(context), symbolTable(symbolTable) {}

  LogicalResult matchAndRewrite(
      CallOp callOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!hasFrontendAttr(callOp, kXlaBackendConfigAttr)) {
      return failure();
    }

    FuncOp func = symbolTable.lookup<FuncOp>(adaptor.getCallee());
    assert(func &&
           ("Failed to lookup function: " + std::string(adaptor.getCallee()))
               .c_str());
    SmallVector<NamedAttribute> namedCompAttrs;
    llvm::copy_if(callOp->getDiscardableAttrs(),
                  std::back_inserter(namedCompAttrs),
                  [](const NamedAttribute& attr) {
                    return attr.getName() != kShardingAttr;
                  });

    auto namedCompOp = rewriter.replaceOpWithNewOp<NamedComputationOp>(
        callOp, callOp->getResultTypes(), adaptor.getCallee(),
        adaptor.getOperands(), /*inShardings=*/nullptr,
        /*outShardings=*/getShardingPerValue(callOp));
    namedCompOp->setAttrs(namedCompAttrs);
    if (func.getBody().empty()) {
      return rewriter.notifyMatchFailure(callOp, [](Diagnostic& diag) {
        diag << "Tried to use an already inlined FuncOp. Expected each CallOp "
                "with backend_config to have a unique FuncOp.";
      });
    }

    inlineRegionAndConvertTerminatorOp<ReturnOp>(
        func.getBody(), namedCompOp.getRegion(), rewriter);
    rewriter.eraseOp(func);

    return success();
  }

 private:
  const SymbolTable& symbolTable;
};

// Converts a `CallOp` with `backend_config` into a `NamedComputationOp`.
class ImportBackendFuncCallsPass
    : public PassWrapper<ImportBackendFuncCallsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportBackendFuncCallsPass)

  void runOnOperation() final {
    // NOTE: Assume that there is a unique callee for each caller. So no need to
    // do a walk and copy the callees if there are multiple callers for the
    // callee.
    MLIRContext& context = getContext();
    ConversionTarget target(context);
    target.addLegalOp<NamedComputationOp, ReturnOp>();
    SymbolTable symbolTable(getOperation());
    target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
      // In case the assumption that each host-callback caller has a unique
      // callee is not true, and an optimized build is being run without
      // verification, make sure that the callee is a function that exists.
      return !hasFrontendAttr(op, kXlaBackendConfigAttr);
    });
    RewritePatternSet patterns(&context);
    patterns.add<BackendFuncCallPattern>(&context, symbolTable);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "sdy-import-backend-func-calls";
  }

  StringRef getDescription() const override {
    return "Creates a pass that converts a `CallOp` with a `backend_config` "
           "attr to a `NamedComputationOp` with the function body inlined and "
           "name of the callee.";
  }
};

}  // namespace

std::unique_ptr<Pass> createImportBackendFuncCallsPass() {
  return std::make_unique<ImportBackendFuncCallsPass>();
}

void registerImportBackendFuncCallsPass() {
  registerPass(createImportBackendFuncCallsPass);
}

}  // namespace sdy
}  // namespace mlir
