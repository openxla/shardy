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
#include <cstdint>
#include <tuple>

#include "llvm/ADT/DenseMap.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_UNFLATTENCALLGRAPHPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

using func::CallOp;
using func::FuncOp;

using ComputationKey =
    std::tuple<StringAttr /*name*/, ManualAxesAttr /*manualAxes*/,
               TensorShardingPerValueAttr /*inShardings*/,
               TensorShardingPerValueAttr /*outShardings*/
               >;

TensorShardingPerValueAttr maybeGetFuncArgShardings(
    FuncOp funcOp, const SymbolTable& symbolTable, bool ignoreShardings) {
  if (ignoreShardings) {
    return TensorShardingPerValueAttr();
  }
  return getFuncArgShardings(funcOp, symbolTable);
}
TensorShardingPerValueAttr maybeGetFuncResultShardings(
    FuncOp funcOp, const SymbolTable& symbolTable, bool ignoreShardings) {
  if (ignoreShardings) {
    return TensorShardingPerValueAttr();
  }
  return getFuncResultShardings(funcOp, symbolTable);
}

ManualAxesAttr getManualAxesAttr(FuncOp funcOp) {
  return funcOp->getAttrOfType<ManualAxesAttr>(kFuncManualAxes);
}

ComputationKey getComputationKey(FuncOp funcOp, const SymbolTable& symbolTable,
                                 bool ignoreShardings = false) {
  return {getOriginalFuncName(funcOp), getManualAxesAttr(funcOp),
          maybeGetFuncArgShardings(funcOp, symbolTable, ignoreShardings),
          maybeGetFuncResultShardings(funcOp, symbolTable, ignoreShardings)};
}

ComputationKey getComputationKey(CallOp callOp, const SymbolTable& symbolTable,
                                 bool ignoreShardings) {
  return getComputationKey(getFuncOpOrDie(callOp.getCallee(), symbolTable),
                           symbolTable, ignoreShardings);
}

llvm::SmallDenseMap<ComputationKey, FuncOp> populateFuncCache(
    ModuleOp moduleOp, const SymbolTable& symbolTable,
    bool dedupFunctionsFully) {
  llvm::SmallDenseMap<ComputationKey, FuncOp> funcCache;
  moduleOp.walk([&](CallOp callOp) {
    FuncOp funcOp = getFuncOpOrDie(callOp.getCallee(), symbolTable);
    ComputationKey funcCacheKey = getComputationKey(
        funcOp, symbolTable, /*ignoreShardings=*/dedupFunctionsFully);
    // Keep the attribute for the original func name as other calls to the
    // same function would still need it to deduplicate.
    funcCache.try_emplace(funcCacheKey, funcOp);
  });

  // Count the calls sites and pick the funcOp with the largest calls.
  if (dedupFunctionsFully) {
    llvm::SmallDenseMap<ComputationKey, int64_t> callCounts;
    moduleOp.walk([&](CallOp callOp) {
      FuncOp funcOp = getFuncOpOrDie(callOp.getCallee(), symbolTable);
      ComputationKey funcCacheKey =
          getComputationKey(funcOp, symbolTable, /*ignoreShardings=*/true);

      // Increment the call count of `funcOp`.
      ComputationKey funcOpKey = getComputationKey(funcOp, symbolTable);
      callCounts[funcOpKey]++;

      // Update `funcCache` with `funcOp` if it has larger call count.
      auto cachedFuncOpIt = funcCache.find(funcCacheKey);
      ComputationKey cachedFuncOpKey =
          getComputationKey(cachedFuncOpIt->second, symbolTable);
      if (callCounts[funcOpKey] > callCounts[cachedFuncOpKey]) {
        cachedFuncOpIt->second = funcOp;
      }
    });
  }
  return funcCache;
}

struct UnflattenCallGraphPass
    : public impl::UnflattenCallGraphPassBase<UnflattenCallGraphPass> {
  using UnflattenCallGraphPassBase::UnflattenCallGraphPassBase;

  // Unflattens the graph. It deduplicates functions with the same
  // input/output shardings *and* the same origin as desribed by the
  // 'original_func_name' attribute attached to the functions.
  //
  // However, when `dedupFunctionsFully` is enabled it disregards input/output
  // shardings and deduplicates all functions on the same origin. This means
  // it needs to pick one of the input/output shardings, and copy operations
  // before and after some calls in order to match the input/output shardings
  // the selected function expects.
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    IRRewriter rewriter(moduleOp.getContext());

    llvm::SmallDenseMap<ComputationKey, FuncOp> funcCache =
        populateFuncCache(moduleOp, symbolTable, dedupFunctionsFully);
    moduleOp.walk([&](CallOp callOp) {
      ComputationKey funcCacheKey = getComputationKey(
          callOp, symbolTable, /*ignoreShardings=*/dedupFunctionsFully);
      FuncOp funcOp = funcCache[funcCacheKey];
      callOp.setCallee(funcOp.getName());
      insertReshardsOnFuncArguments(funcOp, callOp, symbolTable, rewriter);
    });

    moduleOp.walk([&](FuncOp funcOp) {
      funcOp->removeAttr(kOriginalFuncName);
      funcOp->removeAttr(kFuncManualAxes);
    });
  }
};
}  // namespace

}  // namespace sdy
}  // namespace mlir
