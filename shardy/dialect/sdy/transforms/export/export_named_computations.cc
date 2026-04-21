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
#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/propagation/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_EXPORTNAMEDCOMPUTATIONSPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

using func::CallOp;
using func::FuncOp;

void removeDataFlowEdges(ValueRange values, IRRewriter& rewriter) {
  for (Value value : values) {
    if (value.use_empty()) {
      continue;
    }
    if (auto dataFlowEdgeOp = dyn_cast<DataFlowEdgeOp>(*value.user_begin())) {
      SDY_CHECK(value.hasOneUse());
      rewriter.replaceOp(dataFlowEdgeOp, dataFlowEdgeOp.getInput());
    }
  }
}

struct NamedComputationWithCount {
  NamedComputationOp namedComputationOp;
  int64_t callSiteCount;
};

// TODO(enver): TensorShardingPerValueAttr can be nullptr. Drop optional and use
// nullptr. Be careful on handling an empty TensorShardingPerValueAttr properly.
StringAttr createFuncOp(
    NamedComputationOp namedComputationOp, IRRewriter& rewriter,
    SymbolTable& symbolTable,
    std::optional<TensorShardingPerValueAttr> inShardings,
    std::optional<TensorShardingPerValueAttr> outShardings) {
  FuncOp funcOp = FuncOp::create(
      rewriter, namedComputationOp.getLoc(), namedComputationOp.getName(),
      rewriter.getFunctionType(namedComputationOp.getBody().getArgumentTypes(),
                               namedComputationOp.getResultTypes()),
      rewriter.getStringAttr("private"),
      /*argAttrs=*/ArrayAttr(), /*resultAttrs=*/ArrayAttr());
  funcOp->setAttr(kOriginalFuncName, namedComputationOp.getNameAttr());

  rewriter.setInsertionPointToStart(funcOp->getBlock());
  inlineRegionAndConvertTerminatorOp<func::ReturnOp>(
      namedComputationOp.getBody(), funcOp.getBody());

  removeDataFlowEdges(funcOp.getArguments(), rewriter);

  // Copy the input shardings to the func.
  if (inShardings.has_value()) {
    for (auto [i, sharding] : llvm::enumerate(inShardings->getShardings())) {
      funcOp.setArgAttr(i, kShardingAttr, sharding);
    }
  }

  return symbolTable.insert(funcOp);
}

TensorShardingPerValueAttr getFullyClosedLike(
    TensorShardingPerValueAttr shardings) {
  SmallVector<TensorShardingAttr> resultShardings;
  resultShardings.reserve(shardings.size());
  for (TensorShardingAttr sharding : shardings.getShardings()) {
    resultShardings.push_back(TensorShardingAttr::getFullyClosedLike(sharding));
  }
  return TensorShardingPerValueAttr::get(shardings.getContext(),
                                         resultShardings);
}

void exportNamedComputations(ModuleOp moduleOp, SymbolTable& symbolTable) {
  Block& moduleBlock = moduleOp.getRegion().front();
  MLIRContext* ctx = moduleOp.getContext();
  IRRewriter rewriter(moduleOp);

  // NOTE: The walk needs to be in post order, which is the default order, to
  // account for nested named computations.
  SmallVector<Value> callResults;
  moduleOp.walk([&](NamedComputationOp namedComputationOp) {
    rewriter.setInsertionPointToEnd(&moduleBlock);

    // Propagate the shardings from the data flow edges to argument shardings.
    ArrayRef<BlockArgument> blockArgOwners =
        namedComputationOp.getBody().getArguments();
    if (SmallVector<TensorShardingAttr> blockArgShardings =
            getShardingsFromDataFlowEdges(blockArgOwners);
        !blockArgShardings.empty()) {
      namedComputationOp.setInShardingsAttr(
          TensorShardingPerValueAttr::get(ctx, blockArgShardings));
    }
    std::optional<TensorShardingPerValueAttr> inShardings =
        namedComputationOp.getInShardings();

    // Propagate the shardings from the data flow edges to result shardings.
    ResultRange resultOwners = namedComputationOp.getResults();
    if (SmallVector<TensorShardingAttr> resultShardings =
            getShardingsFromDataFlowEdges(resultOwners);
        !resultShardings.empty()) {
      namedComputationOp.setOutShardingsAttr(
          TensorShardingPerValueAttr::get(ctx, resultShardings));
    }
    std::optional<TensorShardingPerValueAttr> outShardings =
        namedComputationOp.getOutShardings();

    StringAttr funcSymName = createFuncOp(
        namedComputationOp, rewriter, symbolTable, inShardings, outShardings);

    // Replace the `NamedComputationOp` with a `CallOp`.
    rewriter.setInsertionPoint(namedComputationOp);
    SmallVector<NamedAttribute> callOpAttrs(
        namedComputationOp->getDiscardableAttrs());
    auto callOp = rewriter.replaceOpWithNewOp<CallOp>(
        namedComputationOp, namedComputationOp.getResultTypes(), funcSymName,
        namedComputationOp.getOperands());
    llvm::append_range(callResults, callOp.getResults());
    callOp->setAttrs(callOpAttrs);
    if (outShardings) {
      setShardings(callOp, *outShardings);
    }
  });
  removeDataFlowEdges(callResults, rewriter);
}

struct ExportNamedComputationsPass
    : public impl::ExportNamedComputationsPassBase<
          ExportNamedComputationsPass> {
  using ExportNamedComputationsPassBase::ExportNamedComputationsPassBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(moduleOp);

    exportNamedComputations(moduleOp, symbolTable);
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
