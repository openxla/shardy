/* Copyright 2026 The MPMD Authors.

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

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_arg_res_attrs.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_SINKCREATETOKENINTOFRAGMENTSPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

template <typename T>
SmallVector<T> filterByBitVector(ArrayRef<T> elements,
                                 const BitVector& toErase) {
  SmallVector<T> filtered;
  for (auto [idx, elem] : llvm::enumerate(elements)) {
    if (!toErase.test(idx)) {
      filtered.push_back(elem);
    }
  }
  return filtered;
}

class SinkCreateTokenIntoFragmentsPass
    : public impl::SinkCreateTokenIntoFragmentsPassBase<
          SinkCreateTokenIntoFragmentsPass> {
  using SinkCreateTokenIntoFragmentsPassBase::
      SinkCreateTokenIntoFragmentsPassBase;

 protected:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!IsMpmdFunction(funcOp)) {
      return;
    }

    IRRewriter rewriter(funcOp.getContext());

    // Pass 1: sink token inputs into all fragments. This replaces all uses of
    // token block arguments with a local stablehlo.create_token and removes
    // the token operands/args from the FragmentOp. After this pass, all uses
    // of token-typed FragmentOp results are gone (tokens only flow between
    // fragments as operands).
    for (FragmentOp fragment : funcOp.getOps<FragmentOp>()) {
      sinkTokenInputsIntoFragment(rewriter, fragment);
    }

    // Pass 2: drop token-typed results from all fragments. By the time we
    // reach here, all uses of token results have been removed by pass 1, so
    // it is safe to rebuild each FragmentOp without token results.
    for (FragmentOp fragment :
         llvm::make_early_inc_range(funcOp.getOps<FragmentOp>())) {
      dropTokenResultsFromFragment(rewriter, fragment);
    }
  }

  // Pass 1: sinks token-typed inputs into the fragment body by inserting a
  // local stablehlo.create_token op, replacing all uses of the corresponding
  // block argument, and removing the token operand from the FragmentOp.
  void sinkTokenInputsIntoFragment(IRRewriter& rewriter, FragmentOp fragment) {
    Block& body = fragment.getRegion().front();
    ValueRange operands = fragment.getInputs();
    Block::BlockArgListType args = body.getArguments();

    if (operands.size() != args.size()) {
      SDY_LOG(ERROR) << "MISMATCH: inputs.size()=" << operands.size()
                     << " != body.args.size()=" << args.size()
                     << " -- skipping this fragment!";
      return;
    }

    BitVector toErase(static_cast<unsigned>(operands.size()), /*t=*/false);

    rewriter.setInsertionPointToStart(&body);

    for (auto [idx, z] : llvm::enumerate(llvm::zip(operands, args))) {
      auto [operand, arg] = z;
      if (!isa<stablehlo::TokenType>(operand.getType())) {
        continue;
      }
      auto newToken =
          stablehlo::CreateTokenOp::create(rewriter, fragment.getLoc());
      arg.replaceAllUsesWith(newToken.getResult());
      toErase.set(static_cast<unsigned>(idx));
    }

    if (toErase.none()) {
      return;
    }

    // Update in_shardings before erasure.
    if (auto inShardings = fragment.getInShardings()) {
      fragment.setInShardingsAttr(sdy::TensorShardingPerValueAttr::get(
          fragment.getContext(),
          filterByBitVector(inShardings->getShardings(), toErase)));
    }

    // Filter arg_attrs if present.
    if (fragment->hasAttr(kArgAttrName)) {
      SetArgAttrs(fragment, filterByBitVector<Attribute>(
                                GetArgAttrsOrCreateDefault(fragment), toErase));
    }

    fragment.getOperation()->eraseOperands(toErase);
    body.eraseArguments(toErase);
  }

  void dropTokenResultsFromFragment(IRRewriter& rewriter, FragmentOp fragment) {
    Block& body = fragment.getRegion().front();
    Operation* terminator = body.getTerminator();

    BitVector tokenResults(static_cast<unsigned>(fragment.getNumResults()),
                           /*t=*/false);
    for (OpResult result : fragment.getResults()) {
      if (isa<stablehlo::TokenType>(result.getType())) {
        // Pass 1 removed all cross-fragment token uses, so by now all token
        // results must be dead.
        SDY_CHECK(result.use_empty())
            << "Token result #" << result.getResultNumber()
            << " still has uses after token-input sinking! Fragment: "
            << fragment;
        tokenResults.set(result.getResultNumber());
      }
    }

    if (tokenResults.none()) {
      return;
    }

    // Erase the token operands from the mpmd.return terminator.
    terminator->eraseOperands(tokenResults);

    // Rebuild the FragmentOp with filtered result types, following the pattern
    // of EliminateUnusedResults in fragment_dce.cc.
    rewriter.setInsertionPoint(fragment);
    auto newFragment = FragmentOp::create(
        rewriter, fragment.getLoc(),
        FilterRange<Type>(fragment.getResultTypes(), tokenResults),
        fragment.getOperands(), fragment.getOriginAttr(),
        fragment.getMeshNameAttr(), fragment.getStageIdAttr());

    // Filter res_attrs if present.
    if (fragment->hasAttr(kResAttrName)) {
      SetResAttrs(newFragment,
                  filterByBitVector<Attribute>(
                      GetResAttrsOrCreateDefault(fragment), tokenResults));
    }

    // Filter out_shardings if present.
    if (auto outShardings = fragment.getOutShardings()) {
      newFragment.setOutShardingsAttr(sdy::TensorShardingPerValueAttr::get(
          fragment.getContext(),
          filterByBitVector(outShardings->getShardings(), tokenResults)));
    }

    // Copy all remaining attributes (origin and mesh_name were already set
    // during creation; res_attrs and out_shardings were already filtered).
    CopyAttributes(fragment, newFragment,
                   /*elided_attrs_set=*/
                   {"origin", "mesh_name", "res_attrs", "out_shardings"});
    newFragment.getRegion().takeBody(fragment.getRegion());

    // Replace uses of the kept (non-token) results with newFragment results.
    BitVector keptResults = tokenResults;
    keptResults.flip();
    for (auto [oldResultIndex, newResult] :
         llvm::zip(keptResults.set_bits(), newFragment.getResults())) {
      rewriter.replaceAllUsesWith(fragment.getResult(oldResultIndex),
                                  newResult);
    }
    rewriter.eraseOp(fragment);
  }
};

}  // namespace
}  // namespace mlir::mpmd
