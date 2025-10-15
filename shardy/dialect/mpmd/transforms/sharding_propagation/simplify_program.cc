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

#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/common/simplify_region_op_base.h"
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_SIMPLIFYPROGRAMPASS
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h.inc"

namespace {

using ::llvm::BitVector;
using ::mlir::PatternRewriter;
using ::mlir::StringRef;
using ::mlir::TypeRange;
using ::mlir::ValueRange;
using ::mlir::mpmd::FragmentOp;
using ::mlir::mpmd::SimplifyRegionOpPatternBase;

class SimplifyFragmentPattern : public SimplifyRegionOpPatternBase<FragmentOp> {
  using SimplifyRegionOpPatternBase::SimplifyRegionOpPatternBase;

 protected:
  FragmentOp createNewOp(FragmentOp op, PatternRewriter& rewriter,
                         TypeRange result_types, ValueRange operands,
                         BitVector erased_results) const override {
    auto newOp = rewriter.create<FragmentOp>(
        op.getLoc(), result_types, operands, op.getOriginAttr(),
        op.getMeshNameAttr(), op.getStageIdAttr());
    // Make sure that any attribute that is not origin or mesh_name (which are
    // copied above) is preserved (e.g., remat or call_counter info).
    for (NamedAttribute attr : op->getAttrs()) {
      auto attrName = attr.getName().strref();
      if (attrName == "origin" || attrName == "mesh_name") {
        continue;
      }
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    return newOp;
  }
};

class SimplifyProgramPass
    : public impl::SimplifyProgramPassBase<SimplifyProgramPass> {
  using SimplifyProgramPassBase::SimplifyProgramPassBase;

 protected:
  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyFragmentPattern>(&getContext());
    if (mlir::failed(
            applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
