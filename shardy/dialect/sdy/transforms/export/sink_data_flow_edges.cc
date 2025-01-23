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

#include <cassert>
#include <memory>  // IWYU pragma: keep
#include <tuple>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_SINKDATAFLOWEDGESPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// Gets a vector of `TensorShardingAttr` for the given edge owner.
//
// Each value in `edgeOwners` is the owner of a data flow edge. If the data flow
// edge already has a sharding, we will copy the sharding. Otherwise, if one
// of the owners in `edgeOwners` has a sharding, we create a fully open sharding
// with the mesh name of the first such sharding for all the other values that
// don't have a sharding.
SmallVector<TensorShardingAttr> getShardingsFromDataFlowEdges(
    ValueRange edgeOwners) {
  SmallVector<TensorShardingAttr> shardings;
  shardings.reserve(edgeOwners.size());

  StringRef meshName;
  for (Value edgeOwner : edgeOwners) {
    TensorShardingAttr sharding;
    if (DataFlowEdgeOp dataFlowEdgeOp =
            DataFlowEdgeOp::getDataFlowEdgeUser(edgeOwner)) {
      sharding = dataFlowEdgeOp.getShardingAttr();
      if (sharding && meshName.empty()) {
        meshName = sharding.getMeshName();
      }
    }
    shardings.push_back(sharding);
  }
  if (meshName.empty()) {
    return {};
  }
  // There is at least one `DataFlowEdgeOp` with a sharding.
  // Replace all empty shardings with fully open shardings.
  // NOTE: this will replace the existing edgeOwner's sharding, if any, though
  // this shouldn't happen as as `sdy-add-data-flow-edges` would have copied it.
  for (auto [sharding, edgeOwner] : llvm::zip_equal(shardings, edgeOwners)) {
    if (!sharding) {
      sharding = TensorShardingAttr::getFullyOpen(
          edgeOwner.getContext(), getTensorRank(edgeOwner), meshName);
    }
  }
  return shardings;
}

// Saves an array of all the origin sharding dictionaries for the given
// `edgeOwners` on `op`. If non exist, nothing is saved.
//
// For debugging the origin shardings, we want to preserve the origin sharding
// dictionaries from the `DataFlowEdgeOp`s on the owning op so that they are
// preserved after the propagation pipeline.
//
// See the `debug-sharding-origins` config on propagation for more details.
//
// TODO(b/388458831): add `saveDebugPropagationInfo` to the pass and pass it in
// here. Can then reserve the right size for `originShardingDicts` and not need
// the `exists` boolean.
void buildOriginShardingDictsFromDataFlowEdges(ValueRange edgeOwners,
                                               Operation* op,
                                               StringRef attrName,
                                               IRRewriter& rewriter) {
  SmallVector<Attribute> originShardingDicts;
  // TODO(b/388458831): pass through a boolean indicating whether the origin
  // sharding debug information is enabled.
  bool exists = false;
  for (Value edgeOwner : edgeOwners) {
    DictionaryAttr dict;
    if (DataFlowEdgeOp dataFlowEdgeOp =
            DataFlowEdgeOp::getDataFlowEdgeUser(edgeOwner)) {
      dict =
          dataFlowEdgeOp->getAttrOfType<DictionaryAttr>(kShardingOriginsAttr);
    }
    if (!dict) {
      dict = rewriter.getDictionaryAttr({});
    } else {
      exists = true;
    }
    originShardingDicts.push_back(dict);
  }
  if (exists) {
    op->setAttr(attrName, rewriter.getArrayAttr(originShardingDicts));
  }
}

struct SinkDataFlowEdgesPass
    : public impl::SinkDataFlowEdgesPassBase<SinkDataFlowEdgesPass> {
  using SinkDataFlowEdgesPassBase::SinkDataFlowEdgesPassBase;

  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp);
    // Copy the sharding from data flow edges to the data flow ops.
    funcOp.walk<WalkOrder::PreOrder>([&](Operation* op) {
      // Since we are doing the walk in preorder with a forward iterator, ops
      // are walked before their users and regions. Since `DataFlowEdgeOp` can
      // only appear inside the data flow op's region or as its user, we always
      // encounter the data flow op before their data flow edges. This means it
      // is safe to erase the `DataFlowEdgeOp` at this point. We need the skip
      // at the end because it's a condition to erase the op. See the
      // documentation for `Operation::walk` for more details.
      if (isa<DataFlowEdgeOp>(op)) {
        DataFlowEdgeOp dataFlowEdgeOp = cast<DataFlowEdgeOp>(op);
        rewriter.replaceOp(dataFlowEdgeOp, dataFlowEdgeOp.getInput());
        return WalkResult::skip();
      }
      auto shardableDataFlowOp = dyn_cast<ShardableDataFlowOpInterface>(op);
      if (!shardableDataFlowOp) {
        return WalkResult::advance();
      }
      ArrayRef<BlockArgument> blockArgOwners =
          shardableDataFlowOp.getBlockArgumentEdgeOwners();
      if (SmallVector<TensorShardingAttr> blockArgShardings =
              getShardingsFromDataFlowEdges(blockArgOwners);
          !blockArgShardings.empty()) {
        shardableDataFlowOp.setBlockArgumentEdgeOwnerShardings(
            blockArgShardings);
      }
      buildOriginShardingDictsFromDataFlowEdges(
          blockArgOwners, op, kBlockArgShardingOriginsAttr, rewriter);

      ResultRange resultOwners = shardableDataFlowOp.getOpResultEdgeOwners();
      if (SmallVector<TensorShardingAttr> resultShardings =
              getShardingsFromDataFlowEdges(resultOwners);
          !resultShardings.empty()) {
        shardableDataFlowOp.setOpResultEdgeOwnerShardings(resultShardings);
      }
      buildOriginShardingDictsFromDataFlowEdges(
          resultOwners, op, kResultShardingOriginsAttr, rewriter);
      return WalkResult::advance();
    });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
