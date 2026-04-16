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

#include "shardy/dialect/sdy/transforms/propagation/utils.h"

#include <algorithm>
#include <iterator>

#include "llvm/ADT/BitVector.h"  // IWYU pragma: keep
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"  // IWYU pragma: keep
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

SmallVector<int> toSetBitsVector(const BitVector& bitVector) {
  SmallVector<int> result;
  llvm::copy(bitVector.set_bits(), std::back_inserter(result));
  return result;
}

namespace {

// Determines whether the two directions are forward and backward.
// FORWARD, BACKWARD -> true
// BACKWARD, FORWARD -> true
// All other instances returns false.
bool bothForwardAndBackward(PropagationDirection lhs,
                            PropagationDirection rhs) {
  return (lhs == PropagationDirection::FORWARD &&
          rhs == PropagationDirection::BACKWARD) ||
         (lhs == PropagationDirection::BACKWARD &&
          rhs == PropagationDirection::FORWARD);
}

}  // namespace

PropagationDirection unionOfPropagationDirections(PropagationDirection d1,
                                                  PropagationDirection d2) {
  return bothForwardAndBackward(d1, d2) ? PropagationDirection::BOTH
                                        : std::max(d2, d1);
}

PropagationDirection intersectionOfPropagationDirections(
    PropagationDirection d1, PropagationDirection d2) {
  return bothForwardAndBackward(d1, d2) ? PropagationDirection::NONE
                                        : std::min(d2, d1);
}

bool isFullyReplicated(TensorShardingAttr sharding) {
  return !sharding || sharding.isFullyReplicated();
}

bool isEquivalent(TensorShardingAttr sharding,
                  TensorShardingAttr anotherSharding) {
  if (!sharding) {
    return isFullyReplicated(anotherSharding);
  }
  return sharding.isEquivalent(anotherSharding);
}

SmallVector<TensorShardingAttr> getShardingsFromDataFlowEdges(
    ValueRange edgeOwners) {
  SmallVector<TensorShardingAttr> shardings;
  shardings.reserve(edgeOwners.size());

  StringRef meshName;
  for (Value edgeOwner : edgeOwners) {
    TensorShardingAttr sharding;
    if (auto dataFlowEdgeOp = DataFlowEdgeOp::lookup(edgeOwner)) {
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

void addDataFlowEdges(ValueRange edgeOwners, IRRewriter& rewriter) {
  // We are iterating the owners in a reversed order because we set the
  // insertion point after each value and we would like to keep the data flow
  // edges for the arguments/results in the same order as they appear.
  for (Value edgeOwner : llvm::reverse(edgeOwners)) {
    rewriter.setInsertionPointAfterValue(edgeOwner);
    if (!isStaticShapedType(edgeOwner.getType())) {
      // Skip non-static-shaped tensors, e.g., tokens.
      continue;
    }
    auto dataFlowEdge = DataFlowEdgeOp::create(
        rewriter, edgeOwner.getLoc(), edgeOwner, getSharding(edgeOwner));
    rewriter.replaceAllUsesExcept(edgeOwner, dataFlowEdge, dataFlowEdge);
  }
}

}  // namespace sdy
}  // namespace mlir
