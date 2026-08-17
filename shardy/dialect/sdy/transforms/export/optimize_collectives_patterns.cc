/* Copyright 2026 The Shardy Authors.

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

#include "shardy/dialect/sdy/transforms/export/optimize_collectives_patterns.h"

#include <cstdint>
#include <optional>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/export/optimize_collectives_util.h"

namespace mlir {
namespace sdy {

LogicalResult AllToAllOptimizationPattern::matchAndRewrite(
    AllToAllOp terminalOp, PatternRewriter& rewriter) const {
  // Matches a producer chain ending at terminalOp.
  std::optional<A2AChain> chain = extractA2AChain(terminalOp);
  if (!chain) {
    return failure();
  }
  // Validates safety invariants and identifies the split dimension.
  std::optional<int64_t> splitDim = getOptimizableSplitDim(*chain);
  if (!splitDim) {
    return failure();
  }
  // Computes the decomposed shapes, shardings, and parameters.
  std::optional<SplitConfig> config = computeSplitConfig(*chain, *splitDim);
  if (!config) {
    return failure();
  }
  // Emits reshapes and combined all-to-all.
  return rewriteA2AChain(*chain, *config, rewriter);
}

void populateOptimizeCollectivesPatterns(RewritePatternSet& patterns) {
  patterns.add<AllToAllOptimizationPattern>(patterns.getContext());
}

}  // namespace sdy
}  // namespace mlir
