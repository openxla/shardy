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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_OPTIMIZE_COLLECTIVES_PATTERNS_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_OPTIMIZE_COLLECTIVES_PATTERNS_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

// Pattern that optimizes collective permute preceding an all-to-all chain.
class AllToAllOptimizationPattern : public OpRewritePattern<AllToAllOp> {
 public:
  using OpRewritePattern<AllToAllOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllToAllOp terminalOp,
                                PatternRewriter& rewriter) const override;
};

using OptimizeCollectivesPattern = AllToAllOptimizationPattern;

// Populates patterns to optimize collective operations.
void populateOptimizeCollectivesPatterns(RewritePatternSet& patterns);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_OPTIMIZE_COLLECTIVES_PATTERNS_H_
