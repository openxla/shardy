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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_OPTIMIZE_COLLECTIVES_UTIL_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_OPTIMIZE_COLLECTIVES_UTIL_H_

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

// Represents a matched collective permute and chain of all-to-all operations.
struct A2AChain {
  CollectivePermuteOp cpOp;
  SmallVector<AllToAllOp> a2aOps;
};

// Represents a single all-to-all stage with its parameters, output sharding,
// and output type.
struct AllToAllStage {
  AllToAllParamListAttr params;
  TensorShardingAttr outSharding;
  RankedTensorType outType;
};

// Configuration computed for decomposing the split dimension and rewriting the
// all-to-all chain.
struct SplitConfig {
  RankedTensorType splitInputType;
  TensorShardingAttr splitInputSharding;
  SmallVector<AllToAllStage> a2aStages;
  RankedTensorType outputType;
  TensorShardingAttr outputSharding;
};

// Extracts a single-use chain of AllToAllOp operations preceded by a
// CollectivePermuteOp, ending at terminalOp.
std::optional<A2AChain> extractA2AChain(AllToAllOp terminalOp);

// Verifies all formal safety invariants and identifies the single optimizable
// split dimension.
std::optional<int64_t> getOptimizableSplitDim(const A2AChain& chain);

// Computes the split shape, sharding attribute, and combined all-to-all
// parameters.
std::optional<SplitConfig> computeSplitConfig(const A2AChain& chain,
                                              int64_t splitDim);

// Emits input reshape, combined all-to-all, output reshape, replaces the
// terminal op, and erases intermediate ops.
LogicalResult rewriteA2AChain(const A2AChain& chain, const SplitConfig& config,
                              PatternRewriter& rewriter);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_OPTIMIZE_COLLECTIVES_UTIL_H_
