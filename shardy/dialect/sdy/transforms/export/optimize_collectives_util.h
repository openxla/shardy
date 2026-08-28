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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

// Represents a matched collective permute and chain of all-to-all operations.
struct AllToAllChain {
  CollectivePermuteOp cpOp;
  SmallVector<AllToAllOp> a2aOps;
};

// Represents a single all-to-all stage with its parameters, output sharding,
// and output type in the optimized all-to-all sequence.
//
// `params` (`AllToAllParamListAttr`) is an MLIR array attribute containing a
// list of `AllToAllParamAttr` entries for `sdy.all_to_all`. Each parameter
// entry specifies:
//   - `axis`: The mesh axis (or sub-axis) across which data is exchanged.
//   - `src_dim`: The source tensor dimension (in the decomposed/reshaped
//      tensor) that is currently sharded along this axis and will be
//      gathered.
//   - `tgt_dim`: The target tensor dimension to which the axis is
//      scattered/sharded.
struct AllToAllStage {
  AllToAllParamListAttr params;
  TensorShardingAttr outSharding;
  RankedTensorType outType;
};

// Plan computed for decomposing the split dimension and rewriting the
// all-to-all chain.
struct AllToAllRewritePlan {
  RankedTensorType splitInputType;
  TensorShardingAttr splitInputSharding;
  SmallVector<AllToAllStage> a2aStages;
  RankedTensorType outputType;
  TensorShardingAttr outputSharding;
};

// Validates tensor and mesh preconditions using `symbolTable`, identifies the
// single modified dimension whose sharding was changed by `cpOp`, and verifies
// that `cpOp` purely permutes at least two axes on that dimension. Returns the
// split dimension index, or std::nullopt if validation fails.
//
// Example:
// - cpOp changes sharding <@mesh, [{"x", "y"}, {}]> to
//   <@mesh, [{"y", "x"}, {}]> -> returns 0
std::optional<int64_t> getSplitDimension(CollectivePermuteOp cpOp,
                                         const SymbolTable& symbolTable);

// Traces consecutive single-use AllToAllOp consumers of `cpOp` that source
// exclusively from `splitDim`, stopping when an AllToAllOp has multiple uses
// or sources from a different dimension.
std::optional<AllToAllChain> extractAllToAllChain(CollectivePermuteOp cpOp,
                                                  int64_t splitDim);

// Returns true if the AllToAllOp chain communicates all permuted axes from
// `splitDim`. Uses `symbolTable` to resolve mesh information.
//
// Example:
// - Permuted axes on splitDim are {"x", "y"}, chain communicates
//   [{"x"}, {"y"}] -> returns true
bool isChainOptimizable(const AllToAllChain& chain, int64_t splitDim,
                        const SymbolTable& symbolTable);

// Computes the split shape, sharding attribute, and combined all-to-all
// parameters using `symbolTable` to resolve mesh information.
std::optional<AllToAllRewritePlan> computeRewritePlan(
    const AllToAllChain& chain, int64_t splitDim,
    const SymbolTable& symbolTable);

// Emits input reshape, combined all-to-all, output reshape, replaces the
// terminal op, and erases intermediate ops.
LogicalResult rewriteAllToAllChain(const AllToAllChain& chain,
                                   const AllToAllRewritePlan& plan,
                                   PatternRewriter& rewriter);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_OPTIMIZE_COLLECTIVES_UTIL_H_
