/* Copyright 2025 The Shardy Authors.

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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_EXPLICIT_RESHARD_UTIL_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_EXPLICIT_RESHARD_UTIL_H_

#include <cassert>
#include <cstdint>
#include <optional>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

// TODO(enver): Use MeshOp instead.
// The struct contains a mesh attribute and its name.
struct Mesh {
  MeshAttr mesh;
  StringRef meshName;
  Mesh() = default;
  Mesh(MeshAttr mesh, StringRef meshName) : mesh(mesh), meshName(meshName) {};
  MLIRContext* getContext() const { return mesh.getContext(); }
  MeshAttr attr() const { return mesh; }
  StringRef name() const { return meshName; }
};

// Returns true if a reshard is needed to go from source sharding to target
// sharding.
bool shouldReshard(TensorShardingAttr sourceSharding,
                   TensorShardingAttr targetSharding);

// Inserts an `sdy.all-reduce` on `use` if `sourceSharding` has unreduced axes
// or sub-axes that aren't in the user's unreduced axes
// (`userSharding.getUnreducedAxes()` if present , or an empty array otherwise.)
//
// The inserted all-reduce will have `sourceSharding.getUnreducedAxes() -
// targetUnreducedAxes` as the reduction axes.
//
// NOTE: we don't verify correctness of an operation that takes an unreduced
// input and produces an unreduced output, we simply respect the existing
// shardings and match cases where an unreduced axis becomes sharded/replicated.
//
// Hard fails if sourceSharding and userSharding have different meshes.
// TODO(b/442783457): Support when the meshes differ only on device order.
//
// Returns the sharding of the inserted all-reduce, or `sourceSharding` if none
// was inserted.
TensorShardingAttr insertAllReduceIfUnreducedToReplicated(
    OpOperand& use, TensorShardingAttr sourceSharding,
    TensorShardingAttr userSharding, const SymbolTable& symbolTable,
    IRRewriter& rewriter);

// Returns the factor sharding of `factorIndex` if present, or std::nullopt
// otherwise.
std::optional<ArrayRef<AxisRefAttr>> getFactorSharding(
    const TensorFactorShardings& factorShardings, int64_t factorIndex);

// Returns unreduced axes of given `sharding`. If `sharding` is null, returns
// empty axes.
ArrayRef<AxisRefAttr> getUnreducedAxes(TensorShardingAttr sharding);

// Returns unreduced axes of given `value`. If its sharding is null, returns
// empty axes.
ArrayRef<AxisRefAttr> getUnreducedAxes(Value value);

// Inserts an `sdy.all-reduce` for each result of `op` if any of its reduction
// factors is sharded in `commonAxesPerFactor`.
// Assume the followings:
// - All op results have the same unreduced axes.
// - All op results have the same mesh as `mesh` ignoring device id orders.
// - Only the reduction factors are properly set in `commonAxesPerFactor`.
// TODO(enver): Change to take axes only for reduction factors.
void insertAllReducesForReductionFactors(
    Operation* op, const AxesPerFactor& commonAxesPerFactor, const Mesh& mesh,
    OpShardingRuleAttr shardingRule, IRRewriter& rewriter);

// Inserts explicit reshards on the operands and results of `op` such that the
// sharding of `op` is compatible with its sharding rule.
//
// Refer to the documentation of `InsertExplicitReshardsPass` for more details.
//
// Assume the followings:
// - All op results have the same unreduced axes.
// - If the op has no results, none of the operands has unreduced axes.
// - Operand and result meshes are the same ignoring device id order.
//
// Returns the common axes per factor.
std::optional<AxesPerFactor> insertExplicitReshardsOnOp(
    Operation* op, ArrayRef<TensorShardingAttr> inShardings,
    ArrayRef<TensorShardingAttr> outShardings, IRRewriter& rewriter,
    const SymbolTable& symbolTable, OpShardingRuleAttr shardingRule,
    const Mesh& mesh);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_EXPLICIT_RESHARD_UTIL_H_
