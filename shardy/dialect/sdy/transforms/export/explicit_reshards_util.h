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

// Returns a concatenated array of operand and result tensor sizes.
SmallVector<int64_t> getTensorSizes(Operation* op);

// Returns true iff any tensor factor sharding has non-empty overflow axes.
bool hasOverflowAxes(const ShardingProjection& shardingProjection);

// Insert explicit reshards for operands and results that change by
// the given `shardingProjection` for a given `op`. The reshards are inserted
// only to make the given operation compatible.
//
// For example,
//
// ```mlir
//   %arg0: tensor<8x32xf32> { sdy.sharding = @mesh, [{}, {"y"}]>}
//   %arg1: tensor<32x16xf32> { sdy.sharding = <@mesh, [{"y"}, {"x"}]>}
//   %0 = stablehlo.dot %arg0, %arg1 { sdy.sharding = <@mesh, [{"x"}, {}]>,
//     sdy.sharding_rule = <([i, k], [k, j])->([i, j])> }
//   %1 = stablehlo.negate %0 {sdy.sharding = <@mesh, [{"x"}, {}]>
//   return %1
// ```
//
// after a call on the stablehlo.dot operation, by the sharding projection,
// i: {}, j: {}, k: {"y"}, the module becomes:
//
// ```mlir
//   %arg0: tensor<8x32xf32> { sdy.sharding = @mesh, [{}, {"y"}]>}
//   %arg1: tensor<32x16xf32> { sdy.sharding = <@mesh, [{"y"}, {"x"}]>}
//   %0 = stablehlo.reshard %arg1 {sdy.sharding = <@mesh, [{"y"}, {}]>}
//   %1 = stablehlo.dot %arg0, %0 { sdy.sharding = <@mesh, [{}, {}]>,
//     sdy.sharding_rule = <([i, k], [k, j])->([i, j])> }
//   %2 = stablehlo.reshard %1 {sdy.sharding = <@mesh, [{"x"}, {}]>}
//   %3 = stablehlo.negate %2 {sdy.sharding = <@mesh, [{"x"}, {}]>
//   return %3
// ```
//
// In the above example, note that the operand and result shardings for
// stablehlo.negate op remained unchanged.
//
// Assumes factor shardings do not have overflow axes.
// TODO(enver): Handle the case when some factor shardings have overflow axes.
//
// Assumes all tensor shardings have the same mesh as `mesh` on axes but may be
// different on device order.
void insertExplicitReshards(Operation* op,
                            ArrayRef<TensorShardingAttr> inShardings,
                            ArrayRef<TensorShardingAttr> outShardings,
                            const ShardingProjection& shardingProjection,
                            UpdateTensorShardings updateTensorShardings,
                            IRRewriter& rewriter,
                            OpShardingRuleAttr shardingRule,
                            const SymbolTable& symbolTable, const Mesh& mesh);

// Inserts an `sdy.all-reduce` for each result of `op`.
//
// Assumes the followings:
// - All op results have the same unreduced axes.
// - All op results have the same mesh as `mesh` ignoring device id orders.
// - `commonAxesPerFactor` is non-empty if `onFullVersion` is true.
//
// In case `onFullVersion` is false, it inserts all reduces only if op results
// have some unreduced axes.
//
// Hard fails if the reduction factors do not have compatible shardings, and op
// results have unreduced axes.
void insertAllReducesForReductionFactors(
    Operation* op, const ShardingProjection& shardingProjection,
    const AxesPerFactor& commonAxesPerFactor, OpShardingRuleAttr shardingRule,
    const Mesh& mesh, IRRewriter& rewriter, bool onFullVersion);

// Finds common factor axes on the operands and results of `op` so that the
// sharding of `op` is compatible with its sharding rule.
//
// Refer to the documentation of `InsertExplicitReshardsPass` for more details.
//
// Assume the followings:
// - All op results have the same unreduced axes.
// - If the op has no results, none of the operands has unreduced axes.
// - Operand and result meshes are the same ignoring device id order.
// - There are no overflow axes.
// - Some shardings are not compatible.
//
// Guarantees to return a non-empty AxesPerFactor.
AxesPerFactor findCommonAxes(const ShardingProjection& shardingProjection,
                             OpShardingRuleAttr shardingRule,
                             ArrayRef<int64_t> tensorSizes, const Mesh& mesh);

// Converts a `sdy.reshard` op to an `sdy.replicated-to-unreduced` op and/or an
// `sdy.sharded-to-unreduced` op. Returns true if the conversion is successful.
//
// `r2u` keeps the sharded size, while `s2u` increases the sharded size. Hence,
// we do `r2u` first and then `s2u`.
//
// The requirements are:
// 1. `op` is a `sdy.reshard` op.
// 2. The input and output shardings have the same mesh.
// 3. The input of `op` is another `sdy.reshard` op or a block argument.
// 4. The input unreduced axes is a strict subset of the output unreduced axes.
bool convertReshardToUnreducedCollectives(Operation* op, IRRewriter& rewriter,
                                          const SymbolTable& symbolTable);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_EXPLICIT_RESHARD_UTIL_H_
