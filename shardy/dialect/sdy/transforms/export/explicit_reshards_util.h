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

// Returns reduction axes that are the union of all axes on reduction factors.
// The result axes are not necessarilly canonicalized.
SmallVector<AxisRefAttr> getReductionAxes(const AxesPerFactor& axesPerFactor,
                                          OpShardingRuleAttr shardingRule);

// Returns true iff any tensor factor sharding has non-empty overflow axes.
bool hasOverflowAxes(const ShardingProjection& shardingProjection);

// Checks if factor sharding is compatible, that is, it satisfies:
// 1. Factors are sharded the same way across operands and results.
// 2. Factors that need replication are unsharded.
//
// Returns the common axes per factor if the factor sharding is compatible.
// Otherwise, returns empty AxesPerFactor.
//
// Assumes factor shardings do not have overflow axes.
AxesPerFactor getCompatibleFactorShardings(
    const ShardingProjection& shardingProjection,
    OpShardingRuleAttr shardingRule);

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

// Inserts an `sdy.all-reduce` for each result of `op` if `reductionAxes`
// is non-empty. Assume the followings:
// - Op has some results.
// - All op results have the same unreduced axes.
// - All op results have the same mesh as `mesh` ignoring device id orders.
void insertAllReducesForReductionFactors(Operation* op,
                                         ArrayRef<AxisRefAttr> reductionAxes,
                                         const Mesh& mesh,
                                         IRRewriter& rewriter);

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
AxesPerFactor findCommonAxes(ArrayRef<TensorShardingAttr> inShardings,
                             ArrayRef<TensorShardingAttr> outShardings,
                             const ShardingProjection& shardingProjection,
                             OpShardingRuleAttr shardingRule,
                             ArrayRef<int64_t> tensorSizes,
                             const SymbolTable& symbolTable, const Mesh& mesh);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_EXPLICIT_RESHARD_UTIL_H_
