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
#include <utility>

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

// The struct contains an array of axes list, and a mesh.
struct AxesPerFactorWithMesh {
  AxesPerFactor axes;
  Mesh mesh;
  AxesPerFactorWithMesh() = default;
  AxesPerFactorWithMesh(AxesPerFactor axes, Mesh mesh)
      : axes(std::move(axes)), mesh(mesh) {};
  bool empty() const { return axes.empty(); }
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
// Returns the sharding of the inserted all-reduce, or `sourceSharding` if none
// was inserted.
TensorShardingAttr insertAllReduceIfUnreducedToReplicated(
    OpOperand& use, TensorShardingAttr sourceSharding,
    TensorShardingAttr userSharding, MeshAttr mesh, IRRewriter& rewriter);

// Returns true if any of `axes` overlaps with `axis`.
bool hasOverlappingAxis(ArrayRef<AxisRefAttr> axes, AxisRefAttr axis);

// Returns the factor sharding of `factorIndex` if present, or std::nullopt
// otherwise.
std::optional<ArrayRef<AxisRefAttr>> getFactorSharding(
    const TensorFactorShardings& factorShardings, int64_t factorIndex);

// Returns whether the (first) result sharding is different than any of the
// operand shardings. If `op` does not have any results, returns false;
bool differentOperandShardingFromFirstResult(Operation* op);

// Inserts explicit reshards on the operands and results of `op` such that the
// sharding of `op` is compatible with its sharding rule.
//
// Refer to the documentation of `InsertExplicitReshardsPass` for more details.
void insertExplicitReshardsOnOp(Operation* op, IRRewriter& rewriter,
                                const SymbolTable& symbolTable,
                                bool onFullVersion);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_EXPLICIT_RESHARD_UTIL_H_
