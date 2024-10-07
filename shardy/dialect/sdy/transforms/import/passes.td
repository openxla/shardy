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

include "mlir/Pass/PassBase.td"

def LiftInlinedMeshesPass : Pass<"sdy-lift-inlined-meshes", "ModuleOp"> {
  let summary = "Lifts inlined `MeshAttr`s in shardings as symbol `MeshOp`s.";
  let description = [{
    Replaces any inlined `MeshAttr` in a `TensorShardingAttr` with a mesh symbol
    name, referencing either an existing or new `MeshOp` in the module, such
    that no two `MeshOp`s with an identical `MeshAttr` (existing `MeshOp`s are
    deduped as well).

    Each new mesh will have the first available name in
    [`mesh`, `mesh_0`, `mesh_1`, ...].
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def AddDataFlowEdgesPass : Pass<"sdy-add-data-flow-edges", "func::FuncOp"> {
  let summary = "Inserts `DataFlowEdgeOp` for every data-flow edge.";
  let description = [{
    Inserts `DataFlowEdgeOp` for every value that is the owner of a data-flow
    edge, i.e., all values returned by `getDataFlowEdgeOwners` on every op in
    the module.

    The inserted `DataFlowEdgeOp` will take the existing sharding of the owner
    target if it exists.

    TODO(b/330339693): update this doc when `getDataFlowEdgeOwners` is removed.
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def ApplyShardingConstraintsPass : Pass<"sdy-apply-sharding-constraints", "func::FuncOp"> {
  let summary = "Applies constraints that dictate the sharding of their input.";
  let description = [{
    Copies the sharding of a `ShardingConstraintOp` to its input if it satisfies
    all of the following:

    * The input doesn't have an existing sharding.
    * The input isn't produced by a `DataFlowEdgeOp`, which holds the sharding
      of all targets of the edge.
    * The sharding of the `ShardingConstraintOp` is fully closed.
    * The input doesn't have any other users of type `ShardingConstraintOp` or
      `ManualComputationOp`.

    Which indicates that the `ShardingConstraintOp` dictates the sharding of
    its input.

    Note that the sharding of a `ShardingConstraintOp` will propagate to its
    input or users during propagation regardless of this pass, but since the
    closed property of a dimension doesn't propagate, it's important to copy the
    sharding to fully respect the constraint in the above cases.

    In addition, if a tensor is used by a chain of `ShardingConstraintOp`s that
    satisfy all of the following:

    * The tensor isn't produced by a `ShardingConstraintOp` and doesn't have any
      other users of type `ShardingConstraintOp` or `ManualComputationOp`.
    * None of the `ShardingConstraintOp`s in the chain have more than one use
      except the last one.
    * The last `ShardingConstraintOp` in the chain doesn't have any users of
      type `ShardingConstraintOp` or `ManualComputationOp` (otherwise it's not
      the last in the chain).

    then this pass replaces all other uses of the input of the chain with the
    result of the last `ShardingConstraintOp` in the chain, as it should
    dictate the sharding of those uses.

    NOTE: The `in_shardings` of a `ManualComputationOp` are in essense sharding
    constraints on the corresponding operands, so this pass will also apply
    their sharding if the above conditions are satisfied (expect for the
    dangling case).
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def ConstantSplitterPass : Pass<"sdy-constant-splitter", "func::FuncOp"> {
  let summary = "Splits constant sub-computations so each has a single use.";
  let description = [{
    Splits constant sub-computations such that they have a single user.

    This ensures that a sharding isn't propagated between different uses of a
    constant sub-computation, as this is considered a false dependency (the uses
    of a constant shouldn't be sharded in the same way just because they use the
    same constant). In effect, each use can have a different sharding that can
    propagate in isolation to its own copy of the constant sub-computation.

    A constant sub-computation is either:
    * a constant or iota op (no operands)
    * a broadcast, slice, or pure element-wise op, whose operands are all
      defined by constant sub-computations (recursively), along with the entire
      sub-computations that define its operands.

    Note that within a constant sub-computation, a value can have multiple uses
    within that sub-computation.

    NOTE: This pass is the MLIR equivalent of xla::HloConstantSplitter,
    needed for the purpose of Shardy Propagation.
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def ShardingGroupImportPass : Pass<"sdy-sharding-group-import", "ModuleOp"> {
  let summary = "Canonicalization and validation pass for sharding groups.";
  let description = [{
    Applies canonicalization and validation to sharding groups upon import.
    Namely these are:

    1) Sharding Group Unification -
       Combines sharding groups using the transitive property of group
       membership. Any time that a tensor T is in a sharding group G1 *and*
       sharding group G2, then we can infer that all members in G1 and G2 should
       be sharded in the same way. Thus we can combine G1 and G2 into a single
       group. The set of canonical group ids after merging will be 0,1,...N-1
       for the minimum set of groups.

    2) Sharding Group Validation
       Validates that sharding groups are well formed and conform to assumptions
       within the implementation. This currently asserts that if a sharding
       group contains a `Value` defined inside the block of a
       `ManualComputationOp`, then all other values in that group must reside in
       the same block.
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def ManualAxesCleanupPass : Pass<"sdy-manual-axes-cleanup", "ModuleOp"> {
  let summary = "Cleans up the use of manual axes in `ManualComputationOp`s";
  let description = [{
    1) For any in/out sharding that hasn't specified a manual axis, add that
       manual axis to its replicated_axes. This is to ensure manual axes are
       always fully specified.
    2) Sorts the manual axes in mesh axis declaration order.
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}
