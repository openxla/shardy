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

def ShardingConstraintToReshardPass : Pass<"sdy-sharding-constraint-to-reshard", "func::FuncOp"> {
  let summary = "Converts ShardingConstraintOp into ReshardOp.";
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def SinkDataFlowEdgesPass : Pass<"sdy-sink-data-flow-edges", "func::FuncOp"> {
  let summary = "Sinks all `DataFlowEdgeOp` into their input.";
  let description = [{
    Moves the sharding of each `DataFlowEdgeOp` to its input (the root target of
    the edge), and replaces the op with its input.

    TODO(tomnatan): consider moving the sharding to all targets that can have a
    sharding attached.
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def UpdateNonDivisibleInputOutputShardingsPass : Pass<"sdy-update-non-divisible-input-output-shardings", "func::FuncOp"> {
  let summary = "Makes FuncOp inputs/outputs evenly sharded, removing any need for padding due to non-divisible shardings.";
  let description = [{
    Users of Shardy expect the function inputs/outputs to be evenly
    divisible/shardable to avoid requiring padding their tensors. Propagation
    may make inputs/outputs have non-divisible shardings, so this pass updates
    them to the largest dimension sharding prefix of the original sharding that
    is evenly sharded.
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def InsertExplicitReshardsPass : Pass<"sdy-insert-explicit-reshards", "func::FuncOp"> {
  let summary = "Inserts explicit reshards to make all operations have compatible shardings.";
  let description = [{
    A compatible sharding essentially means that the operation can accept the
    sharded operands and produce a sharded result without requiring any reshard
    communications (note that the operation might still require communication
    such as all-reduce or halo-swaps).

    After propagation, some opeartions may still have incompatible shardings.

    Please note, when an axis (or sub-axis) is used to shard non-corresponding
    dimensions (e.g. non-contracting dimensions in matmul) across multiple
    tensors, or when an axis shards a dimension in one tensor but not the
    corresponding dimension in the other tensor, it is said that the operation
    has a sharding conflict. Hence, after this pass, the opeartions become
    conflict-free.

    This pass injects reshard operations explicitly so that, for each operation,
    corresponding dimensions become sharded in the same way across all operands
    and results, and every axis (or sub-axis) can only be used to shard a single
    dimension type.

    A clarifying example:

    Input:
    ```mlir
    mesh = <"x"=4, "y"=2>
    %lhs : tensor<8x32xf32> {sdy.sharding=<@mesh, \[{"y"},{"x"}\]>}
    %rhs : tensor<32x16xf32> {sdy.sharding=<@mesh, \[{"y"}, {"x"}\]>}
    stablehlo.dot %lhs, %rhs {sdy.sharding_per_value=<[<@mesh, \[{"x"}, {}\]>]>}
      : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
    ```

    Output:
    ```mlir
    sdy.mesh = <"x"=4, "y"=2>
    %lhs : tensor<8x32xf32> {sdy.sharding=<@mesh, \[{"x"}, {"y"}\]>}
    %rhs : tensor<32x16xf32> {sdy.sharding=<@mesh, \[{"y"}, {"x"}\]>}
    %0 = sdy.reshard %rhs <@mesh, \[{"y"}, {}\]> : tensor<32x16xf32>
    stablehlo.dot %lhs, %0 {sdy.sharding_per_value=<[<@mesh, \[{"x"}, {}\]>]>}
      : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
    ```

    In the example above, there is a conflict since `lhs` and `rhs` tensors
    are both sharded on axis "x" on their non-contracting dimensions. Here,
    `rhs` tensor is resharded, before the dot operation, explicitly to be
    sharded only on its first dimension and on axis "x". This way, the dot
    opearation becomes compatible.
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

def ShardingGroupCleanupPass : Pass<"sdy-sharding-group-cleanup", "ModuleOp"> {
  let summary = "Removes ShardingGroupOps after propagation.";
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}
