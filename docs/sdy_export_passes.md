<!-- Autogenerated by mlir-tblgen; don't manually edit -->

### `-sdy-close-shardings`

_Closes tensor shardings and drops replicated axes._

### `-sdy-constant-or-scalar-merger`

_Merge identical constants and scalar expansions with matching shardings._

Performs a lightweight CSE on constants with identical shardings.

The import pipeline splits and duplicates the constants and scalar
expansions such that sharding is not propagated between different uses of a
constant sub-computation. If the constants have same shardings after
propagation, this pass merges them to save compilation time. See
-sdy-constant-or-scalar-splitter for more info.

### `-sdy-drop-sharding-rules`

_Drops `OpShardingRuleAttr` from all registered ops._

### `-sdy-insert-explicit-reshards`

_Inserts explicit reshards to make all operations have compatible shardings._

A compatible sharding essentially means that the operation can accept the
sharded operands and produce a sharded result without requiring any reshard
communications (note that the operation might still require communication
such as all-reduce or halo-swaps).

After propagation, some operations may still have incompatible shardings.

Note that when an axis (or sub-axis) is used to shard non-corresponding
dimensions (e.g. non-contracting dimensions in matmul) across multiple
tensors, or when an axis shards a dimension in one tensor but not the
corresponding dimension in the other tensor, it is said that the operation
has a sharding conflict. Hence, after this pass, the operations become
conflict-free.

This pass injects reshard operations explicitly so that, for each operation,
corresponding dimensions become sharded in the same way across all operands
and results, and every axis (or sub-axis) can only be used to shard a single
dimension type.

Example:

Input:
```mlir
mesh = <"x"=4, "y"=2>
%lhs : tensor<8x32xf32> {sdy.sharding=<@mesh, \[{"x"}, {"y"}\]>}
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

In the example above, `lhs` and `rhs` are both sharded on axis "x" on their
non-contracting dimensions, which is incompatible. The pass inserts an
explicit reshard on `rhs` before the dot operation, so that the dot
operation has compatible shardings.

### `-sdy-remove-propagation-debug-info`

_Removes propagation debug info (propagation edges and origin shardings) during export._

### `-sdy-remove-sharding-groups`

_Removes ShardingGroupOps after propagation._

### `-sdy-reshard-to-collectives`

_Converts ReshardOp into various Shardy collective ops._

Matches reshard ops and rewrites them into various Shardy collective
ops. After this pass, no reshard ops remain in the module. This pass assumes
that explicit reshards have already been inserted
(`sdy-insert-explicit-reshards`).

Example:

Input:
```mlir
mesh = <"x"=2, "y"=2, "z"=2>
%0 : tensor<16x2xf32> {sdy.sharding<@mesh, \[{"x", "y", "z"}, {}\]>
%1 = sdy.reshard %arg0 <@mesh, \[{"x"}, {}\]> : tensor<16x2xf32>
```

Output:
```mlir
mesh = <"x"=2, "y"=2, "z"=2>
%0 : tensor<16x2xf32> {sdy.sharding<@mesh, \[{"x", "y", "z"}, {}\]>
%1 = sdy.all_gather  \[{"y", "z"}, {}\] %arg0 out_sharding=<@mesh, \[{"x"}, {}\]> : tensor<16x2xf32>
```

In the example above, the tensor `%0 : tensor<16x2xf32>` is sharded as
`\[{"x", "y", "z"}, {}\]`. Then, there's a `reshard` op resharding it as
`\[{"x"}, {}\]`. On the first axes, since the suffix `{"y", "z"}` is removed
after the reshard, we infer that we have all-gathered `{"y", "z"}`. The
second dimension is not changed.


### `-sdy-sharding-constraint-to-reshard`

_Converts ShardingConstraintOp into ReshardOp._

### `-sdy-sink-data-flow-edges`

_Sinks all `DataFlowEdgeOp` into their input._

Moves the sharding of each `DataFlowEdgeOp` to its input (the root target of
the edge), and replaces the op with its input.

#### Options

```
-sink-debug-sharding-origins          : Whether to sink the debug sharding origins info. See `debug-sharding-origins` option in propagation for more info.
-sink-debug-propagation-edge-sharding : Whether to sink the debug propagation edge sharding info. See `debug-propagation-edge-sharding` option in propagation for more info.
```

### `-sdy-temp-explicit-reshards-for-optimizations`

_Inserts explicit reshards for specific optimizations._

This pass is a temporary solution until we can enable the
`sdy-insert-explicit-reshards` pass by default.

It allows us to insert explicit reshards on specific operations for
optimizations.

### `-sdy-update-non-divisible-input-output-shardings`

_Makes FuncOp inputs/outputs evenly sharded, removing any need for padding due to non-divisible shardings._

Users of Shardy expect the function inputs/outputs to be evenly
divisible/shardable to avoid requiring padding their tensors. Propagation
may make inputs/outputs have non-divisible shardings, so this pass updates
them to the largest dimension sharding prefix of the original sharding that
is evenly sharded.
