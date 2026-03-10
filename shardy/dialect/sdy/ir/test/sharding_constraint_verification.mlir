// RUN: sdy_opt %s -split-input-file -verify-diagnostics

sdy.mesh @mesh = <["a"=2,"b"=2]>

// Since ShardingConstraintOp::verify has the same verification as any
// TensorShardingAttr, there is no need to check different types of failures.
func.func @invalid_sharding(%arg0 : tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{sharding doesn't match tensor rank: 2 != 1}}
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"b"}], replicated={"a"}> : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @constraint_sharding_inside_bound_manual_computation(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a",?}, {?}]>] out_shardings=[<@mesh, [{"a",?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) { // expected-note  {{parent bounding this axis as manual}}
    %1 = sdy.sharding_constraint %arg1 <@mesh, [{"a"}, {}]> : tensor<8x32xf32> // expected-error {{op operates on axis "a" which is already bound by a parent sdy.manual_computation op}}
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @constraint_replication_inside_bound_manual_computation(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a",?}, {?}]>] out_shardings=[<@mesh, [{"a",?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) { // expected-note  {{parent bounding this axis as manual}}
    %1 = sdy.sharding_constraint %arg1 <@mesh, [{}, {}], replicated={"a"}> : tensor<8x32xf32> // expected-error {{op operates on axis "a" which is already bound by a parent sdy.manual_computation op}}
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// ShardingConstraintOp accepts tokens, but tokens have no dimensions, so only
// rank-0 shardings (with no replicated/unreduced axes) are valid. Verify that a
// non-rank-0 sharding on a token is rejected.
func.func @token_sharding_constraint_rank_mismatch(%arg0: !stablehlo.token) -> !stablehlo.token {
  // expected-error @+1 {{non-shaped tensors can only have a sharding with rank 0 and no replicated or unreduced axes}}
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}]> : !stablehlo.token
  return %0 : !stablehlo.token
}
