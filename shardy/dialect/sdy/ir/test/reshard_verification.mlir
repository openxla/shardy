// RUN: sdy_opt %s -split-input-file -verify-diagnostics

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Since ReshardOp::verify has the same verification as any TensorShardingAttr,
// there is no need to check different types of failures.
func.func @invalid_sharding(%arg0 : tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{sharding doesn't match tensor rank: 2 != 1}}
  %0 = sdy.reshard %arg0 <@mesh, [{}, {"b"}], replicated={"a"}> : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @constraint_sharding_inside_bound_manual_computation(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a",?}, {?}]>] out_shardings=[<@mesh, [{"a",?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) { // expected-note  {{parent bounding this axis as manual}}
    %1 = sdy.reshard %arg1 <@mesh, [{"a"}, {}]> : tensor<8x32xf32> // expected-error {{op operates on axis "a" which is already bound by a parent sdy.manual_computation op}}
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @constraint_replication_inside_bound_manual_computation(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a",?}, {?}]>] out_shardings=[<@mesh, [{"a",?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) { // expected-note  {{parent bounding this axis as manual}}
    %1 = sdy.reshard %arg1 <@mesh, [{}, {}], replicated={"a"}> : tensor<8x32xf32> // expected-error {{op operates on axis "a" which is already bound by a parent sdy.manual_computation op}}
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}
