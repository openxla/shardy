// RUN: sdy_opt %s -sdy-sharding-constraint-to-reshard | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @sharding_constraint_to_reshard
func.func @sharding_constraint_to_reshard(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {?}]> {foo} : tensor<8x8xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> {foo} :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// Verify that ShardingConstraintOp on a token converts to a 0-dimensional
// ReshardOp.
// CHECK-LABEL: func @token_sharding_constraint_to_reshard
func.func @token_sharding_constraint_to_reshard(%arg0: !stablehlo.token) -> !stablehlo.token {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, []> : !stablehlo.token
  %0 = sdy.sharding_constraint %arg0 <@mesh, []> : !stablehlo.token
  return %0 : !stablehlo.token
}
